# x10 prefill-fixed copy of the round-7 /tmp/prefillprof/prefill_prof.py (pod-local; PFX_TAG labels the arm).
# Pod-local profiling harness (NOT committed). The production MTP serving prefill on ONE TP8 row: the vLLM class
# (Qwen36MTPContractForCausalLM) built with the plugin's arguments (max_batch_size 8, max_seq_len 262144, KV pool
# QWEN36_MAX_TOKENS_ALL_USERS tokens), warmed in the plugin's order (eager prefill, eager decode, traced prefill, traced
# decode), then prefill_forward called exactly as vllm_tt_plugin's submit_prefill calls it (tokens = the whole prefix up
# to the chunk end, prompt_lens = chunk end, start_pos = cached/computed tokens, intermediate_prefill_mask for non-final
# chunks, empty_slots). Device params = production (1 CQ, 1 GiB trace region, l1_small 24576, FABRIC_1D_RING, 8 KB payload).
#
#   PP_MODE   timing   unperturbed: one synchronize at the end of every prefill_forward call
#             phases   every wrapped step (trace replays, eager MTP warm, snapshot save/restore, slot write, readbacks, H2D
#                      staging, host planning) bracketed by synchronize_device: exclusive wall time per step
#             profile  device profiler (run under `python -m tracy -r -p`): ReadDeviceProfiler after every call; the trace
#                      replays of every call are logged (trace id order) so the report's replay sessions map to calls
#             join     B-1 rows decoding in the production deferred/chained fused mode (depth-1 async pipeline, prefill over
#                      the pending step) + one joiner: per-step wall, and the phases of the join step when PP_JOIN_PHASES=1
#   PP_SCEN   comma list of scenarios (see SCENARIOS); default: all of the mode's defaults
#   PP_OUT    jsonl output (default /tmp/prefillprof/res.jsonl); PP_REPS repetitions of the short scenario (default 3)
#   PP_WORKER_L1_SIZE  worker_l1_size for profiled runs (the instrumented gdn_spec_step needs 1 KB more kernel-config ring)
import collections
import json
import os
import random
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.tp_common import fabric_router_config

B = 8
BS = 64
MAX_SEQ = int(os.environ.get("PP_MAX_SEQ", "262144"))
POOL = int(os.environ.get("QWEN36_MAX_TOKENS_ALL_USERS", "525312"))
MODE = os.environ.get("PP_MODE", "timing")
OUT = os.environ.get("PP_OUT", "/tmp/pfx/res/prof.jsonl")
TAG = os.environ.get("PFX_TAG", "")
PROF = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
REPS = int(os.environ.get("PP_REPS", "3"))
_WL1 = os.environ.get("PP_WORKER_L1_SIZE")
_ROUTER = fabric_router_config()
DEVICE_PARAMS = [
    {
        "l1_small_size": 24576,
        "num_command_queues": 1,
        "trace_region_size": 1 << 30,
        "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        **({"fabric_router_config": _ROUTER} if _ROUTER is not None else {}),
        **({"worker_l1_size": int(_WL1)} if _WL1 else {}),
    }
]

_SYNC = ttnn.synchronize_device  # the real one (wrappers below must not time themselves)
_EXEC = ttnn.execute_trace


def chunks(L, cap, start=0):
    """vLLM scheduler chunks of a prompt of L tokens from `start` (cap tokens each; ends 128-aligned since cap is)."""
    out, s = [], start
    while s < L:
        e = min(L, s + cap)
        out.append((s, e, e >= L))
        s = e
    return out


def prod_chunks(L, start=0):
    """Production cap table [[0, 16384], [131072, 8192]] (by the depth the chunk starts at)."""
    out, s = [], start
    while s < L:
        cap = 16384 if s < 131072 else 8192
        e = min(L, s + cap)
        out.append((s, e, e >= L))
        s = e
    return out


SCENARIOS = {
    # single-call prompts (one vLLM step: everything <= 16K is one chunk in production)
    "short": [[(0, L, True)] for L in [int(x) for x in os.environ.get("PP_LENS", "128,181,512,2048,8192").split(",")] for _ in range(REPS)],
    "p32k_prod": [prod_chunks(32768)] * 2,  # production: [0,16K) non-final, [16K,32K) final
    "p32k_8k": [chunks(32768, 8192)] * 2,  # 8K steps: offsets 0/8K/16K non-final, 24K final
    "p40k_8k": [chunks(40960, 8192)] * 2,  # clean non-final 8K chunks at 0/8K/16K/24K/32K + final at 32K
    "p64k_prod": [prod_chunks(65536)],
    "p64k_8k": [chunks(65536, 8192)],
    "p128k_prod": [prod_chunks(131072)],
    "resume": "special",
}
DEFAULTS = {
    "timing": "short,p32k_prod,p32k_8k,p40k_8k,p64k_prod,p64k_8k,p128k_prod,resume",
    "phases": "short,p32k_prod,p40k_8k,p64k_prod,resume",
    "profile": "p40k_8k,short1,p32k_prod",
}
SCENARIOS["short1"] = [[(0, L, True)] for L in (64, 2048, 4096)]
SCENARIOS["prof8k"] = [chunks(40960, 8192)[:4]]  # non-final 8K chunks at 0/8K/16K/24K (request released after the 4th)
SCENARIOS["short64"] = [[(0, 64, True)]]
SCENARIOS["call8k"] = [[(0, 8192, False)]]  # one non-final 8K chunk from 0 (big replay + drafter warm + saves), released


def _load():
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except Exception:  # noqa: BLE001
        return None


def emit(rec):
    rec = dict(rec, pfx_tag=TAG, ts=round(time.time(), 3), loadavg=_load())
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    logger.info(f"[PP] {json.dumps(rec)}")


_WARMUP_READS = os.environ.get("PP_PROF_WARMUP_READS", "0") == "1"


def _read_prof(mesh, tag, warmup=False):
    if warmup and not _WARMUP_READS:
        return
    if PROF:
        _SYNC(mesh)
        t0 = time.time()
        ttnn.ReadDeviceProfiler(mesh)
        logger.info(f"[PP] ReadDeviceProfiler after {tag}: {time.time() - t0:.2f}s")


class Tracker:
    """Wraps model / engine / ttnn callables. phases=True: sync before and after every wrapped call, exclusive wall per
    label (nested wrapped calls subtracted). Always: the trace replays of the current call (label per trace id)."""

    def __init__(self, mesh, phases):
        self.mesh = mesh
        self.phases = phases
        self.on = False
        self.stack = []
        self.acc = collections.defaultdict(float)
        self.cnt = collections.Counter()
        self.replays = []
        self.trace_names = {}
        self._undo = []

    def reset(self):
        self.acc.clear()
        self.cnt.clear()
        self.replays = []

    def _enter(self):
        _SYNC(self.mesh)
        self.stack.append([time.perf_counter(), 0.0])

    def _exit(self, label):
        _SYNC(self.mesh)
        t0, child = self.stack.pop()
        dt = time.perf_counter() - t0
        self.acc[label] += dt - child
        self.cnt[label] += 1
        if self.stack:
            self.stack[-1][1] += dt

    def wrap(self, obj, name, label=None):
        orig = getattr(obj, name)
        label = label or name
        trk = self

        def w(*a, **k):
            if not (trk.on and trk.phases):
                return orig(*a, **k)
            trk._enter()
            try:
                return orig(*a, **k)
            finally:
                trk._exit(label)

        setattr(obj, name, w)
        self._undo.append((obj, name, orig))

    def install_ttnn(self):
        trk = self

        def exec_trace(dev, tid, *a, **k):
            name = trk.trace_names.get(int(tid), f"other{int(tid)}")
            if trk.on:
                trk.replays.append(name)
            if not (trk.on and trk.phases):
                return _EXEC(dev, tid, *a, **k)
            trk._enter()
            try:
                return _EXEC(dev, tid, *a, **k)
            finally:
                trk._exit("replay:" + name)

        ttnn.execute_trace = exec_trace
        self._undo.append((ttnn, "execute_trace", _EXEC))
        for fname, label in (
            ("copy_host_to_device_tensor", "h2d_copy"),
            ("from_torch", "from_torch"),
            ("to_torch", "to_torch(readback)"),
            ("copy", "ttnn.copy(eager)"),
        ):
            self.wrap(ttnn, fname, label)

    def uninstall(self):
        for obj, name, orig in reversed(self._undo):
            setattr(obj, name, orig)
        self._undo = []


def build(mesh):
    from transformers import AutoConfig

    from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36MTPContractForCausalLM as C

    t0 = time.time()
    hf = AutoConfig.from_pretrained(os.environ["HF_MODEL"], trust_remote_code=True)
    gen = C.initialize_vllm_model(hf, mesh, B, max_seq_len=MAX_SEQ)
    model = gen.model[0]
    nblk = POOL // BS
    shape = [nblk, model.args.n_local_kv_heads, BS, model.args.head_dim]
    kv = gen.allocate_kv_cache(shape, torch.bfloat16, 16)
    t1 = time.time()
    # Plugin warmup order (vllm_tt_plugin model_runner.warmup_model). The vision tower compile is skipped (text only).
    gen.warmup_vision = lambda *a, **k: None
    num_blocks = -(-MAX_SEQ // BS)
    gen.warmup_model_prefill(kv_cache=kv, enable_trace=False, can_sample_on_device=False)
    _read_prof(mesh, "warmup prefill eager", warmup=True)
    gen.warmup_model_decode(
        kv_cache=kv, enable_trace=False, max_batch_size=B, num_blocks=num_blocks, can_sample_on_device=False
    )
    _read_prof(mesh, "warmup decode eager", warmup=True)
    gen.warmup_model_prefill(kv_cache=kv, enable_trace=True, can_sample_on_device=False)
    _read_prof(mesh, "warmup prefill traced", warmup=True)
    gen.warmup_model_decode(
        kv_cache=kv, enable_trace=True, max_batch_size=B, num_blocks=num_blocks, can_sample_on_device=False
    )
    _read_prof(mesh, "warmup decode traced", warmup=True)
    emit(dict(kind="setup", load_s=round(t1 - t0, 1), warmup_s=round(time.time() - t1, 1), nblk=nblk, max_seq=MAX_SEQ))
    return gen, model, kv, nblk


def trace_names(model, eng):
    names = {}
    if getattr(model, "_chunked_trace_id", None) is not None:
        names[int(model._chunked_trace_id)] = f"chunk{model._chunked_chunk_size}"
    big = (getattr(model, "_chunk_sets", {}) or {}).get("big")
    if big and big.get("_chunked_trace_id") is not None:
        names[int(big["_chunked_trace_id"])] = f"chunk{big['_chunked_chunk_size']}"
    for b, tr in (getattr(model, "_short_traces", {}) or {}).items():
        if getattr(tr, "trace_id", None) is not None:
            names[int(tr.trace_id)] = f"short{b}"
    for attr in ("_gdn_slot_write_traces",):
        d = getattr(model, attr, None)
        if isinstance(d, dict):
            for k, v in d.items():
                tid = v[0] if isinstance(v, (tuple, list)) else v
                try:
                    names[int(tid)] = f"slotwrite{k}"
                except Exception:  # noqa: BLE001
                    pass
    for attr in ("_mtpw_trace", "_fa_trace", "_fs_trace", "_ch_trace_g", "_ch_trace_s"):
        v = getattr(eng, attr, None)
        if v is not None:
            try:
                names.setdefault(int(v), attr.strip("_"))
            except Exception:  # noqa: BLE001
                pass
    for i, v in enumerate(getattr(eng, "_seed_trace", None) or []):
        if v is not None:
            names.setdefault(int(v), f"seed_trace{i}")
    return names


class Requests:
    """Unique token streams and disjoint page-table block ranges (block 0 = vLLM null block, last = MTP scratch)."""

    def __init__(self, nblk, seed=1234):
        self.rng = random.Random(seed)
        self.next_blk = 1
        self.nblk = nblk

    def tokens(self, L, base=None):
        t = torch.tensor([self.rng.randrange(1000, 150000) for _ in range(L)], dtype=torch.int32)
        if base is not None:
            t[: base.numel()] = base[: min(L, base.numel())]
        return t

    def blocks(self, L):
        n = -(-(L + 8) // BS) + 1
        if self.next_blk + n >= self.nblk - 2:
            self.next_blk = 1
        out = list(range(self.next_blk, self.next_blk + n))
        self.next_blk += n
        return out


def run_call(gen, kv, mesh, toks, blocks, s, e, final, slot, trk, tag):
    tokens = toks[:e].reshape(1, -1)
    pt = torch.tensor([blocks], dtype=torch.int32)
    kw = dict(start_pos=[s], empty_slots=[slot])
    if not final:
        kw["intermediate_prefill_mask"] = [True]
    eng = gen._engine()
    fb0 = getattr(eng, "resume_fallbacks", 0)
    pc0 = mesh.num_program_cache_entries()
    trk.reset()
    trk.on = True
    t0 = time.perf_counter()
    lg, _ = gen.prefill_forward(tokens, pt, kv, [e], **kw)
    t1 = time.perf_counter()
    _SYNC(mesh)
    t2 = time.perf_counter()
    trk.on = False
    rec = dict(
        kind="call",
        mode=MODE,
        tag=tag,
        start=s,
        end=e,
        n=e - s,
        final=final,
        call_ms=round(1e3 * (t1 - t0), 2),
        wall_ms=round(1e3 * (t2 - t0), 2),
        tok_s=round((e - s) / (t2 - t0), 1),
        replays=dict(collections.Counter(trk.replays)),
        replay_order=trk.replays,
        fallback=getattr(eng, "resume_fallbacks", 0) - fb0,
        new_programs=mesh.num_program_cache_entries() - pc0,
    )
    if final:
        import hashlib

        lv = lg.reshape(-1).float()
        rec["logits_md5"] = hashlib.md5(lv.numpy().tobytes()).hexdigest()[:16]
        rec["argmax"] = int(lv.argmax())
    if trk.phases:
        rec["phases_ms"] = {k: round(1e3 * v, 3) for k, v in sorted(trk.acc.items(), key=lambda x: -x[1])}
        rec["phase_counts"] = dict(trk.cnt)
        rec["phases_sum_ms"] = round(1e3 * sum(trk.acc.values()), 2)
    emit(rec)
    _read_prof(mesh, f"{tag} [{s},{e})")
    return rec


def run_request(gen, kv, mesh, reqs, calls, trk, tag, slot=0, toks=None):
    L = calls[-1][1]
    toks = reqs.tokens(L) if toks is None else toks
    blocks = reqs.blocks(L)
    out = []
    for s, e, final in calls:
        out.append(run_call(gen, kv, mesh, toks, blocks, s, e, final, slot, trk, tag))
    gen.release_request(slot)
    tot = sum(r["wall_ms"] for r in out)
    emit(dict(kind="request", tag=tag, L=L, calls=len(out), wall_ms=round(tot, 1), tok_s=round((L - calls[0][0]) / tot * 1e3, 1)))
    return out, toks, blocks


def run_resume(gen, kv, mesh, reqs, trk):
    """Multi-turn follow-ups. Turn 1 (L1 tokens, from 0); turn 2 = turn 1's prompt + new tokens, vLLM start_pos = the
    turn-1 prompt rounded down to a 64-token block (the follow-up shares the whole turn-1 prompt). Shows which snapshot the
    resume finds (plan) and what it costs. Also turn 2' with start_pos covering 512 response tokens (the snapshot gap)."""
    eng = gen._engine()
    for L1, extra in ((8492, 700), (5000, 600), (20000, 1500), (40000, 900)):
        toks1 = reqs.tokens(L1)
        blocks = reqs.blocks(L1 + extra + 600)
        for s, e, final in prod_chunks(L1):
            run_call(gen, kv, mesh, toks1, blocks, s, e, final, 0, trk, f"resume_t1_L{L1}")
        gen.release_request(0)
        toks2 = reqs.tokens(L1 + extra, base=toks1)
        sp = (L1 // BS) * BS
        cache = gen.model[0]._prefix_cache
        from models.demos.blackhole.qwen36.tt.prefix_cache import spec_resume_cap

        cap = spec_resume_cap(sp, L1 + extra)
        pos, _ = cache.lookup(toks2, cap)
        emit(dict(kind="resume_plan", L1=L1, start_pos=sp, cap=cap, snapshot=pos, positions=list(cache._positions)[-12:]))
        for s, e, final in prod_chunks(L1 + extra, start=sp):
            run_call(gen, kv, mesh, toks2, blocks, s, e, final, 0, trk, f"resume_t2_L{L1}")
        gen.release_request(0)


def phases_install(trk, gen, model, eng):
    trk.install_ttnn()
    for name in (
        "_prefill_fast",
        "_fast_segment",
        "_replay_spec_short",
        "_warm_mtp_chunk",
        "_capture_last_row",
        "_warm_resume_slot",
        "_read_replicated_logits",
        "_chunk_exact_resume",
        "_claim_low_row",
        "_note_resume_fallback",
    ):
        if hasattr(eng, name):
            trk.wrap(eng, name, "eng." + name)
    for name in (
        "_bind_gdn_prefill_scratch",
        "_unbind_gdn_prefill_scratch",
        "_build_request_rope",
        "plan_spec_prefix",
        "_prefill_traced_chunked_tp",
        "_replay_chunks",
        "prefill_masked_bucket",
        "_masked_bucket_logits_tp",
        "_write_gdn_slot_from_scratch",
        "_replay_short_trace",
        "spec_feed_rows",
        "ttnn_mtp_prefill_forward",
        "_reset_gdn_state_for_new_sequence",
        "_set_vision_merge",
        "_rope_tp_cos_sin_torch",
    ):
        if hasattr(model, name):
            trk.wrap(model, name, "model." + name)
    cache = getattr(model, "_prefix_cache", None)
    if cache is not None:
        for name in ("save", "restore", "lookup", "pin"):
            trk.wrap(cache, name, "prefix_cache." + name)
    for name in ("_table", "_ensure_ready"):
        trk.wrap(gen, name, "vllm." + name)


def device_replay_probe(model, eng, mesh, reqs, nblk):
    """Device time of ONE big-chunk replay at depth d (no host staging between replays): stage once, replay 3x
    back-to-back with only the chunk-start / page-table inputs, sync around. Bound GDN scratch; state is garbage."""
    big = model.big_chunk_rows()
    if not big:
        return
    L = 65536 + big
    toks = reqs.tokens(L)
    blocks = reqs.blocks(L)
    pt = torch.tensor([blocks], dtype=torch.int32)
    pt_rec = model._fit_spec_page_table(pt, eng.nb_prefill)
    prev = model._bind_gdn_prefill_scratch()
    try:
        depths = [int(x) for x in os.environ.get("PP_PROBE_DEPTHS", "0,8192,16384,24576,32768,49152,65536").split(",")]
        reps = int(os.environ.get("PP_PROBE_REPS", "3"))
        for d in depths:
            ts = []
            for rep in range(reps):
                with model._big_chunk_set():
                    rep_ = ttnn.ReplicateTensorToMesh(mesh)
                    buf_blocks = int(model._chunk_full_page_table_buf.shape[-1])
                    ptp = pt_rec
                    if ptp.shape[1] < buf_blocks:
                        ptp = torch.cat([ptp, torch.zeros(1, buf_blocks - ptp.shape[1], dtype=ptp.dtype)], dim=1)
                    else:
                        ptp = ptp[:, :buf_blocks]
                    pt_host = ttnn.from_torch(ptp, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=None, mesh_mapper=rep_)
                    _SYNC(mesh)
                    t0 = time.perf_counter()
                    model._replay_chunks(toks.reshape(1, -1), ptp, pt_host, d, 1, big, None, None, None)  # stage + replay
                    _SYNC(mesh)
                    if PROF:  # profiled: one replay per depth (bounded profiler buffer), timed with its staging
                        ts.append(1e3 * (time.perf_counter() - t0))
                        continue
                    t0 = time.perf_counter()
                    _EXEC(mesh, model._chunked_trace_id, cq_id=0, blocking=False)  # same inputs again: pure replay
                    _SYNC(mesh)
                    ts.append(1e3 * (time.perf_counter() - t0))
            emit(dict(kind="replay_probe", depth=d, rows=big, ms=[round(x, 2) for x in ts], tok_s=round(big / (min(ts) / 1e3), 1)))
            _read_prof(mesh, f"replay probe depth {d}")
    finally:
        model._unbind_gdn_prefill_scratch(prev)


def join_mode(gen, model, kv, mesh, eng, nblk):
    """Depth-1 deferred chained pipeline (the production async spec serving shape) with live rows, then a joiner prefill
    over the pending step (QWEN36_SPEC_PREFILL_OVERLAP order), via the vLLM class's prefill_forward (start_pos=[0]).
    Records the token-delivery gap of a live row across the join vs its steady interval."""
    from models.demos.blackhole.qwen36.tests import test_spec_continuous as tc
    from models.demos.blackhole.qwen36.tests.test_spec_async_engine import AsyncDriver

    NBR = int(os.environ.get("PP_JOIN_NB", "160"))  # blocks per row in the driver's tables (8K tokens)
    tc.B, tc.NB, tc.NUM_BLOCKS = B, NBR, min(nblk - 2, B * NBR + 64)
    os.environ["QWEN36_SPEC_CHAIN_AUDIT_EVERY"] = os.environ.get("QWEN36_SPEC_CHAIN_AUDIT_EVERY", "8")

    class ProdAsync(AsyncDriver):
        def admit(self, row, rid, prompt, max_new=400):
            assert self.reqs[row] is None
            req = {"rid": rid, "prompt": list(prompt), "P": len(prompt), "blocks": [], "proposal": [], "acc": 1, "max_new": max_new}
            table_row = self._blocks_for(req, len(prompt) - 1)
            toks = torch.tensor([prompt], dtype=torch.int32)
            logits, _ = gen.prefill_forward(toks, table_row[None], kv, [len(prompt)], start_pos=[0], empty_slots=[row])
            req["out"] = [int(logits.reshape(-1).argmax())]
            self.reqs[row] = req

    rng = random.Random(7)
    ctx = int(os.environ.get("PP_JOIN_CTX", "2048"))
    widths = [int(x) for x in os.environ.get("PP_JOIN_WIDTHS", "7,3,1").split(",")]
    lens = [int(x) for x in os.environ.get("PP_JOIN_LENS", "128,512,2048,8192").split(",")]
    trk = Tracker(mesh, phases=os.environ.get("PP_JOIN_PHASES", "0") == "1")
    if trk.phases:
        phases_install(trk, gen, model, eng)
        for name in ("verify", "_seed_fresh_rows", "_join", "_warm_mtp_last_rows", "_replay_mtp_warm", "_decode_forward_deferred",
                     "propose_draft_tokens", "_chain_drop", "_def_finish_any", "_reconcile_fresh_rows"):
            if hasattr(eng, name):
                trk.wrap(eng, name, "eng." + name)
    else:
        trk.install_ttnn()  # replay log only
    trk.trace_names = trace_names(model, eng)
    rid = 1000
    for w in widths:
        for L in lens:
            for rep in range(int(os.environ.get("PP_JOIN_REPS", "3"))):
                drv = ProdAsync(eng, seed=rep + 17, sampled=lambda s: False)
                for u in range(w):
                    drv.admit(u, rid, [rng.randrange(1000, 150000) for _ in range(ctx)], max_new=100000)
                    rid += 1
                deliver = collections.defaultdict(list)  # row -> resolve times
                orig_resolve = drv.resolve

                def resolve(drv=drv, deliver=deliver, orig=orig_resolve):
                    rows = drv.pend[0][1] if drv.pend else []
                    orig()
                    t = time.perf_counter()
                    for r in rows:
                        deliver[r].append(t)

                drv.resolve = resolve
                steady = 30
                for i in range(steady):
                    drv.enq()
                    if len(drv.pend) == 2:
                        drv.resolve()
                # join over the pending step (production order with QWEN36_SPEC_PREFILL_OVERLAP=1)
                jrow = w
                prompt = [rng.randrange(1000, 150000) for _ in range(L)]
                trk.reset()
                trk.on = True
                pc0 = mesh.num_program_cache_entries()
                tj0 = time.perf_counter()
                drv.admit(jrow, rid, prompt, max_new=100000)
                tj1 = time.perf_counter()
                drv.resolve()
                tj2 = time.perf_counter()
                drv.host_rows.add(jrow)
                drv.chain_on_tip = True
                rid += 1
                step_ms = []
                for i in range(12):
                    t0 = time.perf_counter()
                    drv.enq()
                    if len(drv.pend) == 2:
                        drv.resolve()
                    step_ms.append(round(1e3 * (time.perf_counter() - t0), 2))
                    if i == 1:
                        trk.on = False
                trk.on = False
                drv.drain()
                d0 = deliver[0]
                gaps = [b - a for a, b in zip(d0, d0[1:])]
                base = statistics.median(gaps[5:steady - 2]) if len(gaps) > 10 else None
                jgap = max(b - a for a, b in zip(d0, d0[1:]) if b >= tj0) if d0 else None
                rec = dict(
                    kind="join", width=w, L=L, rep=rep, ctx=ctx,
                    itl_ms=round(1e3 * base, 2) if base else None,
                    max_gap_ms=round(1e3 * jgap, 2) if jgap else None,
                    stall_ms=round(1e3 * (jgap - base), 2) if (jgap and base) else None,
                    prefill_call_ms=round(1e3 * (tj1 - tj0), 2),
                    resolve_pending_ms=round(1e3 * (tj2 - tj1), 2),
                    steps_after_ms=step_ms,
                    replays=dict(collections.Counter(trk.replays)),
                    new_programs=mesh.num_program_cache_entries() - pc0,
                )
                if trk.phases:
                    rec["phases_ms"] = {k: round(1e3 * v, 3) for k, v in sorted(trk.acc.items(), key=lambda x: -x[1])}
                    rec["phase_counts"] = dict(trk.cnt)
                emit(rec)
                for r in list(drv.live()):
                    drv._finish(r)
    trk.uninstall()


@pytest.mark.timeout(10800)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_prefill_prof(mesh_device):
    mesh = mesh_device
    mesh.enable_program_cache()
    flags = {k: v for k, v in os.environ.items() if k.startswith(("QWEN", "VLLM_TT", "TT_METAL_PROF"))}
    emit(dict(kind="env", mode=MODE, flags=flags))
    gen, model, kv, nblk = build(mesh)
    eng = gen._engine()
    if MODE == "join":
        join_mode(gen, model, kv, mesh, eng, nblk)
        return
    trk = Tracker(mesh, phases=(MODE == "phases"))
    if MODE == "phases":
        phases_install(trk, gen, model, eng)
    else:
        trk.install_ttnn()  # replay log only (wrappers are pass-through without phases)
    trk.trace_names = trace_names(model, eng)
    emit(dict(kind="trace_names", names={str(k): v for k, v in trk.trace_names.items()}))
    reqs = Requests(nblk)
    if MODE == "profile":
        _read_prof(mesh, "flush before the measured calls")  # bounded: at most one device buffer of warmup markers
    else:
        # settle: one 4K and one 16K prompt (first-call effects, host caches)
        run_request(gen, kv, mesh, reqs, [(0, 4096, True)], trk, "settle4k")
        run_request(gen, kv, mesh, reqs, prod_chunks(20480), trk, "settle20k")
    scen = [s for s in os.environ.get("PP_SCEN", DEFAULTS.get(MODE, "")).split(",") if s]
    for name in scen:
        if name == "probe":
            device_replay_probe(model, eng, mesh, reqs, nblk)
            continue
        if name == "resume":
            run_resume(gen, kv, mesh, reqs, trk)
            continue
        for calls in SCENARIOS[name]:
            run_request(gen, kv, mesh, reqs, calls, trk, name)
    c = getattr(model, "_prefix_cache", None)
    if c is not None:
        emit(dict(kind="cache_summary", summary=c.summary(), stats=dict(c.stats)))
    trk.uninstall()
