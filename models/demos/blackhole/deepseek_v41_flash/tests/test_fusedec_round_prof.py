# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-op device profile of ONE decode round of the shipped speculative runner (traces M = mono, A = verify, S = sampler, B = tail: accept + commit + drafter) on the 4x8 galaxy.

The model is built like the demo (``create_tt_model``) on a layer subset (FD_LAYERS, default one layer of every type) or the full 40 layers; the spec runner of verification length FD_K is
captured (compile pass + traces of its position buckets) with the OpRecorder armed: every ttnn call of a captured trace is recorded with its caller and the device-operation id range
(== GLOBAL CALL COUNT >> 10 of the device profiler digest), plus phase marks (embedding, engramN, layerN, head, sampler, accept/commit/drafter). The traces are then replayed with the
device profiler read after every replay (TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1): the digest rows of a replay carry the kernel duration of every program on
every chip. Offline join and tables: tools/fusedec_roundtab.py.

Env: FD_LAYERS (comma list / a-b ranges), FD_B (users, 4 / 8 / 16), FD_K (3 / 5), FD_MAX_CTX (build context, default 131584), FD_BASE_POS (0: short positions, dense bucket; else the block
positions BASE_POS + (0, 7, 13, 21, ...): a long-context window with the decode indexer ON), FD_BUCKET (position bucket list of the runner, default "1024" short / "<BASE_POS rounded>" long),
FD_OUT (output dir), FD_REPS (profiled replays per trace, default 3), FD_WALL (1: unprofiled replay wall time of every trace first), FD_ROUTES (comma list of routes.pt of
dsv4-logs/moe_place/runs/*: the router output of every MoE layer is replaced by real routes, users x consecutive steps), FD_ROUTE_SEED, FD_ONLY (comma list of traces to profile, default M,A,S,B).
"""

import json
import os
import shutil
import statistics
import sys
import time

import numpy as np
import pytest
import torch

_FW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 5)))
if _FW_ROOT not in sys.path:
    sys.path.insert(0, _FW_ROOT)

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.op_table_recorder import OpRecorder
from models.demos.blackhole.deepseek_v41_flash.tt.common import create_tt_model, default_page_params
from models.demos.blackhole.deepseek_v41_flash.tt.generator import Generator
from models.tt_transformers.tt.common import PagedAttentionConfig


def _layers(s):
    out = []
    for p in s.split(","):
        if "-" in p:
            a, b = p.split("-")
            out += list(range(int(a), int(b) + 1))
        elif p.strip():
            out.append(int(p))
    return out


LAYERS = _layers(os.environ.get("FD_LAYERS", "0,1,2,3,14,20,21,24"))
B = int(os.environ.get("FD_B", "8"))
K = int(os.environ.get("FD_K", "3"))
MAX_CTX = int(os.environ.get("FD_MAX_CTX", "131584"))
BASE_POS = int(os.environ.get("FD_BASE_POS", "0"))
OUT = os.environ.get("FD_OUT", "/mnt/tt-data/ssinghal/dsv4-logs/fusedec/run_default")
REPS = int(os.environ.get("FD_REPS", "3"))
WALL = os.environ.get("FD_WALL", "1") == "1"
ROUTES = [p for p in os.environ.get("FD_ROUTES", "").split(",") if p]
ONLY = os.environ.get("FD_ONLY", "M,A,S,B").split(",")
GSM = "models/demos/blackhole/deepseek_v41_flash/demo/sample_prompts/input_data_gsm8k_128.json"

PARAMS = [
    pytest.param(
        {
            "l1_small_size": 16384,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "trace_region_size": int(os.environ.get("DSV41_TRACE_REGION", "1200000000")),
        },
        id="ring",
    )
]


def _log(m):
    print(m, flush=True)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# capture bookkeeping
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class Cap:
    """trace id -> (row range of the recorder, mark range); the recorder is armed between begin / end of every trace capture."""

    def __init__(self, rec):
        self.rec, self.traces, self.cur = rec, {}, None
        ob, oe = ttnn.begin_trace_capture, ttnn.end_trace_capture
        cap = self

        def begin(*a, **k):
            cap.cur = (len(rec.rows), len(rec.marks))
            rec.enabled, rec.depth = True, 0
            return ob(*a, **k)

        def end(md, tid, *a, **k):
            r = oe(md, tid, *a, **k)
            rec.enabled = False
            if cap.cur is not None:
                n0, m0 = cap.cur
                for row in rec.rows[n0:]:
                    row["tid"] = int(tid)
                cap.traces[int(tid)] = dict(rows=[n0, len(rec.rows)], marks=[m0, len(rec.marks)])
            cap.cur = None
            return r

        ttnn.begin_trace_capture, ttnn.end_trace_capture = begin, end


def _wrap_marks(rec, dec):
    def wrap(obj, attr, name):
        f = getattr(obj, attr)

        def w(*a, **k):
            if rec.enabled:
                rec.mark(name)
            return f(*a, **k)

        setattr(obj, attr, w)

    wrap(dec.embedding, "forward", "embedding")
    wrap(dec.head, "forward", "head")
    if hasattr(dec.head, "sample"):
        wrap(dec.head, "sample", "head_sample")
    f_eng = dec._engram_fwd

    def eng(lid, x, *a, **k):
        if rec.enabled:
            rec.mark(f"engram{lid}")
        return f_eng(lid, x, *a, **k)

    dec._engram_fwd = eng
    wrap(dec, "forward_sampler", "sampler")
    wrap(dec, "_accept_commit_draft", "accept_commit_draft")
    wrap(dec, "commit", "commit")
    for nm in ("write_main_full", "draft_full"):
        if hasattr(dec.drafter, nm):
            wrap(dec.drafter, nm, "drafter_" + nm)
    for lid, layer, _ in dec.layers:
        f = layer.forward

        def lw(*a, _f=f, _l=lid, **k):
            if rec.enabled:
                rec.mark(f"layer{_l}")
            return _f(*a, **k)

        layer.forward = lw


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# real routing (optional)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def _load_routes(paths):
    arrs = []
    for p in paths:
        r = torch.load(p)
        a = np.stack([np.stack([t.numpy() for t in st]) for st in r]).transpose(1, 0, 2, 3)  # [L, S, B, 6]
        arrs.append(a)
    S = min(a.shape[1] for a in arrs)
    return np.concatenate([a[:, :S] for a in arrs], axis=2)


def _inject_routes(md, dec, R, U, n, seed):
    """every MoE layer: the router still runs (its op stays in the trace) but moe.forward gets real routes (users x consecutive steps) of the layer: [4T,6] ids, T = U*n rows per mesh row."""
    rows, cols = md.shape
    T = U * n
    rng = np.random.default_rng(seed)
    S, Bu = R.shape[1], R.shape[2]
    mapper = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols))
    held = []
    for lid, layer, _ in dec.layers:
        Rl = R[lid]
        us = rng.choice(Bu, size=min(rows * U, Bu), replace=False)
        s0 = int(rng.integers(0, S - n + 1))
        ids = Rl[s0 : s0 + n][:, us].transpose(1, 0, 2).reshape(-1, 6)  # user-major rows
        ids = torch.from_numpy(np.ascontiguousarray(ids[: rows * T])).to(torch.int32).reshape(rows * T, 1, 1, 6)
        sc = torch.rand(rows * T, 6) + 0.1
        sc = (sc / sc.sum(1, keepdim=True)).to(torch.bfloat16).reshape(rows * T, 1, 1, 6)
        t_i = ttnn.from_torch(
            ids,
            device=md,
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        t_s = ttnn.from_torch(
            sc,
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        held.append((t_i, t_s))
        moe = layer.moe
        orig = moe.forward

        def fwd(h, h_tok, forced_routing=None, fold=None, _o=orig, _m=moe, _s=t_s, _i=t_i):
            _m.gate.forward(h)  # keeps the router program in the graph (its time is part of the round)
            return _o(h, h_tok, (_s, _i), fold=fold) if fold is not None else _o(h, h_tok, (_s, _i))

        moe.forward = fwd
    return held


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def _setup(md, rec):
    from models.demos.blackhole.deepseek_v41_flash.tt import spec_policy

    pp = default_page_params(MAX_CTX, B // 4)
    args, m, _, _ = create_tt_model(
        md,
        B,
        MAX_CTX,
        PagedAttentionConfig(block_size=pp["page_block_size"], max_num_blocks=pp["page_max_num_blocks_per_dp"]),
        layer_ids=LAYERS,
        log=_log,
    )
    gen = Generator([m], [args], md, tokenizer=args.tokenizer)
    gen.spec_choice = spec_policy.resolve_apply(B, md.shape[0], m.max_ctx)
    assert gen.spec_choice.on, f"needs speculative decoding: {gen.spec_choice.describe()}"
    gen.enable_spec()
    spec = gen.spec
    runner = spec.runners[K] if hasattr(spec, "runners") else spec
    # real text: B windows of the GSM8K prompts
    X = None
    try:
        prompts = json.load(open(GSM))
        texts = [p["prompt"] if isinstance(p, dict) else str(p) for p in prompts]
        rows = []
        for b in range(B):
            ids = args.tokenizer.encode(texts[b % len(texts)])
            off = 24 + 5 * b
            w = ids[off : off + 6]
            assert len(w) == 6
            rows.append(w)
        X = torch.tensor(rows)
    except Exception as e:  # noqa
        _log(f"FD tokens: fallback to random ids ({type(e).__name__}: {e})")
        X = torch.randint(100, 100000, (B, 6), generator=torch.Generator().manual_seed(0))
    base = (
        torch.tensor([68 + 11 * b for b in range(B)])
        if not BASE_POS
        else BASE_POS + torch.tensor([7 * b for b in range(B)])
    )
    for b in range(B):
        m.pool.release(b)
        m.pool.admit(b, int(base[b]) + 64)
    m.pool.sync_page_table()
    return gen, runner, X, base


def _csv_copy(tag):
    prof = os.environ.get("TT_METAL_PROFILER_DIR")
    if not prof:
        return
    src = os.path.join(prof, ".logs", "cpp_device_perf_report.csv")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(OUT, f"{tag}_cpp.csv"))
    else:
        _log(f"FD no digest at {src}")


def _wall(md, fn, reps=20, outer=3):
    for _ in range(3):
        fn()
    ttnn.synchronize_device(md)
    ms = []
    for _ in range(outer):
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        ttnn.synchronize_device(md)
        ms.append((time.perf_counter() - t) / reps * 1e3)
    return statistics.median(ms)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", PARAMS, indirect=True)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_fusedec_round_prof(mesh_device):
    md = mesh_device
    os.makedirs(OUT, exist_ok=True)
    rec = OpRecorder()
    rec.install()
    cap = Cap(rec)
    gen, runner, X, base = _setup(md, rec)
    dec = runner.dec
    _wrap_marks(rec, dec)
    held = None
    if ROUTES:
        held = _inject_routes(
            md, dec, _load_routes(ROUTES), runner.U, runner.n, int(os.environ.get("FD_ROUTE_SEED", "7"))
        )
        _log(f"FD real routes injected from {ROUTES}")
    Xw = torch.cat([X, X[:, :1].repeat(1, 8)], dim=1)[:, :8]  # >= n columns for every runner of the set
    master = gen.spec.seedr if hasattr(gen.spec, "seedr") else runner
    if getattr(master, "siblings", None):
        master._capture_all(lambda nn: Xw[:, :nn], base)
    else:
        runner.capture(Xw[:, : runner.n], base)
    Xr = Xw[:, : runner.n]
    snaps = dec.snapshot_states()
    runner._ensure(base)
    runner._use(base)
    dec.set_force(runner.n - 1)
    runner._feed(Xr, base)
    tids = dict(M=runner.tid, A=runner.tid_a, B=runner.tid_b)
    meta = dict(
        layers=LAYERS,
        B=B,
        k=K,
        n=runner.n,
        U=runner.U,
        T=runner.U * runner.n,
        base=base.tolist(),
        bucket=str(runner.cur_bucket),
        tids={k: int(v) for k, v in tids.items()},
        sampler_tids={str(f): int(t) for f, t in runner.tid_s.tids.items()},
        routes=ROUTES,
    )
    # ---- op rows and marks of the captured traces
    json.dump(
        dict(rows=rec.rows, marks=rec.marks, traces=cap.traces, meta=meta), open(os.path.join(OUT, "oprows.json"), "w")
    )
    _log(f"FD captured: {len(rec.rows)} recorded calls, traces {cap.traces.keys()} meta {meta}")

    def replay(name):
        if name == "S":
            return lambda: runner.tid_s.replay()
        return lambda: ttnn.execute_trace(md, tids[name], cq_id=0, blocking=False)

    # ---- unprofiled wall times
    walls = {}
    if WALL:
        for name in ("M", "A", "B"):
            if name in ONLY:
                walls[name] = _wall(md, replay(name))
        for tag, cfg in (
            ("S_greedy", dict(temperature=0.0, top_k=1, top_p=1.0)),
            ("S_topk20", dict(temperature=1.0, top_k=20, top_p=1.0)),
            ("S_full", dict(temperature=1.0, top_k=0, top_p=0.95)),
        ):
            runner.sample_cfg = dict(cfg, gens=torch.Generator().manual_seed(0))
            runner._set_sample_params()
            walls[tag] = _wall(md, replay("S"))
        _log("FD_WALL_MS " + json.dumps({k: round(v, 4) for k, v in walls.items()}))
    json.dump(dict(meta=meta, walls=walls), open(os.path.join(OUT, "walls.json"), "w"))
    # ---- profiled replays
    profmap = {}
    if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
        order = [n for n in ("M", "A", "B") if n in ONLY]
        if "S" in ONLY:
            order += ["S_greedy", "S_topk20", "S_full"]
        for name in order:
            if name.startswith("S_"):
                cfg = dict(
                    S_greedy=dict(temperature=0.0, top_k=1, top_p=1.0),
                    S_topk20=dict(temperature=1.0, top_k=20, top_p=1.0),
                    S_full=dict(temperature=1.0, top_k=0, top_p=0.95),
                )[name]
                runner.sample_cfg = dict(cfg, gens=torch.Generator().manual_seed(0))
                runner._set_sample_params()
                fn = replay("S")
                sess_tid = int(runner.tid_s.tids[runner.samp.fast])
            else:
                fn = replay(name)
                sess_tid = int(tids[name])
            ttnn.synchronize_device(md)
            ttnn.ReadDeviceProfiler(md)
            for r in range(REPS):
                fn()
                ttnn.synchronize_device(md)
                ttnn.ReadDeviceProfiler(md)
            _csv_copy(name)
            profmap[name] = sess_tid
            json.dump(profmap, open(os.path.join(OUT, "prof_map.json"), "w"))
            _log(f"FD profiled {name} trace {sess_tid} x{REPS}")
    dec.restore_states(snaps)
    _log("FD_DONE")
