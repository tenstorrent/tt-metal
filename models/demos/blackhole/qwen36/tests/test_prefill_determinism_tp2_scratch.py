# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""In-process repro of the TP>1 traced-prefill nondeterminism on a (1,2) mesh (p1e).

RESOLVED (p1e, 2026-09-21): the racing op was the UN-FUSED GDN out-projection all-gather (tp_common.all_gather_then_matmul_prefill,
the QWEN36_GDN_OUT_MODE=agmm -> "ag_mm" fallback every TP != 4 takes): the only prefill collective without an entry
barrier_semaphore. Its per-call output is written by the peer while this device may still be inside the preceding
silu*mul / rms_norm ops whose freed tensors reuse that address (traces replay back-to-back, so the write lands mid-op).
Flag matrix on this mesh (profiles/pd/p1e_*.json): QWEN36_AGMM_BARRIER=0/1/2 and QWEN36_AGMM_PERSISTENT=1 leave the
chunk-trace lengths 8/8 distinct (the AGMM entry barrier IS active and captured at TP=2 -- Linear, 2 devices); the
eager masked chunk is deterministic; QWEN36_AGMM_OUT_AG_BARRIER=1 (now the default) gives 1/8 at unchanged TTFT.
QWEN36_AGMM_OUT_AG_BARRIER=0 reproduces the bug.

ONE process, ONE (1,2) mesh (FABRIC_1D), the real 27B loaded once (TP=2, bf8 paged KV). For each prompt length T
the SAME T-token prompt is prefilled N times through the path vLLM serving uses (prefill_traced_chunked: masked
bucket trace for T < 2048, chunk-outer trace + masked tail for T >= 2048) and the first-step logits are compared
bit-for-bit across repeats AND across the two device replicas.

Env knobs (P1E_*), all optional:
  P1E_B          1 (default) -> prefill_traced_chunked on the B=1 buffers (vLLM max_num_seqs=1 shape);
                 8 -> prefill_paged_slots (vLLM max_num_seqs=8 shape, what p1d_run_vllm_serve_tp2_worktree.sh ran)
  P1E_LENS       comma list of prompt lengths (default 64,128,257,1025,2048,2049,4096)
  P1E_REPEATS    repeats per length (default 8)
  P1E_LAYERS     n_layers (default: all 64)
  P1E_CHUNK_TRACE 1 (default) -> capture the 2048 chunk-outer trace; 0 -> eager chunk fallback for T >= 2048
  P1E_LAYER_DUMP 1 -> per-layer hidden dump (eager masked bucket only): first layer whose output differs
  P1E_OUT        JSON result path
  QWEN36_PREFILL_BUCKET_TRACE 1 (serving default) -> traced masked bucket; 0 -> eager masked bucket
  QWEN36_AGMM_BARRIER 0/1/2, QWEN36_AGMM_PERSISTENT, QWEN36_AGMM_OUT_AG_BARRIER, QWEN36_GDN_CONV ... (the bisection flag matrix;
                 QWEN36_GDN_OUT_MODE=mmrs_fp32 and an un-fused MLP TT_FATAL/TT_THROW at TP=2 -- grid/L1 -- so they are not knobs here)

Run (chips 1,2 as one (1,2) mesh):
  source profiles/pd/p1e_env.sh; pytest -svq models/demos/blackhole/qwen36/tests/test_prefill_determinism_tp2_scratch.py
"""
import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

BLOCK = 64
BPU = 80  # blocks per slot region (4095 tokens + headroom)
RAWIDS = os.environ.get("P1B_RAWIDS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "p1b_rawids.json"))


def _prompt(T):
    d = json.load(open(RAWIDS))
    ids65, ids128 = d["rawids_65"], d["rawids_128"]
    base = ids65 + ids128[65:]
    if T <= 128:
        return base[:T]
    if T == 129:
        return ids128 + [ids65[3]]
    if T <= 2048:
        return d["rawids_2048"][:T]
    if T == 2049:
        return d["rawids_2049"]
    return (d["rawids_2049"] * ((T // 2049) + 1))[:T]


def _region(u):
    return list(range(u * BPU, (u + 1) * BPU))


def _distinct(tensors):
    out = []
    for t in tensors:
        if not any(torch.equal(t, d) for d in out):
            out.append(t)
    return out


@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 24576, "trace_region_size": 1073741824}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 2), id="1x2")], indirect=True)
def test_prefill_determinism_tp2(mesh_device, reset_seeds, ensure_gc):
    B = int(os.environ.get("P1E_B", "1"))
    lens = [int(x) for x in os.environ.get("P1E_LENS", "64,128,257,1025,2048,2049,4096").split(",")]
    reps = int(os.environ.get("P1E_REPEATS", "8"))
    n_layers = int(os.environ["P1E_LAYERS"]) if os.environ.get("P1E_LAYERS") else None
    chunk_trace = os.environ.get("P1E_CHUNK_TRACE", "1") == "1"
    layer_dump = os.environ.get("P1E_LAYER_DUMP", "0") == "1"
    out_path = os.environ.get("P1E_OUT", os.path.join(os.getcwd(), f"p1e_determinism_B{B}.json"))
    nd = mesh_device.get_num_devices()
    assert nd == 2, f"expected a (1,2) mesh, got {nd} devices"
    num_blocks = max(8, B) * BPU
    max_seq = max(8192, ((max(lens) + 2047) // 2048) * 2048)
    t0 = time.perf_counter()
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=B, max_seq_len=max_seq, n_layers=n_layers)
    assert model.use_tp
    args = model.args
    vocab = args.vocab_size
    topo = args.ccl_topology()
    logger.info(
        f"[p1e] cluster_type={ttnn.cluster.get_cluster_type()} ccl_topology={topo} "
        f"num_links(axis1)={model.tt_ccl.get_num_links(1)} AGMM_BARRIER={os.environ.get('QWEN36_AGMM_BARRIER')} "
        f"GDN_OUT_MODE={os.environ.get('QWEN36_GDN_OUT_MODE')} BUCKET_TRACE={os.environ.get('QWEN36_PREFILL_BUCKET_TRACE')} "
        f"chunk_trace={chunk_trace} B={B} n_layers={n_layers}"
    )
    model.allocate_kv_caches((num_blocks + 1, args.n_local_kv_heads, BLOCK, args.head_dim), ttnn.bfloat8_b, batch_size=B)
    logger.info(f"[p1e] model loaded in {time.perf_counter()-t0:.0f}s; bucket_trace={model._mb_trace_buckets}")

    batched = B > 1
    t0 = time.perf_counter()
    warmup_pt = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
    prev = model._bind_gdn_prefill_scratch() if batched else None
    try:
        model.capture_prefill_trace_chunked(mesh_device, warmup_pt, chunk_size=2048, capture_chunk_trace=chunk_trace)
    finally:
        if prev is not None:
            model._unbind_gdn_prefill_scratch(prev)
    if batched:
        model.warmup_gdn_slot_write()
    ttnn.synchronize_device(mesh_device)
    logger.info(
        f"[p1e] prefill warmup {time.perf_counter()-t0:.0f}s; programs={mesh_device.num_program_cache_entries()} "
        f"chunk_trace_id={model._chunked_trace_id} mb_traces={sorted(model._mb_traces)}"
    )
    n_pc = mesh_device.num_program_cache_entries()

    # Optional per-(sub)layer dump: wrap layer.forward + attention/MLP forwards to stash a sha1 digest of each
    # prefill output (both device replicas). Only meaningful on EAGER paths (reads inside a trace are impossible):
    # QWEN36_PREFILL_BUCKET_TRACE=0 for the masked bucket, P1E_CHUNK_MODE=eager_body for the chunk body.
    layer_out = []
    if layer_dump:
        import hashlib

        comp0 = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

        def _digest(y):
            t = ttnn.to_torch(y, mesh_composer=comp0)
            return hashlib.sha1(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:12]

        def _wrap(obj, name, tag, cond=None):
            orig = getattr(obj, name)

            def _w(*a, _orig=orig, _tag=tag, _cond=cond, **kw):
                y = _orig(*a, **kw)
                if _cond is None or _cond(a, kw):
                    ys = y if isinstance(y, (tuple, list)) else (y,)
                    layer_out.append((_tag, "|".join(_digest(t) for t in ys if isinstance(t, ttnn.Tensor))))
                return y

            setattr(obj, name, _w)

        cur = {"li": -1}

        def _wrap_mod(mod, name, tag, cond=None):
            """Wrap a module-level function (op call site) and digest every tensor it returns."""
            orig = getattr(mod, name, None)
            if orig is None:
                return

            def _w(*a, _orig=orig, _tag=tag, _cond=cond, **kw):
                y = _orig(*a, **kw)
                if _cond is None or _cond(a, kw):
                    ys = y if isinstance(y, (tuple, list)) else (y,)
                    for j, t in enumerate(ys):
                        if isinstance(t, ttnn.Tensor):
                            layer_out.append((f"L{cur['li']}.{_tag}[{j}]", _digest(t)))
                return y

            setattr(mod, name, _w)

        def _set_li(li):
            def _c(a, kw):
                cur["li"] = li
                return kw.get("mode") == "prefill"

            return _c

        for li, layer in enumerate(model.layers):
            att = layer.attention
            orig_fwd = layer.forward

            def _fwd(*a, _orig=orig_fwd, _li=li, **kw):
                cur["li"] = _li
                return _orig(*a, **kw)

            layer.forward = _fwd
            if layer.is_full_attention:
                _wrap(att, "forward_prefill_paged", f"L{li}.attn")
            else:
                _wrap(att, "forward_prefill", f"L{li}.gdn")
                if os.environ.get("P1E_LAYER_DUMP_DEEP", "0") == "1":
                    _wrap(att, "_project_qkvzab", f"L{li}.gdn.inproj")
                    if hasattr(att, "_conv1d_prefill_kda"):
                        _wrap(att, "_conv1d_prefill_kda", f"L{li}.gdn.kdaconv")
            _wrap(layer.feed_forward, "forward", f"L{li}.mlp", cond=lambda a, kw: a[0].shape[-2] > 32)
            _wrap(layer, "forward", f"L{li}.out", cond=lambda a, kw: kw.get("mode") == "prefill")
        if os.environ.get("P1E_LAYER_DUMP_DEEP", "0") == "1":
            # Op-level digests inside the layers (module attributes the model code calls through).
            from models.demos.blackhole.qwen36.tt import tp_common as tpc
            from models.demos.blackhole.qwen36.tt.gdn import fused_chunk as fc
            from models.demos.blackhole.qwen36.tt.attention import tp as attn_tp
            from models.demos.blackhole.qwen36.tt import mlp as mlp_mod

            _wrap_mod(tpc, "all_gather_matmul_prefill", "agmm")
            _wrap_mod(tpc, "all_gather_swiglu_prefill", "agmm_swiglu")
            _wrap_mod(tpc, "all_gather_then_matmul_prefill", "ag_mm")
            _wrap_mod(fc, "chunk_gated_delta_rule_fused_adapter", "gdn_chunk")
            _wrap_mod(attn_tp, "tt_all_reduce", "attn_allreduce", cond=lambda a, kw: a[0].shape[-2] > 32)
            from models.tt_transformers.tt import ccl as ccl_mod

            # mlp.py imports tt_all_reduce inside forward, so wrap the ccl module attribute it fetches.
            _wrap_mod(ccl_mod, "tt_all_reduce", "mlp_allreduce", cond=lambda a, kw: a[0].shape[-2] > 32)

    # P1E_CHUNK_MODE=eager_body: run the captured chunk body EAGERLY (same ops, same persistent buffers, no trace)
    # by intercepting execute_trace of the chunk trace id; the bucket traces still replay. Trace-vs-eager control
    # for exactly the chunk op sequence (the eager fallback _prefill_chunked_eager_tp takes the masked GDN branch).
    if os.environ.get("P1E_CHUNK_MODE", "trace") == "eager_body":
        _real_exec = ttnn.execute_trace

        def _exec(device, tid, cq_id=0, blocking=False):
            if model._chunked_trace_id is not None and tid == model._chunked_trace_id:
                out = model._forward_prefill_chunk_tp(
                    model._chunk_token_buf,
                    model._chunk_cos_buf,
                    model._chunk_sin_buf,
                    model._chunk_start_idx_tensor,
                    model._chunk_full_page_table_buf,
                    model._chunk_page_table_buf,
                )
                ttnn.copy(out, model._chunked_trace_output)
                ttnn.deallocate(out)
                return
            return _real_exec(device, tid, cq_id=cq_id, blocking=blocking)

        ttnn.execute_trace = _exec

    comp = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    pt_width = 1024
    results = {}
    slot = 0
    for T in lens:
        ids = _prompt(T)
        assert len(ids) == T
        toks = torch.tensor([ids], dtype=torch.long)
        nblk = -(-T // BLOCK)
        # Request blocks 0..nblk-1 (the pad block is the KV cache's last block, never in the row).
        blocks = list(range(max(nblk, BPU)))
        assert nblk <= num_blocks, f"T={T} needs {nblk} blocks > {num_blocks}"
        row = torch.zeros(1, max(pt_width, len(blocks)), dtype=torch.int32)
        row[0, :nblk] = torch.tensor(blocks[:nblk], dtype=torch.int32)
        per_rep, replica_mismatch, dumps, times = [], 0, [], []
        for r in range(reps):
            layer_out.clear()
            t1 = time.perf_counter()
            if batched:
                hl = model.prefill_paged_slots([toks], row, [slot], valid_lens=[T])
                lg_all = hl[0].reshape(-1, vocab).float()
                lg = lg_all[0].clone()
                ttnn.synchronize_device(mesh_device)
                times.append(time.perf_counter() - t1)
            else:
                lgt = model.prefill_traced_chunked(toks, row, actual_len=T)
                ttnn.synchronize_device(mesh_device)
                times.append(time.perf_counter() - t1)
                lg_all = ttnn.to_torch(lgt, mesh_composer=comp).reshape(-1, vocab).float()
                lg = lg_all[0].clone()
                if lg_all.shape[0] >= 2 and not torch.equal(lg_all[0], lg_all[-1]):
                    replica_mismatch += 1
                ttnn.deallocate(lgt)
            per_rep.append(lg)
            if layer_dump:
                dumps.append(list(layer_out))
        distinct = _distinct(per_rep)
        ref = per_rep[0]
        maxdiff = max((lg - ref).abs().max().item() for lg in per_rep)
        argmax = sorted({int(torch.argmax(lg)) for lg in per_rep})
        top2 = []
        for lg in per_rep:
            v, i = torch.topk(lg, 2)
            top2.append([int(i[0]), int(i[1]), round(float(v[0] - v[1]), 4)])
        first_bad_layer = None
        if layer_dump and len(dumps) > 1:
            for li in range(min(len(d) for d in dumps)):
                if any(dumps[0][li] != d[li] for d in dumps[1:]):
                    first_bad_layer = dumps[0][li][0]
                    n_bad = sum(any(dumps[0][j] != d[j] for d in dumps[1:]) for j in range(min(len(d) for d in dumps)))
                    logger.info(f"[p1e] T={T} first differing sublayer {first_bad_layer}; {n_bad} differing sublayers total")
                    break
        rec = {
            "T": T,
            "repeats": reps,
            "distinct_logits": len(distinct),
            "max_logit_diff": maxdiff,
            "argmax_set": argmax,
            "top2_per_run": top2,
            "replica_mismatch_runs": replica_mismatch,
            "first_bad_layer": first_bad_layer,
            "new_programs": mesh_device.num_program_cache_entries() - n_pc,
            "ttft_ms": [round(1e3 * t, 1) for t in times],
        }
        results[str(T)] = rec
        logger.info(
            f"[p1e] T={T:5d} distinct={len(distinct)}/{reps} maxdiff={maxdiff:.4g} argmax={argmax} "
            f"replica_mismatch={replica_mismatch} first_bad_layer={first_bad_layer} new_programs={rec['new_programs']} "
            f"ttft_ms(min/med)={min(times)*1e3:.0f}/{sorted(times)[len(times)//2]*1e3:.0f}"
        )
        json.dump(
            {
                "B": B,
                "n_layers": n_layers,
                "chunk_trace": chunk_trace,
                "chunk_mode": os.environ.get("P1E_CHUNK_MODE", "trace"),
                "env": {
                    k: os.environ.get(k)
                    for k in (
                        "QWEN36_PREFILL_BUCKET_TRACE",
                        "QWEN36_AGMM_BARRIER",
                        "QWEN36_GDN_OUT_MODE",
                        "QWEN36_AGMM_OUT_AG_BARRIER",
                        "QWEN36_GDN_CONV",
                        "QWEN36_AGMM_PERSISTENT",
                        "QWEN36_GDN_PROJ_CHUNKS",
                        "QWEN36_AGMM_LAYOUT",
                    )
                },
                "topology": str(topo),
                "results": results,
            },
            open(out_path, "w"),
            indent=1,
        )
    nondet = [T for T, r in results.items() if r["distinct_logits"] > 1 or r["replica_mismatch_runs"]]
    logger.info(f"[p1e] NONDETERMINISTIC lengths: {nondet}")
    assert not nondet, f"nondeterministic TP=2 prefill at T={nondet}"
