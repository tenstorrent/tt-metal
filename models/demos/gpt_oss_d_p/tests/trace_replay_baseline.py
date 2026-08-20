# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Trace-replay baseline (Lever 0): how much of the GPT-OSS prefill fixed floor is host dispatch?

The runtime is fully EAGER today: every op of every layer is dispatched from host, per chunk. This
test quantifies the host-dispatch share of the ~15.6 ms/layer fixed floor by timing ONE one-shot
prefill chunk both ways on the same built runtime:

  1. EAGER    — `prefill_chunk` (the production path), 1 JIT-warm + 3 timed, median.
  2. TRACED   — capture `model.prefill_forward` in a ttnn trace once, then `execute_trace` x10,
                median. Replay is pure device execution + one tiny dispatch, so
                (eager - replay) / eager ~= the host-dispatch overhead share.

Trace mechanics copied from `models/demos/minimax_m3/tests/perf/test_model_perf.py` (same
deepseek_prefill op family — MoE dispatch/combine/unified-FFN + CCLs — traced successfully there):
  * open the mesh with `trace_region_size` (env TRACE_REGION, default 500 MB);
  * inputs must be PERSISTENT across replays: the decoder layer frees its OWN input
    (layer.py: `residual.deallocate(True)` after the residual add), so we keep `x_persist`
    (embedded once, outside capture) and run the forward on `ttnn.clone(x_persist)` — the clone is
    part of the trace and re-reads the original's stable address on every replay;
  * warm the exact captured sequence once eagerly so no JIT compile lands inside the capture;
  * `begin_trace_capture` / `end_trace_capture` / `execute_trace(blocking=True)` / `release_trace`.

We capture at the `prefill_forward` level (not `prefill_chunk`) because `prefill_chunk`
deallocates its token input after embedding — the embed is ONE op, so the eager baseline
(prefill_chunk, embed included) vs replay (forward only) asymmetry is negligible over N layers.

If capture throws (an op not trace-capturable), we print `[trace] CAPTURE_FAILED: ...` and exit 0 —
"not currently traceable" is a valid experiment outcome.

One-shot only (cached_len=0, gather-Q): FABRIC_1D, no torus/ring needed — any healthy galaxy node.

GALAXY-GATED. Env: HF_MODEL, GPT_OSS_WEIGHTS_FROM_CACHE=1, EXPERT_DTYPE=bf8,
TT_MESH_GRAPH_DESC_PATH=.../single_bh_galaxy_mesh_graph_descriptor.textproto,
PREFILL_NUM_LAYERS (opt), TRACE_CHUNK (default 5120), TRACE_REGION (bytes, default 5e8).
"""

import os
import resource
import statistics
import sys
import time

import ttnn

ROWS, COLS = 4, 8
GALAXY_NUM_DEVICES = 32
CHUNK = int(os.getenv("TRACE_CHUNK", "5120"))
EAGER_ITERS = 4  # first iter JIT-warms (no compile() here); median of the last 3
REPLAY_ITERS = 10
TRACE_REGION = int(os.getenv("TRACE_REGION", "500000000"))  # bump if end_trace_capture reports overflow


def _raise_nproc_limit():
    """Raise RLIMIT_NPROC to the hard limit so the cold JIT kernel build (a burst of g++/collect2
    procs) doesn't fail with posix_spawn "Operation not permitted". Same as the galaxy PCC harness."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    if soft != resource.RLIM_INFINITY and (hard == resource.RLIM_INFINITY or soft < hard):
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))
            print(f"[trace] raised RLIMIT_NPROC soft {soft} -> {hard}", flush=True)
        except (ValueError, OSError) as e:
            print(f"[trace] WARNING: could not raise RLIMIT_NPROC (soft={soft}): {e}", file=sys.stderr)


def main():
    _raise_nproc_limit()
    if ttnn.get_num_devices() < GALAXY_NUM_DEVICES:
        print(f"[trace] SKIP: needs galaxy ({GALAXY_NUM_DEVICES}); have {ttnn.get_num_devices()}", flush=True)
        return 0
    from models.demos.gpt_oss_d_p.tt.model_config import ModelArgs
    from models.demos.gpt_oss_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)  # one-shot gather-Q; no torus/ring needed
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(ROWS, COLS), trace_region_size=TRACE_REGION)
    print(
        f"[trace] mesh {tuple(mesh.shape)} ndev={mesh.get_num_devices()} trace_region={TRACE_REGION}",
        flush=True,
    )
    try:
        model_args = ModelArgs(mesh_device=mesh)
        hf_config = model_args.hf_config
        num_layers = hf_config.num_hidden_layers
        nl = os.getenv("PREFILL_NUM_LAYERS")
        if nl:
            num_layers = int(nl)
            hf_config.num_hidden_layers = num_layers
        expert_dtype = ttnn.bfloat8_b if os.getenv("EXPERT_DTYPE", "bf4") == "bf8" else ttnn.bfloat4_b
        cache_path = model_args.weight_cache_path(ttnn.bfloat8_b)
        if os.getenv("GPT_OSS_WEIGHTS_FROM_CACHE") == "1":
            state_dict = {}
        else:
            state_dict = ModelArgs.load_state_dict(model_args.weights_path)

        cfg = TtPrefillRuntimeConfig(
            num_layers=num_layers,
            max_seq_len=CHUNK,  # one-shot: whole cache is one chunk
            mesh_shape=(ROWS, COLS),
            chunk_size=CHUNK,
            num_users=1,
            expert_weight_dtype=expert_dtype,
            cache_dtype=ttnn.bfloat8_b,
            weight_cache_path=cache_path,
            owns_kv_cache=True,
        )
        rt = TtPrefillRuntime(mesh, hf_config, state_dict, cfg)
        del state_dict
        print(f"[trace] built; chunk={CHUNK} num_layers={num_layers}", flush=True)

        toks = [0] * CHUNK

        # ---- 1) EAGER baseline: the production prefill_chunk path (embed + forward + out dealloc).
        # prefill_chunk consumes (deallocates) its input tensor, so build a fresh input per call.
        ts = []
        for _ in range(EAGER_ITERS):
            inp = rt.make_chunk_input(toks, CHUNK)
            t0 = time.perf_counter()
            rt.prefill_chunk(inp, slot_id=0, actual_start=0, actual_end=CHUNK, skip_lm_head=True, chunk_size=CHUNK)
            ttnn.synchronize_device(mesh)
            ts.append(time.perf_counter() - t0)
        eager_s = statistics.median(ts[1:])  # drop the JIT-warm first iter
        print(
            f"[trace] eager per-iter ms: {['%.1f' % (t * 1e3) for t in ts]} -> median(last 3) = "
            f"{eager_s * 1e3:.1f} ms",
            flush=True,
        )

        # ---- 2) TRACE capture of prefill_forward with a PERSISTENT input.
        # Embed ONCE outside capture; every fwd() consumes a clone (the layer frees its own input),
        # so x_persist survives capture and stays at a stable address for execute_trace's replay.
        tok_dev = rt.make_chunk_input(toks, CHUNK)
        x_persist = rt._embed_tokens(tok_dev)
        ttnn.deallocate(tok_dev)

        # EXACTLY the kwargs prefill_chunk passes (tt_prefill_runtime.py, cached_len=0 one-shot).
        fwd = lambda: rt.model.prefill_forward(
            ttnn.clone(x_persist),
            rot_mats_global=rt.rope_indexed[CHUNK],  # persistent (runtime-owned; never deallocated)
            kv_cache=rt.kv_cache,
            cached_len=0,
            user_id=0,
            get_last_token=-1,
            skip_lm_head=True,
            indexed_rope=True,
            on_layer_complete=None,
        )

        tid = None
        out = None
        try:
            warm = fwd()  # warm the exact captured sequence (ops already JIT-compiled by the eager loop)
            ttnn.synchronize_device(mesh)
            if warm is not None:
                warm.deallocate(True)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            out = fwd()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.synchronize_device(mesh)
        except Exception as e:
            # Best-effort unwind so close_mesh_device doesn't hang mid-capture.
            if tid is not None:
                try:
                    ttnn.end_trace_capture(mesh, tid, cq_id=0)
                    ttnn.release_trace(mesh, tid)
                except Exception:
                    pass
            print(f"[trace] CAPTURE_FAILED: {str(e)[:200]}", flush=True)
            print(
                f"[trace] RESULT ({num_layers} layers, chunk {CHUNK}): eager median = {eager_s * 1e3:.1f} ms "
                f"({eager_s * 1e3 / num_layers:.2f} ms/layer), trace-replay = N/A (capture failed)",
                flush=True,
            )
            return 0
        print(f"[trace] captured tid={tid}", flush=True)

        # ---- 3) Replay timing: blocking execute_trace, per-iter wall clock, median of 10.
        rs = []
        for _ in range(REPLAY_ITERS):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh)
            rs.append(time.perf_counter() - t0)
        replay_s = statistics.median(rs)
        print(f"[trace] replay per-iter ms: {['%.1f' % (t * 1e3) for t in rs]}", flush=True)

        share = (eager_s - replay_s) / eager_s * 100.0
        print(
            f"[trace] RESULT ({num_layers} layers, chunk {CHUNK}): "
            f"eager median = {eager_s * 1e3:.1f} ms, trace-replay median = {replay_s * 1e3:.1f} ms, "
            f"host-overhead share = {share:.1f}% "
            f"(per-layer: eager {eager_s * 1e3 / num_layers:.2f} ms, replay {replay_s * 1e3 / num_layers:.2f} ms)",
            flush=True,
        )

        ttnn.release_trace(mesh, tid)
        if out is not None:
            out.deallocate(True)
        print("[trace] DONE", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
