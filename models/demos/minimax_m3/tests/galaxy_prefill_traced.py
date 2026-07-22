# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M3 REAL-WEIGHTS prefill: TRACED (device-bound) throughput vs eager, on the galaxy (SP=8 × TP=4 + EP=32).

`prefill_forward` is fully traceable (on-device EP bridge + device router, no host ops mid-forward), so we
capture a ttnn device trace of one prefill and time the BLOCKING replay. The replay is pure device execution
(the host only kicks it off), so its wall is device-bound — unlike the eager path, which dispatches every op
from host and is dominated by that dispatch. Both paths time the IDENTICAL forward (embedding is done once,
up front), so the only difference is live host dispatch vs trace replay: the gap is pure host-dispatch overhead.

Why it matters: eager prefill wall is mostly host dispatch, so it (a) understates device throughput and
(b) hides device-time optimizations. Measure device wins on the traced wall.

Env:
  HF_MODEL           model path (also the default tilized-weight-cache dir)               [required]
  PREFILL_SEQ_LEN    token count for the default random prompt                            [default 5120]
  PREFILL_TRACE_DIR  replay a specific golden prompt instead (metadata.json token_ids)    [optional]
  TT_CACHE_PATH      tilized weight-cache dir (overrides the default of HF_MODEL)          [optional]
  PREFILL_NUM_LAYERS build/run only the first N decoder layers (partial-model runs)       [default all]
  PERF_REPS          traced-replay repetitions (amortizes the one-shot dispatch)          [default 10]
  TRACE_REGION       trace buffer bytes (bump for the full 60-layer model)                [default 4.5e9]
  EXPERT_DTYPE       bf4 (default) / bf8 routed-expert weight dtype

Run (full model, 5120 random tokens; weights load from HF_MODEL's tilized cache):
  HF_MODEL=... TRACE_REGION=4500000000 \
    python models/demos/minimax_m3/tests/galaxy_prefill_traced.py
"""

import json
import math
import os
import random
import resource
import sys
import time
from pathlib import Path

import ttnn

MSA_MIN_TOKENS = 16 * 128  # MSA top-k needs >= 16 blocks; floor/pad the one-shot sequence to this


def _raise_nproc_limit():
    """The tt-metal parallel kernel JIT spawns many procs; raise the soft NPROC limit to the hard cap so
    clone3 does not fail with EAGAIN mid-build."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))
    except (ValueError, OSError):
        pass


def main():
    from models.demos.minimax_m3.tt.attention import allocate_kv_caches
    from models.demos.minimax_m3.tt.model_config import ModelArgs
    from models.demos.minimax_m3.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig
    from models.demos.minimax_m3.tt.weight_cache import weight_cache_is_complete

    _raise_nproc_limit()

    # Token ids drive the forward, but this is a pure perf measurement — the dispatch/op graph is the
    # same for any tokens — so a golden prompt is optional. Default to a deterministic random sequence of
    # PREFILL_SEQ_LEN tokens; set PREFILL_TRACE_DIR to replay a specific golden prompt instead.
    golden_dir = os.environ.get("PREFILL_TRACE_DIR")
    if golden_dir:
        token_ids = list(json.load(open(Path(golden_dir) / "metadata.json"))["token_ids"])
    else:
        rng = random.Random(0)
        token_ids = [rng.randrange(1000) for _ in range(int(os.getenv("PREFILL_SEQ_LEN", "5120")))]
    n_tokens = len(token_ids)
    # One-shot: a single chunk == total, padded up to a multiple of 1024 and at least the MSA floor.
    total = max(MSA_MIN_TOKENS, math.ceil(n_tokens / 1024) * 1024)
    padded = token_ids + [0] * (total - n_tokens)
    reps = int(os.getenv("PERF_REPS", "10"))
    trace_region = int(float(os.getenv("TRACE_REGION", "4500000000")))
    rows, cols = 8, 4

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols), trace_region_size=trace_region)
    try:
        model_args = ModelArgs(mesh_device=mesh)
        hf_config = model_args.hf_config
        num_layers = hf_config.num_hidden_layers
        nl = os.getenv("PREFILL_NUM_LAYERS")
        if nl:
            num_layers = int(nl)
            hf_config.num_hidden_layers = num_layers
            os.environ.setdefault("M3_LOAD_NLAYERS", str(num_layers))

        # Weights: load from the tilized per-tensor cache when complete (empty state_dict skips the ~869GB
        # bf16 source read); otherwise read the source once to populate it.
        cache_path = model_args.weight_cache_path(ttnn.bfloat8_b)
        expert_dtype = ttnn.bfloat8_b if os.getenv("EXPERT_DTYPE", "bf4") == "bf8" else ttnn.bfloat4_b
        cache_only = os.getenv("M3_FORCE_LOAD_WEIGHTS") != "1" and (
            os.getenv("M3_WEIGHTS_FROM_CACHE") == "1"
            or weight_cache_is_complete(cache_path, hf_config, num_layers, expert_dtype)
        )
        state_dict = {} if cache_only else ModelArgs.load_state_dict(model_args.weights_path)
        cfg = TtPrefillRuntimeConfig(
            num_layers=num_layers,
            max_seq_len=total,
            mesh_shape=(rows, cols),
            chunk_size=total,  # one-shot
            num_users=1,
            weight_cache_path=cache_path,
        )
        runtime = TtPrefillRuntime(mesh, hf_config, state_dict, cfg)
        del state_dict
        # Engine-owned KV cache (stateless runtime): allocate here, pass into compile/prefill_chunk/forward.
        kv_cache = allocate_kv_caches(
            mesh, num_layers=num_layers, max_seq_len=total, num_users=1, head_dim=hf_config.head_dim
        )
        print(f"[traced] compiling ({num_layers}L, one-shot {total} tok) ...", flush=True)
        runtime.compile(kv_cache)

        # prefill_forward frees its own input, so keep the embedded hidden state PERSISTENT and forward a
        # fresh clone each call; the clone is what the trace captures, so every replay reproduces it.
        # Embedding is done ONCE here — outside both timed regions — so the eager and traced measurements
        # time the IDENTICAL fwd() and the only difference is live dispatch vs trace replay.
        model = runtime.model
        _tok = runtime.make_chunk_input(padded)
        x_persist = runtime._embed_tokens(_tok)
        ttnn.deallocate(_tok)

        def fwd():
            return model.prefill_forward(
                ttnn.clone(x_persist),
                rot_mats_global=runtime.rope_indexed,
                kv_cache=kv_cache,
                cached_len=0,
                user_id=0,
                get_last_token=-1,
                skip_lm_head=True,
                indexed_rope=True,
            )

        # --- eager: the same fwd() dispatched live from host (every op enqueued per call) ---
        for _ in range(2):  # warm (compile + program cache)
            o = fwd()
            ttnn.synchronize_device(mesh)
            if o is not None:
                o.deallocate(True)
        te = []
        for _ in range(reps):
            t0 = time.perf_counter()
            o = fwd()
            ttnn.synchronize_device(mesh)
            te.append(time.perf_counter() - t0)
            if o is not None:
                o.deallocate(True)
        eager_ms = 1e3 * sorted(te)[len(te) // 2]

        # --- traced: capture the same fwd() once, time the blocking replay ---
        o = fwd()
        ttnn.synchronize_device(mesh)
        if o is not None:
            o.deallocate(True)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        fwd()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.synchronize_device(mesh)
        # signposts bracket ONLY the replay so a tracy device-perf run can compare the replay's
        # device-kernel-active time to this wall (the gap = trace-replay device-idle / dispatch stalls).
        try:
            from tracy import signpost
        except ImportError:
            signpost = lambda **kw: None  # noqa: E731 (no-op when not under tracy)
        signpost(header="start")
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh)
        traced_ms = 1e3 * (time.perf_counter() - t0) / reps
        signpost(header="stop")
        ttnn.release_trace(mesh, tid)

        print(
            f"\n[traced] {num_layers}L one-shot prefill, {n_tokens} real tokens (processed {total}):\n"
            f"  eager  wall: {eager_ms:8.2f} ms  ({n_tokens / (eager_ms / 1e3):,.0f} tok/s)\n"
            f"  traced wall: {traced_ms:8.2f} ms  ({n_tokens / (traced_ms / 1e3):,.0f} tok/s)\n"
            f"  host-dispatch gap (eager/traced): {eager_ms / traced_ms:.2f}x  ({eager_ms - traced_ms:.1f} ms)",
            flush=True,
        )
    finally:
        ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
