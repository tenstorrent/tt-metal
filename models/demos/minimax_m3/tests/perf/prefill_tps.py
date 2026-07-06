# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M3 prefill throughput — traced, device-bound tok/s for the REAL SP=8 × TP=4 + EP=32 path.

This is the perf counterpart to tests/galaxy_prefill_kv_pcc.py (which owns cache-PCC and only does
a rough host-timed throughput check). Here we get a CLEAN device-bound number by hoisting all host
prep (embedding + SP-resharded RoPE) OUT of the timed region and capturing a ttnn trace of just
`model.ttnn_prefill_forward` — the pure on-device chunk forward — then replaying it BLOCKING ×K so
per-call dispatch amortizes to ~0. wall ≈ device time.

Phase 1 (this file): tok/s for ONE 5120-token chunk (cached_len=0 — the "first 5k" of a prompt).
A >5k prompt processes as: first 5k -> KV cache -> next 5k reads that prefix -> output. The
cache-read chunk (cached_len>0) bakes a fixed offset into its trace, so it needs its own capture;
Phase 1 measures the first (dominant, repeating) chunk. Phase 2 will add tracy signposts around
MSA / MoE / dense-attn / CCL to break the number down by region (see build_signpost_regions TODO).

The number is device-timing-representative even though the full 60-layer run still has the open L35
overflow / chunked-indexer accuracy bugs — the ops execute at real shapes. Quote it as
"perf-representative, accuracy WIP".

Env:
  PERF_CHUNK      chunk size in tokens (== the single chunk we time)                   [default 5120]
  PERF_LAYERS     build/run only the first N decoder layers (also sets M3_LOAD_NLAYERS) [default: all 60]
  PERF_REPS       blocking replays to average over                                      [default 10]
  PERF_SKIP_LM_HEAD  "1" -> cache-fill only (the cost that repeats per chunk); "0" -> include
                     final norm + lm_head (the "output result" step of the last chunk)  [default 1]
  PERF_TRACE      "1" -> capture+replay trace (clean); "0" -> eager blocking loop (fallback) [default 1]
  PERF_EXPERT_DTYPE  expert matmul dtype: "bf8" or "bf4" — DOMINATES MoE cost, so set it to whatever
                     we actually ship (must match the populated weight cache)                 [default bf8]
  HF_MODEL        real MiniMax-M3 weights dir (read by ModelArgs)

Run (on the Blackhole galaxy, weights present on disk):
  export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto
  export HF_MODEL=/data/vmelnykov/MiniMax-M3-ref
  python3 models/demos/minimax_m3/tests/perf/prefill_tps.py
"""

import os
import resource
import sys
import time

import torch
from loguru import logger

import ttnn


def _raise_nproc_limit():
    """tt-metal parallel kernel JIT spawns many short-lived g++/make procs; a low RLIMIT_NPROC makes
    the build fail with 'posix_spawn: Operation not permitted'. Raise soft -> hard (no privilege needed)."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    if soft != resource.RLIM_INFINITY and (hard == resource.RLIM_INFINITY or soft < hard):
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))
            print(f"[prefill-tps] raised RLIMIT_NPROC soft {soft} -> {hard}")
        except (ValueError, OSError) as e:
            print(f"[prefill-tps] WARNING: could not raise RLIMIT_NPROC (soft={soft}): {e}", file=sys.stderr)


def _load_state_dict(model_args, hf_config, num_layers, expert_dtype):
    """Cache-only weight load (DeepSeek trick): if the tilized .tensorbin cache is complete, pass an
    EMPTY state_dict and never touch the ~869GB bf16 source. Mirrors galaxy_prefill_kv_pcc.py."""
    from models.demos.minimax_m3.tt.model_config import ModelArgs
    from models.demos.minimax_m3.tt.weight_cache import weight_cache_is_complete

    cache_path = model_args.weight_cache_path(expert_dtype)
    force_load = os.getenv("M3_FORCE_LOAD_WEIGHTS") == "1"
    cache_only = not force_load and (
        os.getenv("M3_WEIGHTS_FROM_CACHE") == "1"
        or weight_cache_is_complete(cache_path, hf_config, num_layers, expert_dtype)
    )
    if cache_only:
        print("[prefill-tps] weight cache complete -> cache-only load (skipping bf16 source read)", flush=True)
        return {}, cache_path
    print("[prefill-tps] loading real bf16 weights (slow source read) ...", flush=True)
    return ModelArgs.load_state_dict(model_args.weights_path), cache_path


def main():
    from models.demos.minimax_m3.tt.model_config import ModelArgs
    from models.demos.minimax_m3.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

    _raise_nproc_limit()

    chunk = int(os.getenv("PERF_CHUNK", "5120"))
    reps = int(os.getenv("PERF_REPS", "10"))
    skip_lm_head = os.getenv("PERF_SKIP_LM_HEAD", "1") == "1"
    use_trace = os.getenv("PERF_TRACE", "1") == "1"
    rows, cols = 8, 4  # SP=8 (rows), TP=4 (cols), EP=32

    assert chunk % 1024 == 0 and chunk >= 2048, f"PERF_CHUNK={chunk} must be a multiple of 1024 and >= 2048 (MSA)"

    # A trace of the full 60-layer forward is large; give the trace region room (bump for real 60L).
    trace_region = int(os.getenv("TRACE_REGION", "1000000000"))
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols), trace_region_size=trace_region if use_trace else 0)
    print(f"[prefill-tps] mesh {tuple(mesh.shape)} ndev={mesh.get_num_devices()} chunk={chunk} reps={reps} "
          f"skip_lm_head={skip_lm_head} trace={use_trace}", flush=True)
    try:
        model_args = ModelArgs(mesh_device=mesh)  # HF_MODEL
        hf_config = model_args.hf_config
        num_layers = hf_config.num_hidden_layers
        nl_override = os.getenv("PERF_LAYERS")
        if nl_override:
            num_layers = int(nl_override)
            hf_config.num_hidden_layers = num_layers
            os.environ.setdefault("M3_LOAD_NLAYERS", str(num_layers))
            print(f"[prefill-tps] PERF_LAYERS={num_layers}: first {num_layers} layers only", flush=True)

        # Expert matmul dtype DOMINATES MoE cost — profile what we ship. NOTE: TtPrefillRuntimeConfig
        # defaults to bf4, but galaxy_prefill_kv_pcc.py's cache path uses bf8; we set both consistently
        # here and pass it explicitly into the config so the model build and cache path agree.
        expert_dtype = ttnn.bfloat4_b if os.getenv("PERF_EXPERT_DTYPE", "bf8") == "bf4" else ttnn.bfloat8_b
        state_dict, cache_path = _load_state_dict(model_args, hf_config, num_layers, expert_dtype)

        cfg = TtPrefillRuntimeConfig(
            num_layers=num_layers,
            max_seq_len=chunk,  # single chunk -> cached_len=0 ("first 5k")
            mesh_shape=(rows, cols),
            chunk_size=chunk,
            num_users=1,
            expert_weight_dtype=expert_dtype,
            weight_cache_path=cache_path,
        )
        runtime = TtPrefillRuntime(mesh, hf_config, state_dict, cfg)
        del state_dict

        print(f"[prefill-tps] compiling ({num_layers}L, SP=8 × TP=4 + EP=32) ...", flush=True)
        runtime.compile()

        # --- Host prep, done ONCE and kept persistent (NOT part of the timed device forward). ---
        # Random tokens: op shapes (and thus device timing) are content-independent — MoE dispatch uses
        # a fixed-capacity buffer and every expert runs dense, so routing skew doesn't change the graph.
        gen = torch.Generator().manual_seed(0)
        tokens = torch.randint(0, hf_config.vocab_size, (chunk,), generator=gen, dtype=torch.int32).tolist()
        x_persist = runtime.make_chunk_input(tokens)  # embedded + SP-sharded; kept alive across replays
        rope_abs = runtime._sp_reshard_rope(0, chunk)  # RoPE for absolute positions [0, chunk); persistent
        last = ((chunk - 1) // 32) * 32  # last real token's 32-tile (only used when skip_lm_head=False)

        # The decoder layer frees its OWN input buffer, so each forward consumes its input -> clone the
        # persistent input per call so x_persist survives capture and stays at a stable replay address.
        def fwd():
            return runtime.model.ttnn_prefill_forward(
                ttnn.clone(x_persist),
                rot_mats_global=rope_abs,
                kv_cache=runtime.kv_cache,
                cached_len=0,
                user_id=0,
                get_last_token=(-1 if skip_lm_head else last),
                skip_lm_head=skip_lm_head,
            )

        tokens_per_chunk = chunk  # whole system processes `chunk` tokens per forward (SP-sharded 640/chip)

        if use_trace:
            fwd()  # warmup / compile the exact call
            ttnn.synchronize_device(mesh)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            out = fwd()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.synchronize_device(mesh)
            print(f"[prefill-tps] traced forward captured (chunk={chunk}, {num_layers}L)", flush=True)

            t0 = time.perf_counter()
            for _ in range(reps):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh)
            t1 = time.perf_counter()
            if out is not None:
                ttnn.deallocate(out)
        else:
            # Fallback: eager blocking forward loop (still amortizes dispatch over reps; includes a bit
            # more host overhead than the traced path).
            out = fwd()
            if out is not None:
                ttnn.deallocate(out)
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(reps):
                out = fwd()
                if out is not None:
                    ttnn.deallocate(out)
            ttnn.synchronize_device(mesh)
            t1 = time.perf_counter()

        wall = (t1 - t0) / reps
        print(
            f"\n[prefill-tps] SP=8 × TP=4 + EP=32  layers={num_layers}  chunk={chunk} tokens  reps={reps}  "
            f"{'traced' if use_trace else 'eager'}  skip_lm_head={skip_lm_head}\n"
            f"  wall-clock / chunk : {wall*1e3:.2f} ms\n"
            f"  prefill throughput : {tokens_per_chunk / wall:,.0f} tok/s\n",
            flush=True,
        )
    finally:
        ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
