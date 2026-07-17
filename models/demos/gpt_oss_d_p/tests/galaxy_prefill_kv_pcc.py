# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GPT-OSS REAL-WEIGHTS prefill: throughput + per-layer KV-cache PCC on the galaxy (SP=4 × TP=8 + EP=32).

Mirrors ``minimax_m3/tests/galaxy_prefill_kv_pcc.py``. Builds the full 36-layer model with real bf16
weights via ``TtPrefillRuntime``, runs prefill over the golden trace's prompt, measures throughput,
then PCC-checks every layer's post-RoPE K / raw V against the golden trace from
``scripts/generate_golden_kv_cache.py`` (GQA: NO index_k, unlike M3).

GALAXY-GATED: needs the (4,8) Blackhole galaxy (32 devices) for TP=8 + EP=32. Auto-SKIPs (returns 0)
when not on a galaxy or when no golden trace is provided.

Env:
  PREFILL_TRACE_DIR   golden trace dir (metadata.json + kv_cache/layer_N.safetensors)     [required]
  PREFILL_CHUNKED     "1" -> chunked (NOT SUPPORTED yet: cache-read is NotImplemented); "0" -> one-shot [default 0]
  PREFILL_CHUNK_SIZE  chunk size in tokens (chunked mode only)                             [default 5120]
  PREFILL_TPS_ITERS   prefill repetitions for the throughput measurement                   [default 1]
  PREFILL_NUM_LAYERS  build/run only the first N decoder layers (faster partial-model runs) [default: all]
  EXPERT_DTYPE        MoE routed-expert weight dtype: "bf4" or "bf8"                        [default bf4]
  GPT_OSS_WEIGHTS_FROM_CACHE  "1" -> pass an empty state_dict (load tilized weights from the TTNN cache)
  HF_MODEL            real gpt-oss weights dir (read by ModelArgs)

Run (single Blackhole galaxy, after weights + golden are staged):
  export HF_MODEL=/path/to/gpt-oss-120b
  export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto
  PREFILL_TRACE_DIR=/path/to/golden/longbook_8192 \\
    python3 models/demos/gpt_oss_d_p/tests/galaxy_prefill_kv_pcc.py
"""

import json
import math
import os
import resource
import statistics
import sys
import time
from pathlib import Path

import ttnn

# The (4,8) galaxy = 32 devices. Below this we can't do TP=8 + EP=32, so auto-skip.
GALAXY_NUM_DEVICES = 32
ROWS, COLS = 4, 8  # SP=4 (rows), TP=8 (cols), EP=32


def _raise_nproc_limit():
    """Raise RLIMIT_NPROC to the hard limit so tt-metal's parallel kernel JIT (a burst of g++/make
    procs) doesn't starve with EAGAIN mid-build. See M3's harness for the full rationale."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    if soft != resource.RLIM_INFINITY and (hard == resource.RLIM_INFINITY or soft < hard):
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))
            print(f"[prefill-pcc] raised RLIMIT_NPROC soft {soft} -> {hard}")
        except (ValueError, OSError) as e:
            print(f"[prefill-pcc] WARNING: could not raise RLIMIT_NPROC (soft={soft}): {e}", file=sys.stderr)


def plan(n_tokens, chunk_size, chunked, sp):
    """Resolve (n_chunks, chunk, total). one-shot: a single chunk == total, padded up to a multiple of
    TILE_SIZE*sp (=128 at sp=4) so build_indexed_rope's chunk_size % (TILE_SIZE*sp) == 0 holds and the
    SP shard stays tile-aligned. chunked: full chunk_size chunks (tail padded)."""
    align = ttnn.TILE_SIZE * sp
    if chunked:
        chunk = math.ceil(chunk_size / align) * align
        n_chunks = max(1, math.ceil(n_tokens / chunk))
    else:
        chunk = max(align, math.ceil(n_tokens / align) * align)
        n_chunks = 1
    return n_chunks, chunk, n_chunks * chunk


def main():
    _raise_nproc_limit()

    golden_dir = os.environ.get("PREFILL_TRACE_DIR")
    if not golden_dir:
        print("[prefill-pcc] SKIP: set PREFILL_TRACE_DIR to a golden trace dir", flush=True)
        return 0
    if ttnn.get_num_devices() < GALAXY_NUM_DEVICES:
        print(
            f"[prefill-pcc] SKIP: needs the (4,8) galaxy ({GALAXY_NUM_DEVICES} devices) for TP=8 + EP=32; "
            f"have {ttnn.get_num_devices()}",
            flush=True,
        )
        return 0

    token_ids = list(json.load(open(Path(golden_dir) / "metadata.json"))["token_ids"])
    n_tokens = len(token_ids)
    chunked = os.getenv("PREFILL_CHUNKED", "0") == "1"
    chunk_size = int(os.getenv("PREFILL_CHUNK_SIZE", "5120"))
    tps_iters = int(os.getenv("PREFILL_TPS_ITERS", "1"))

    n_chunks, chunk, total = plan(n_tokens, chunk_size, chunked, ROWS)
    print(
        f"[prefill-pcc] golden={golden_dir} n_tokens={n_tokens} "
        f"mode={'chunked' if chunked else 'one-shot'} chunk={chunk} n_chunks={n_chunks} total={total} "
        f"tps_iters={tps_iters}",
        flush=True,
    )
    if chunked:
        # P2 not yet implemented: the GQA cache-read attention path (cached_len>0) raises
        # NotImplementedError in attention/prefill.py, so chunk 2+ fails. Warn loudly.
        print(
            "[prefill-pcc] WARNING: chunked prefill drives the cache-READ attention path (cached_len>0), "
            "which is NotImplementedError today (see attention/prefill.py). Expect a failure on chunk 2. "
            "Use one-shot (PREFILL_CHUNKED=0) until the ring/paged chunked SDPA lands.",
            flush=True,
        )

    from models.demos.gpt_oss_d_p.tt.model_config import ModelArgs
    from models.demos.gpt_oss_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(ROWS, COLS))
    print(f"[prefill-pcc] mesh opened {tuple(mesh.shape)} ndev={mesh.get_num_devices()}", flush=True)
    try:
        model_args = ModelArgs(mesh_device=mesh)  # HF_MODEL
        hf_config = model_args.hf_config
        num_layers = hf_config.num_hidden_layers
        nl_override = os.getenv("PREFILL_NUM_LAYERS")
        if nl_override:
            num_layers = int(nl_override)
            hf_config.num_hidden_layers = num_layers
            print(f"[prefill-pcc] PREFILL_NUM_LAYERS={num_layers}: first {num_layers} layers only", flush=True)

        expert_dtype = ttnn.bfloat8_b if os.getenv("EXPERT_DTYPE", "bf4") == "bf8" else ttnn.bfloat4_b
        cache_path = model_args.weight_cache_path(ttnn.bfloat8_b)

        if os.getenv("GPT_OSS_WEIGHTS_FROM_CACHE") == "1":
            print("[prefill-pcc] GPT_OSS_WEIGHTS_FROM_CACHE=1 -> empty state_dict (load tilized cache)", flush=True)
            state_dict = {}
        else:
            print("[prefill-pcc] loading real bf16 weights (slow: safetensors read) ...", flush=True)
            state_dict = ModelArgs.load_state_dict(model_args.weights_path)

        cfg = TtPrefillRuntimeConfig(
            num_layers=num_layers,
            max_seq_len=total,
            mesh_shape=(ROWS, COLS),
            chunk_size=chunk,
            num_users=1,
            expert_weight_dtype=expert_dtype,
            weight_cache_path=cache_path,
            owns_kv_cache=True,  # standalone harness owns its cache (runtime.kv_cache)
        )
        runtime = TtPrefillRuntime(mesh, hf_config, state_dict, cfg)
        del state_dict

        print(f"[prefill-pcc] compiling ({num_layers}L, SP={ROWS} × TP={COLS} + EP=32) ...", flush=True)
        runtime.compile()

        padded = token_ids + [0] * (total - n_tokens)

        def run_once():
            for c in range(n_chunks):
                a = c * chunk
                inp = runtime.make_chunk_input(padded[a : a + chunk])
                runtime.prefill_chunk(inp, slot_id=0, actual_start=a, actual_end=min(a + chunk, n_tokens))
            ttnn.synchronize_device(mesh)

        times = []
        for i in range(tps_iters):
            t0 = time.perf_counter()
            run_once()
            dt = time.perf_counter() - t0
            times.append(dt)
            print(
                f"[prefill-pcc] iter {i}: {dt * 1000:.1f} ms  "
                f"{n_tokens / dt:.1f} tok/s (real)  {total / dt:.1f} tok/s (incl pad)",
                flush=True,
            )
        med = statistics.median(times)
        print(
            f"[prefill-pcc] THROUGHPUT over {tps_iters} iters: median {n_tokens / med:.1f} tok/s (real), "
            f"{total / med:.1f} tok/s (processed); wall median {med * 1000:.1f} ms",
            flush=True,
        )

        # --- accuracy: per-layer KV PCC vs golden (K permuted HF->Meta over head_dim; V raw) ---
        min_pcc = runtime.kv_cache_pcc_check(slot_id=0, n_chunks=n_chunks, trace_dir=golden_dir)
        print(f"[prefill-pcc] min KV PCC across {num_layers} layers = {min_pcc:.5f}", flush=True)
        print("[prefill-pcc] DONE", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
