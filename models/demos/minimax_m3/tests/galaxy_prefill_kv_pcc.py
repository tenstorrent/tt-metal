# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M3 REAL-WEIGHTS prefill: throughput + KV-cache PCC on the galaxy (SP=8 × TP=4 + EP=32).

Builds the full 60-layer model with real bf16 weights via TtPrefillRuntime, runs prefill over the
golden trace's prompt (one-shot or chunked), measures throughput over a few iterations, then
PCC-checks every layer's post-RoPE K / raw V / MSA index_k against the golden trace from
scripts/generate_golden_kv_cache.py.

Env:
  PREFILL_TRACE_DIR   golden trace dir (metadata.json + kv_cache/layer_N.safetensors)   [required]
  PREFILL_CHUNKED     "1" -> chunked prefill (chunk loop, cache-read path); "0" -> one-shot  [default 0]
  PREFILL_CHUNK_SIZE  chunk size in tokens (chunked mode)                                  [default 5120]
  PREFILL_TPS_ITERS   prefill repetitions for the throughput measurement (less noise)      [default 1]
  HF_MODEL            real MiniMax-M3 weights dir (read by ModelArgs)

Run (after weights are present on disk):
  cd /data/philei/tt-metal
  export TT_METAL_HOME=/data/philei/tt-metal PYTHONPATH=/data/philei/tt-metal
  source python_env/bin/activate
  export HF_MODEL=/data/vmelnykov/MiniMax-M3-ref
  export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto
  # chunked over the 8192-token golden (two 5120 chunks), 5 timed iterations:
  PREFILL_CHUNKED=1 PREFILL_TPS_ITERS=5 \
    PREFILL_TRACE_DIR=/data/philei/models/minimax-m3-prefill-cache/golden/longbook_8192 \
    python3 models/demos/minimax_m3/tests/galaxy_prefill_kv_pcc.py
"""

import json
import math
import os
import resource
import statistics
import sys
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn


def _raise_nproc_limit():
    """tt-metal JIT-compiles device kernels in parallel, and each `g++ -flto=auto` fans out to
    `make -j<nproc>` — a burst of hundreds/thousands of short-lived processes. A low RLIMIT_NPROC
    (e.g. a 512 soft default) makes clone3 fail with EAGAIN mid-build, which gcc reports as
    "posix_spawn: Operation not permitted" and aborts the kernel link. Raise the soft limit to the
    hard limit (allowed without privileges) so the build never starves."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    if soft != resource.RLIM_INFINITY and (hard == resource.RLIM_INFINITY or soft < hard):
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))
            print(f"[prefill-pcc] raised RLIMIT_NPROC soft {soft} -> {hard} (tt-metal JIT build spawns many procs)")
        except (ValueError, OSError) as e:
            print(
                f"[prefill-pcc] WARNING: could not raise RLIMIT_NPROC (soft={soft}); if the kernel build "
                f"fails with 'posix_spawn: Operation not permitted', run `ulimit -u 1048576` first: {e}",
                file=sys.stderr,
            )


def plan(n_tokens, chunk_size, chunked):
    """Resolve (n_chunks, chunk, total). chunked: full chunk_size chunks (tail padded). one-shot:
    a single chunk == total, padded up to a multiple of 1024 (MSA needs S%1024==0, S>=2048; 1024 is
    also a multiple of sp=8 so the SP shard stays tile-aligned)."""
    if chunked:
        chunk = chunk_size
        n_chunks = max(1, math.ceil(n_tokens / chunk))
    else:
        chunk = max(1024, math.ceil(n_tokens / 1024) * 1024)
        n_chunks = 1
    return n_chunks, chunk, n_chunks * chunk


def check_kv_pcc(runtime, golden_dir, n_tokens, num_layers, hf_config):
    """Per-layer K / V / index_k PCC: device cache (gather_layer) vs the golden trace. The device
    stores K / index_k Meta-RoPE swizzled over the rotary slice; the golden is HF half-split, so
    permute the golden's rotary slice (identity tail) before comparing. V is raw (no swizzle)."""
    from safetensors import safe_open

    from models.common.utility_functions import comp_pcc

    head_dim = hf_config.head_dim
    rotary_dim = getattr(hf_config, "rotary_dim", head_dim)
    half = rotary_dim // 2
    src = list(range(head_dim))
    for m in range(rotary_dim):
        src[m] = half * (m % 2) + (m // 2)
    src = torch.tensor(src, dtype=torch.long)

    kv_dir = Path(golden_dir) / "kv_cache"
    logger.info(f"[kv-pcc] per-layer K / V / index_k vs golden ({golden_dir}):")
    mins = {"k": 1.0, "v": 1.0, "index_k": 1.0}
    for L in range(num_layers):
        dev_k, dev_v, dev_ik = runtime.gather_layer(slot_id=0, layer_idx=L, n_tokens=n_tokens)
        with safe_open(str(kv_dir / f"layer_{L}.safetensors"), framework="pt") as h:
            keys = set(h.keys())
            g_k = h.get_tensor(f"key_cache_layer_{L}").float()[:, :, :n_tokens, :][..., src]  # HF -> Meta
            g_v = h.get_tensor(f"value_cache_layer_{L}").float()[:, :, :n_tokens, :]
            has_ik = f"index_k_cache_layer_{L}" in keys
            g_ik = h.get_tensor(f"index_k_cache_layer_{L}").float()[:, :, :n_tokens, :][..., src] if has_ik else None

        pcc_k = float(comp_pcc(g_k, dev_k, 0.0)[1])
        pcc_v = float(comp_pcc(g_v, dev_v, 0.0)[1])
        mins["k"], mins["v"] = min(mins["k"], pcc_k), min(mins["v"], pcc_v)
        line = f"  layer {L:>2}: K={pcc_k:.5f} V={pcc_v:.5f}"
        if has_ik:
            pcc_ik = float(comp_pcc(g_ik, dev_ik, 0.0)[1])
            mins["index_k"] = min(mins["index_k"], pcc_ik)
            line += f" index_k={pcc_ik:.5f}"
        logger.info(line)

    logger.info(
        f"[kv-pcc] min PCC across {num_layers} layers: "
        f"K={mins['k']:.5f} V={mins['v']:.5f} index_k={mins['index_k']:.5f}"
    )
    return mins


def main():
    from models.demos.minimax_m3.tt.model_config import ModelArgs
    from models.demos.minimax_m3.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

    _raise_nproc_limit()  # tt-metal parallel kernel JIT needs a high process limit (see fn docstring)

    golden_dir = os.environ.get("PREFILL_TRACE_DIR")
    if not golden_dir:
        print("ERROR: set PREFILL_TRACE_DIR to a golden trace dir", file=sys.stderr)
        return 1
    token_ids = list(json.load(open(Path(golden_dir) / "metadata.json"))["token_ids"])
    n_tokens = len(token_ids)
    chunked = os.getenv("PREFILL_CHUNKED", "0") == "1"
    chunk_size = int(os.getenv("PREFILL_CHUNK_SIZE", "5120"))
    tps_iters = int(os.getenv("PREFILL_TPS_ITERS", "1"))

    rows, cols = 8, 4  # SP=8 (rows), TP=4 (cols), EP=32
    n_chunks, chunk, total = plan(n_tokens, chunk_size, chunked)
    print(
        f"[prefill-pcc] golden={golden_dir} n_tokens={n_tokens} "
        f"mode={'chunked' if chunked else 'one-shot'} chunk={chunk} n_chunks={n_chunks} total={total} "
        f"tps_iters={tps_iters}",
        flush=True,
    )
    if not chunked and total < 2048:
        print(
            f"[prefill-pcc] WARNING: one-shot total={total} < 2048 — MSA (sparse) layers 3-59 require "
            f"S>=2048 and S%1024==0 and will fail. Use PREFILL_CHUNKED=1 instead.",
            flush=True,
        )

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols))
    print(f"[prefill-pcc] mesh opened {tuple(mesh.shape)} ndev={mesh.get_num_devices()}", flush=True)
    try:
        model_args = ModelArgs(mesh_device=mesh)  # HF_MODEL
        hf_config = model_args.hf_config
        num_layers = hf_config.num_hidden_layers

        print("[prefill-pcc] loading real bf16 weights + EP placement (slow first run) ...", flush=True)
        state_dict = ModelArgs.load_state_dict(model_args.weights_path)
        cfg = TtPrefillRuntimeConfig(
            num_layers=num_layers,
            max_seq_len=total,
            mesh_shape=(rows, cols),
            chunk_size=chunk,
            num_users=1,
            weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
        )
        runtime = TtPrefillRuntime(mesh, hf_config, state_dict, cfg)
        del state_dict

        print(f"[prefill-pcc] compiling ({num_layers}L, SP=8 × TP=4 + EP=32) ...", flush=True)
        runtime.compile()

        # --- throughput: each iteration re-fills slot 0; the cache is valid after the last ---
        padded = token_ids + [0] * (total - n_tokens)

        def run_once():
            for c in range(n_chunks):
                a = c * chunk
                inp = runtime.make_chunk_input(padded[a : a + chunk])
                runtime.prefill(inp, slot_id=0, actual_start=a, actual_end=min(a + chunk, n_tokens))
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
            f"[prefill-pcc] THROUGHPUT over {tps_iters} iters: median {n_tokens / med:.1f} tok/s (real prompt), "
            f"{total / med:.1f} tok/s (processed); wall median {med * 1000:.1f} ms "
            f"[min {min(times) * 1000:.1f}, max {max(times) * 1000:.1f}]",
            flush=True,
        )

        # --- accuracy: per-layer KV PCC vs golden ---
        check_kv_pcc(runtime, golden_dir, n_tokens, num_layers, hf_config)
        print("[prefill-pcc] DONE", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
