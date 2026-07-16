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
  PREFILL_NUM_LAYERS  build/run only the first N decoder layers (faster partial-model runs; also auto-sets
                      M3_LOAD_NLAYERS so only those layers' weight shards are read)          [default: all]
  EXPERT_DTYPE        MoE routed-expert weight dtype: "bf4" or "bf8" (cache holds both)      [default bf4]
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


# MSA (sparse) layers 3-59 pick the top-TOPK_BLOCKS of BLOCK_SIZE-token blocks; topk_large_indices
# aborts unless at least that many blocks exist, so the sparse path floors the sequence here. Keep in
# sync with TOPK_BLOCKS * BLOCK_SIZE in models/demos/minimax_m3/tt/attention/msa.py.
MSA_MIN_TOKENS = 16 * 128  # = 2048


def plan(n_tokens, chunk_size, chunked):
    """Resolve (n_chunks, chunk, total). chunked: full chunk_size chunks (tail padded). one-shot:
    a single chunk == total, padded up to a multiple of 1024 and at least MSA_MIN_TOKENS (MSA needs
    S%1024==0 and S>=2048; 1024 is also a multiple of sp=8 so the SP shard stays tile-aligned)."""
    if chunked:
        chunk = chunk_size
        n_chunks = max(1, math.ceil(n_tokens / chunk))
    else:
        chunk = max(MSA_MIN_TOKENS, math.ceil(n_tokens / 1024) * 1024)
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
    from models.demos.minimax_m3.tt.weight_cache import weight_cache_is_complete

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
    if not chunked and n_tokens < MSA_MIN_TOKENS:
        bang = "!" * 80
        print(
            f"\n{bang}\n"
            f"[prefill-pcc] WARNING: prompt is only {n_tokens} tokens, below the MSA sparse floor of\n"
            f"  {MSA_MIN_TOKENS} (TOPK_BLOCKS*BLOCK_SIZE = 16*128). Layers 3-59 select the top-16 of\n"
            f"  128-token blocks, and topk_large_indices aborts with fewer than 16 blocks. PADDING the\n"
            f"  sequence {n_tokens} -> {total} tokens (token 0) so the sparse path can run.\n"
            f"  * ACCURACY IS STILL VALID: KV PCC compares only the first {n_tokens} real tokens, and\n"
            f"    causal masking keeps the trailing pad (positionally future) from touching them.\n"
            f"  * THROUGHPUT IS NOT: tok/s below is dominated by {total - n_tokens} pad tokens — ignore it\n"
            f"    for tiny prompts and measure perf on a >= {MSA_MIN_TOKENS}-token trace instead.\n"
            f"{bang}\n",
            flush=True,
        )
    if chunked and chunk < MSA_MIN_TOKENS:
        print(
            f"[prefill-pcc] WARNING: chunked chunk={chunk} < MSA floor {MSA_MIN_TOKENS}; the first chunk "
            f"(cache empty) has < 16 blocks and MSA topk will abort. Use PREFILL_CHUNK_SIZE >= {MSA_MIN_TOKENS}.",
            flush=True,
        )

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols))
    print(f"[prefill-pcc] mesh opened {tuple(mesh.shape)} ndev={mesh.get_num_devices()}", flush=True)
    try:
        model_args = ModelArgs(mesh_device=mesh)  # HF_MODEL
        hf_config = model_args.hf_config
        num_layers = hf_config.num_hidden_layers
        nl_override = os.getenv("PREFILL_NUM_LAYERS")
        if nl_override:
            num_layers = int(nl_override)
            hf_config.num_hidden_layers = num_layers  # build/run only the first N decoder layers
            # Only read the safetensors shards holding layers 0..N-1 (+ embed/norm/lm_head) — skips most of
            # the NFS source read. The full model's first N layers are causal-identical to the truncated
            # model's, so the golden's first N layers are still the correct reference.
            os.environ.setdefault("M3_LOAD_NLAYERS", str(num_layers))
            print(
                f"[prefill-pcc] PREFILL_NUM_LAYERS={num_layers}: first {num_layers} layers only "
                f"(M3_LOAD_NLAYERS={os.environ['M3_LOAD_NLAYERS']})",
                flush=True,
            )

        # Weight loading. The bf16 source backbone is ~869GB — larger than host RAM here — so reading
        # it every run thrashes the page cache for >1h. Every weight module already loads its tilized
        # tensor from a per-tensor .tensorbin cache via ttnn.as_tensor(cache_file_name=); on a cache hit
        # the source tensor is ignored. So once the cache is populated we pass an EMPTY state_dict and
        # never touch the source — DeepSeek's state_dict={} + check_cache_complete trick.
        #   M3_FORCE_LOAD_WEIGHTS=1  force the source read (to (re)populate the cache / first run)
        #   M3_WEIGHTS_FROM_CACHE=1  force cache-only even if the completeness check is unsure
        cache_path = model_args.weight_cache_path(ttnn.bfloat8_b)
        # EXPERT_DTYPE selects the MoE routed-expert weight dtype (bf4 default / bf8). The tilized
        # cache holds both, so either stays on the fast cache path. Same knob name as the generate
        # harnesses (galaxy_generate_m3*.py). It feeds TtPrefillRuntimeConfig.expert_weight_dtype below.
        expert_dtype = ttnn.bfloat8_b if os.getenv("EXPERT_DTYPE", "bf4") == "bf8" else ttnn.bfloat4_b
        print(
            f"[prefill-pcc] expert_dtype={expert_dtype} (EXPERT_DTYPE={os.getenv('EXPERT_DTYPE', 'bf4')})", flush=True
        )
        force_load = os.getenv("M3_FORCE_LOAD_WEIGHTS") == "1"
        cache_only = not force_load and (
            os.getenv("M3_WEIGHTS_FROM_CACHE") == "1"
            or weight_cache_is_complete(cache_path, hf_config, num_layers, expert_dtype)
        )
        if cache_only:
            print(
                "[prefill-pcc] tilized weight cache complete -> loading from cache, "
                "skipping the ~869GB bf16 source read",
                flush=True,
            )
            state_dict = {}
        else:
            print("[prefill-pcc] loading real bf16 weights + EP placement (slow: bf16 source read) ...", flush=True)
            state_dict = ModelArgs.load_state_dict(model_args.weights_path)
        cfg = TtPrefillRuntimeConfig(
            num_layers=num_layers,
            max_seq_len=total,
            mesh_shape=(rows, cols),
            chunk_size=chunk,
            num_users=1,
            weight_cache_path=cache_path,
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
