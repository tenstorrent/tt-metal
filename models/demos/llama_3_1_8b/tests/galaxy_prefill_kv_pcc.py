# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Llama 3.1 8B REAL-WEIGHTS prefill: throughput + per-layer KV-cache PCC on the galaxy.

Mirrors ``gpt_oss_d_p/tests/galaxy_prefill_kv_pcc.py``. Builds the full 32-layer model with real
bf16 weights via ``TtPrefillRuntime``, runs prefill over the golden trace's prompt, measures
throughput, then PCC-checks every layer's post-RoPE K / raw V against the golden trace.

GALAXY-GATED: needs the (8,4) Blackhole galaxy (32 devices) for SP=8 + TP=4. Auto-SKIPs when not
on a galaxy or when no golden trace is provided.

Env:
  PREFILL_TRACE_DIR   golden trace dir (metadata.json + kv_cache/layer_N.safetensors)     [required]
  PREFILL_HF_MODEL    real Llama weights dir                                              [default /mnt/models/meta-llama/Llama-3.1-8B-Instruct]
  PREFILL_CHUNKED     "1" -> chunked (SP ring cache-read); "0" -> one-shot                [default 0]
  PREFILL_CHUNK_SIZE  chunk size in tokens (chunked mode only)                            [default 5120]
  PREFILL_TPS_ITERS   prefill repetitions for throughput measurement                      [default 1]
  PREFILL_NUM_LAYERS  build/run only the first N decoder layers                           [default: all]
  LLAMA_WEIGHTS_FROM_CACHE  "1" -> pass empty state_dict (load tilized weights from TTNN cache)
  PREFILL_TTNN_CACHE  TTNN weight-cache root                                              [optional]

Run (single Blackhole galaxy, after weights + golden are staged):
  export PREFILL_HF_MODEL=/mnt/models/meta-llama/Llama-3.1-8B-Instruct
  export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto
  PREFILL_TRACE_DIR=/mnt/models/meta-llama/Llama-3.1-8B-Instruct/golden/synthetic_10240 \\
    python3 models/demos/llama_3_1_8b/tests/galaxy_prefill_kv_pcc.py
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

GALAXY_NUM_DEVICES = 32
ROWS, COLS = 8, 4  # SP=8 (rows), TP=4 (cols)


def _raise_nproc_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    if soft != resource.RLIM_INFINITY and (hard == resource.RLIM_INFINITY or soft < hard):
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))
            print(f"[prefill-pcc] raised RLIMIT_NPROC soft {soft} -> {hard}")
        except (ValueError, OSError) as e:
            print(f"[prefill-pcc] WARNING: could not raise RLIMIT_NPROC (soft={soft}): {e}", file=sys.stderr)


def plan(n_tokens, chunk_size, chunked, sp):
    """Resolve (n_chunks, chunk, total)."""
    align = ttnn.TILE_SIZE * sp  # 32 * 8 = 256
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
            f"[prefill-pcc] SKIP: needs the (8,4) galaxy ({GALAXY_NUM_DEVICES} devices) for SP=8 + TP=4; "
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

    from models.demos.llama_3_1_8b.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

    _linear = os.getenv("PREFILL_TOPOLOGY", "ring") == "linear"
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D if _linear else ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(ROWS, COLS))
    print(f"[prefill-pcc] mesh opened {tuple(mesh.shape)} ndev={mesh.get_num_devices()}", flush=True)
    try:
        from transformers import AutoConfig

        model_path = os.environ.get("PREFILL_HF_MODEL") or "/mnt/models/meta-llama/Llama-3.1-8B-Instruct"
        hf_config = AutoConfig.from_pretrained(model_path)
        hf_config = getattr(hf_config, "text_config", hf_config)
        num_layers = hf_config.num_hidden_layers
        nl_override = os.getenv("PREFILL_NUM_LAYERS")
        if nl_override:
            num_layers = int(nl_override)
            hf_config.num_hidden_layers = num_layers
            print(f"[prefill-pcc] PREFILL_NUM_LAYERS={num_layers}: first {num_layers} layers only", flush=True)

        kv_cache_dtype = ttnn.bfloat16 if os.getenv("KV_CACHE_DTYPE", "bf8") == "bf16" else ttnn.bfloat8_b
        print(f"[prefill-pcc] kv_cache_dtype={kv_cache_dtype}", flush=True)

        env_cache = os.environ.get("PREFILL_TTNN_CACHE", "")
        cache_path = None
        if env_cache:
            cache_path = Path(env_cache) / f"llama_3_1_8b_bh_{ROWS * COLS}dev" / f"{ROWS}x{COLS}"
            cache_path.mkdir(parents=True, exist_ok=True)

        if os.getenv("LLAMA_WEIGHTS_FROM_CACHE") == "1":
            print("[prefill-pcc] LLAMA_WEIGHTS_FROM_CACHE=1 -> empty state_dict (load tilized cache)", flush=True)
            state_dict = {}
        else:
            from models.demos.llama_3_1_8b.tt.runners.adapters.llama_3_1_8b import _load_safetensors_state_dict

            print(f"[prefill-pcc] loading real bf16 weights from {model_path} ...", flush=True)
            state_dict = _load_safetensors_state_dict(model_path)

        cfg = TtPrefillRuntimeConfig(
            num_layers=num_layers,
            max_seq_len=total,
            mesh_shape=(ROWS, COLS),
            default_chunk_size=chunk,
            num_users=1,
            cache_dtype=kv_cache_dtype,
            weight_cache_path=cache_path,
            owns_kv_cache=True,
            topology=ttnn.Topology.Linear if _linear else ttnn.Topology.Ring,
        )
        runtime = TtPrefillRuntime(mesh, hf_config, state_dict, cfg)
        del state_dict

        print(f"[prefill-pcc] compiling ({num_layers}L, SP={ROWS} × TP={COLS}) ...", flush=True)
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

        min_pcc = runtime.kv_cache_pcc_check(slot_id=0, n_chunks=n_chunks, trace_dir=golden_dir)
        print(f"[prefill-pcc] min KV PCC across {num_layers} layers = {min_pcc:.5f}", flush=True)
        _pcc_min = os.environ.get("LLAMA_KV_PCC_MIN")
        if _pcc_min is not None and min_pcc < float(_pcc_min):
            print(f"[prefill-pcc] FAIL: min KV PCC {min_pcc:.5f} < LLAMA_KV_PCC_MIN={_pcc_min}", flush=True)
            return 1
        print("[prefill-pcc] DONE", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
