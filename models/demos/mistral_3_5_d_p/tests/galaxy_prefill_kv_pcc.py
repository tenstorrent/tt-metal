#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""REAL-WEIGHTS Mistral-Medium-3.5 prefill: throughput + per-layer KV-cache PCC on the galaxy.

Pattern: ``minimax_m3/tests/galaxy_prefill_kv_pcc.py`` (and its gpt-oss twin). This is the P1 and P2
gate: build the model with real checkpoint weights through ``TtPrefillRuntime``, run prefill over the
golden trace's prompt, measure throughput, then PCC every layer's post-RoPE K and raw V against the
golden trace. Dense GQA, so there is no index_k to compare.

  ``PREFILL_CHUNKED=0`` (default) — one-shot: a single chunk covering the whole prompt. P1's gate.
  ``PREFILL_CHUNKED=1``           — multi-chunk: chunk N attends the prefix chunks 0..N-1 left in the
                                    cache, and must reach the same per-layer PCC. P2's gate.

GALAXY-GATED: needs the spec's target mesh (32 devices). Auto-SKIPs (exit 0) when the mesh is
smaller or no golden trace is provided, so it is safe to invoke anywhere.

Env:
  PREFILL_TRACE_DIR   golden trace dir (metadata.json + kv_cache/layer_N.safetensors)   [required]
  HF_MODEL            checkpoint dir read by ModelArgs. Defaults to the trace dir, which is where
                      the generator leaves ``reference_weights.pt`` — the weights the golden was
                      produced FROM, which is what makes this comparison meaningful.
  PREFILL_CHUNKED     "1" -> chunked; "0" -> one-shot                                    [0]
  PREFILL_CHUNK_SIZE  chunk size in tokens (chunked mode)                    [the spec's 5120]
  PREFILL_NUM_LAYERS  build/run only the first N layers (a partial-model run)      [all]
  PREFILL_TPS_ITERS   prefill repetitions for the throughput measurement             [1]
  MISTRAL_LINEAR_FABRIC  "1" (default) -> FABRIC_1D + Topology.Linear (a plain-grid galaxy);
                      "0" -> FABRIC_1D_RING + the torus descriptor
  MISTRAL_KV_PCC_MIN  gate: exit non-zero when the min PCC falls below this   [unset -> report only]
  MISTRAL_WEIGHT_DTYPE  "bf8" (the spec) or "bf16" — a bring-up A/B of the weight dataformat's cost
                      on per-layer KV PCC at depth. NOT a serving knob: the spec fixes bf8.  [bf8]

Run:
  export TT_METAL_HOME=$PWD TT_METAL_RUNTIME_ROOT=$PWD
  export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto
  PREFILL_TRACE_DIR=/path/to/golden python3 models/demos/mistral_3_5_d_p/tests/galaxy_prefill_kv_pcc.py
"""

from __future__ import annotations

import json
import math
import os
import resource
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import ttnn  # noqa: E402

from models.demos.mistral_3_5_d_p.spec import SPEC  # noqa: E402

ROWS, COLS = SPEC.mesh_shape  # (4, 8): SP=4 rows, TP=8 cols
GALAXY_NUM_DEVICES = ROWS * COLS


def raise_nproc_limit():
    """Raise RLIMIT_NPROC to the hard limit so tt-metal's parallel kernel JIT (a burst of g++ / make
    processes) does not starve with EAGAIN mid-build."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    if soft != resource.RLIM_INFINITY and (hard == resource.RLIM_INFINITY or soft < hard):
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))
            print(f"[prefill-pcc] raised RLIMIT_NPROC soft {soft} -> {hard}", flush=True)
        except (ValueError, OSError) as e:
            print(f"[prefill-pcc] WARNING: could not raise RLIMIT_NPROC (soft={soft}): {e}", file=sys.stderr)


def plan(n_tokens, chunk_size, chunked, sp):
    """Resolve ``(n_chunks, chunk, capacity)``.

    Both modes must satisfy the block-cyclic constraints: the chunk is a multiple of
    ``TILE_SIZE * sp`` (so each SP shard is tile-aligned) and the CAPACITY is a whole number of
    chunks (so the whole-cache indexed rope tiles it).

    one-shot: one chunk covering the prompt, padded up to the alignment.
    chunked:  ``chunk_size`` rounded up to the alignment, enough chunks to cover the prompt.
    """
    align = ttnn.TILE_SIZE * sp
    if chunked:
        chunk = math.ceil(chunk_size / align) * align
        n_chunks = max(1, math.ceil(n_tokens / chunk))
    else:
        chunk = max(align, math.ceil(n_tokens / align) * align)
        n_chunks = 1
    return n_chunks, chunk, n_chunks * chunk


def main():
    raise_nproc_limit()

    golden_dir = os.environ.get("PREFILL_TRACE_DIR")
    if not golden_dir:
        print("[prefill-pcc] SKIP: set PREFILL_TRACE_DIR to a golden trace dir", flush=True)
        return 0
    if ttnn.get_num_devices() < GALAXY_NUM_DEVICES:
        print(
            f"[prefill-pcc] SKIP: needs the spec's target mesh {SPEC.mesh_shape} "
            f"({GALAXY_NUM_DEVICES} devices); have {ttnn.get_num_devices()}",
            flush=True,
        )
        return 0

    metadata = json.load(open(Path(golden_dir) / "metadata.json"))
    token_ids = list(metadata["token_ids"])
    n_tokens = len(token_ids)
    chunked = os.getenv("PREFILL_CHUNKED", "0") == "1"
    chunk_size = int(os.getenv("PREFILL_CHUNK_SIZE", str(SPEC.chunk_size)))
    tps_iters = int(os.getenv("PREFILL_TPS_ITERS", "1"))

    n_chunks, chunk, capacity = plan(n_tokens, chunk_size, chunked, ROWS)
    print(
        f"[prefill-pcc] golden={golden_dir} n_tokens={n_tokens} "
        f"mode={'chunked' if chunked else 'one-shot'} chunk={chunk} n_chunks={n_chunks} "
        f"capacity={capacity} tps_iters={tps_iters}",
        flush=True,
    )

    from models.demos.mistral_3_5_d_p.tests.test_factory import linear_fabric
    from models.demos.mistral_3_5_d_p.tt.model_config import ModelArgs
    from models.demos.mistral_3_5_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

    linear = linear_fabric()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D if linear else ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(ROWS, COLS))
    print(f"[prefill-pcc] mesh opened {tuple(mesh.shape)} ndev={mesh.get_num_devices()} linear={linear}", flush=True)
    try:
        # The weights the golden was produced from live next to it unless HF_MODEL overrides.
        os.environ.setdefault("HF_MODEL", golden_dir)
        model_args = ModelArgs(mesh_device=mesh, max_seq_len=capacity)
        hf_config = model_args.hf_config

        num_layers = int(os.getenv("PREFILL_NUM_LAYERS") or metadata.get("num_layers", hf_config.num_hidden_layers))
        hf_config.num_hidden_layers = num_layers
        # The golden's dims win: a reduced-width golden must be compared against a model of the same
        # width, or every PCC is meaningless.
        for key, attr in (
            ("hidden_size", "hidden_size"),
            ("intermediate_size", "intermediate_size"),
            ("vocab_size", "vocab_size"),
        ):
            if key in metadata:
                setattr(hf_config, attr, metadata[key])
        weight_override = os.getenv("MISTRAL_WEIGHT_DTYPE", "bf8")
        print(f"[prefill-pcc] weight dataformat: {weight_override} (spec default: bf8)", flush=True)
        print(
            f"[prefill-pcc] layers={num_layers} hidden={hf_config.hidden_size} "
            f"inter={hf_config.intermediate_size} vocab={hf_config.vocab_size} "
            f"kv_heads={hf_config.num_key_value_heads} head_dim={hf_config.head_dim}",
            flush=True,
        )

        print("[prefill-pcc] loading real weights through the production loader ...", flush=True)
        state_dict = ModelArgs.load_state_dict(model_args.weights_path, num_layers=num_layers)

        config = TtPrefillRuntimeConfig(
            num_layers=num_layers,
            max_seq_len=capacity,
            servable_seq_len=capacity,  # this harness serves exactly the capacity it allocated
            mesh_shape=(ROWS, COLS),
            default_chunk_size=chunk,
            num_users=1,
            cache_dtype=SPEC.kv_cache_dtype,
            weight_cache_path=None,
            weight_dtype=(ttnn.bfloat16 if os.getenv("MISTRAL_WEIGHT_DTYPE") == "bf16" else None),
            owns_kv_cache=True,  # the standalone harness owns its cache
            topology=ttnn.Topology.Linear if linear else ttnn.Topology.Ring,
        )
        runtime = TtPrefillRuntime(mesh, hf_config, state_dict, config)
        del state_dict

        print(f"[prefill-pcc] compiling ({num_layers}L, SP={ROWS} x TP={COLS}) ...", flush=True)
        runtime.compile()

        padded = token_ids + [0] * (capacity - n_tokens)

        def run_once():
            for c in range(n_chunks):
                start = c * chunk
                inp = runtime.make_chunk_input(padded[start : start + chunk])
                runtime.prefill_chunk(
                    inp, slot_id=0, actual_start=start, actual_end=min(start + chunk, max(n_tokens, start + 1))
                )
            ttnn.synchronize_device(mesh)

        times = []
        for i in range(tps_iters):
            t0 = time.perf_counter()
            run_once()
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            print(
                f"[prefill-pcc] iter {i}: {elapsed * 1000:.1f} ms  "
                f"{n_tokens / elapsed:.1f} tok/s (real)  {capacity / elapsed:.1f} tok/s (incl pad)",
                flush=True,
            )
        median = statistics.median(times)
        print(
            f"[prefill-pcc] THROUGHPUT over {tps_iters} iters: median {n_tokens / median:.1f} tok/s (real), "
            f"{capacity / median:.1f} tok/s (processed); wall median {median * 1000:.1f} ms",
            flush=True,
        )

        min_pcc = runtime.kv_cache_pcc_check(
            slot_id=0, n_chunks=n_chunks, trace_dir=golden_dir, chunk_size=chunk, real_len=n_tokens
        )
        print(f"[prefill-pcc] min KV PCC across {num_layers} layers = {min_pcc:.5f}", flush=True)

        floor = os.environ.get("MISTRAL_KV_PCC_MIN")
        if floor is not None and min_pcc < float(floor):
            print(f"[prefill-pcc] FAIL: min KV PCC {min_pcc:.5f} < MISTRAL_KV_PCC_MIN={floor}", flush=True)
            return 1
        print("[prefill-pcc] DONE", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
