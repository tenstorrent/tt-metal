#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""P1 / P2 — per-layer KV on the Blackhole Galaxy vs the CPU golden trace, with REAL weights.

This is the graded artifact of the bring-up. Everything before it runs on random weights; this is
the first and only place where real weights, the full layer count, the target parallelism and the
chunked-prefill machinery all interact.

    # P1 — one-shot (no chunking)
    export TT_METAL_HOME=$PWD TT_METAL_RUNTIME_ROOT=$PWD
    export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto
    export HF_MODEL=/mnt/models/meta-llama/Llama-3.1-8B-Instruct
    PREFILL_CHUNKED=0 \\
    PREFILL_TRACE_DIR=/mnt/models/meta-llama/Llama-3.1-8B-Instruct/golden/synthetic_5120 \\
      python3 models/demos/llama_3_1_8b_d_p/tests/galaxy_prefill_kv_pcc.py

    # P2 — multi-chunk, exercising the ring cache-read path
    PREFILL_CHUNKED=1 PREFILL_CHUNK_SIZE=5120 \\
    PREFILL_TRACE_DIR=/mnt/models/meta-llama/Llama-3.1-8B-Instruct/golden/synthetic_10240 \\
      python3 models/demos/llama_3_1_8b_d_p/tests/galaxy_prefill_kv_pcc.py

Both modes compare against the SAME kind of trace: chunk N attending the prefix chunks 0..N-1 left
in the cache must produce the same KV as processing the whole sequence at once.

## Two conventions the comparison has to undo

**Meta head-dim order.** The device stores K with its head-dim columns in Meta interleaved order,
because q/k are permuted at load so the rope ops can consume Meta tables
(`utils/weight_conversion.py`). The trace stores K in HF half-split order — it is a property of the
model, not of this package — so the golden is permuted here. Skipping this reads as UNCORRELATED
(PCC ~0.008), which is at least an unmissable failure. V is never rotated and needs no permutation.

**Block-cyclic sequence layout.** The cache is SP-sharded block-cyclic with period `chunk_size`, so
natural position `p` lives at a computed `(chip, local row)`. `_natural_order_index` inverts it.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import ttnn  # noqa: E402
from models.demos.llama_3_1_8b_d_p.config import MeshConfig  # noqa: E402
from models.demos.llama_3_1_8b_d_p.reference.config import LlamaConfigConstants  # noqa: E402
from models.demos.llama_3_1_8b_d_p.reference.model import REF_DTYPE  # noqa: E402
from models.demos.llama_3_1_8b_d_p.tt.attention.kv_cache import cache_capacity  # noqa: E402
from models.demos.llama_3_1_8b_d_p.tt.ccl import CCLManager, default_topology  # noqa: E402
from models.demos.llama_3_1_8b_d_p.tt.model import Model  # noqa: E402
from models.demos.llama_3_1_8b_d_p.tt.model_config import ModelArgs, resolve_weights_path  # noqa: E402
from models.demos.llama_3_1_8b_d_p.utils.general_utils import get_default_num_links  # noqa: E402
from models.demos.llama_3_1_8b_d_p.utils.weight_conversion import hf_to_meta_head_dim  # noqa: E402

SPEC = json.loads((Path(__file__).resolve().parent.parent / "llama_3_1_8b.spec.json").read_text())
PCC_TARGET = SPEC["acceptance"]["pcc_target"]
PCC_LOWER_BOUND = SPEC["acceptance"]["pcc_lower_bound"]
TARGET_MESH = (SPEC["parallelism"]["sp"], SPEC["parallelism"]["tp"])


def comp_pcc(golden: torch.Tensor, got: torch.Tensor) -> float:
    a, b = golden.detach().float().flatten(), got.detach().float().flatten()
    assert a.shape == b.shape, f"shape mismatch {tuple(golden.shape)} vs {tuple(got.shape)}"
    a, b = a - a.mean(), b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if torch.allclose(a, b) else 0.0
    return min(1.0, max(-1.0, float((a @ b) / denom)))


def load_trace(trace_dir: Path):
    """metadata + the per-layer (K, V) tensors the device is graded against."""
    metadata = json.loads((trace_dir / "metadata.json").read_text())
    kv_dir = trace_dir / "kv_cache"
    layers = []
    for i in range(metadata["num_layers"]):
        t = load_file(str(kv_dir / f"layer_{i}.safetensors"))
        layers.append((t[f"key_cache_layer_{i}"], t[f"value_cache_layer_{i}"]))
    return metadata, layers


def _natural_order_index(n_tokens: int, chunk_global: int, sp: int, capacity: int) -> torch.Tensor:
    """Cache row index for each natural token position, inverting the block-cyclic layout.

    Position `p` sits in slab `p // chunk_global`, on chip `(p % chunk_global) // chunk_local`, at
    local row `slab * chunk_local + (p % chunk_local)`. Read back with the sequence concatenated
    over the SP rows, chip `c` occupies rows `[c * capacity/sp, (c+1) * capacity/sp)`.
    """
    chunk_local = chunk_global // sp
    tokens_per_dev = capacity // sp
    p = torch.arange(n_tokens)
    chip = (p % chunk_global) // chunk_local
    local_row = (p // chunk_global) * chunk_local + (p % chunk_local)
    return chip * tokens_per_dev + local_row


def main() -> int:
    trace_dir = os.getenv("PREFILL_TRACE_DIR")
    if not trace_dir:
        logger.error("set PREFILL_TRACE_DIR to a golden trace directory")
        return 2
    trace_dir = Path(trace_dir)
    chunked = os.getenv("PREFILL_CHUNKED", "0") == "1"
    num_layers_override = os.getenv("PREFILL_NUM_LAYERS")

    metadata, golden_layers = load_trace(trace_dir)
    token_ids = metadata["token_ids"]
    n_tokens = metadata["n_tokens"]
    config = LlamaConfigConstants.from_json()

    num_layers = int(num_layers_override) if num_layers_override else metadata["num_layers"]
    reduced = num_layers != config.num_hidden_layers or metadata.get("reduced_depth", False)
    if reduced:
        logger.warning(
            f"REDUCED RUN: {num_layers} of {config.num_hidden_layers} layers. This is a diagnostic, "
            "not a bring-up grade — label every number it produces as reduced."
        )

    chunk_size = int(os.getenv("PREFILL_CHUNK_SIZE", SPEC["shapes"]["chunk_size"])) if chunked else n_tokens
    assert n_tokens % chunk_size == 0, (
        f"trace has {n_tokens} tokens, which is not a whole number of {chunk_size}-token chunks"
    )
    n_chunks = n_tokens // chunk_size
    logger.info(
        f"{'CHUNKED' if chunked else 'ONE-SHOT'}: {n_tokens} tokens, {n_chunks} x {chunk_size}, {num_layers} layers"
    )

    weights_path = resolve_weights_path(required=True)
    logger.info(f"REAL weights: {weights_path}")

    # The fabric must be configured BEFORE the mesh opens — a standalone script gets none of the
    # pytest device fixture's setup, and opening without it dies with "Trying to get un-initialized
    # fabric context". FABRIC_1D + Topology.Linear: this galaxy is a plain grid with no wrap-around
    # links, so a ring fabric cannot map on it.
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(*TARGET_MESH))
    try:
        rows, cols = tuple(mesh_device.shape)
        assert (rows, cols) == TARGET_MESH, f"opened {(rows, cols)}, spec asks {TARGET_MESH}"
        mesh_config = MeshConfig(mesh_device.shape, tp=cols)
        ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=default_topology())
        sp = mesh_config.sp
        assert n_tokens % sp == 0 and chunk_size % (32 * sp) == 0

        model_args = ModelArgs(mesh_device=mesh_device)
        state_dict = ModelArgs.load_state_dict(weights_path)
        model = Model(
            mesh_device,
            config.to_hf_config(),
            state_dict,
            ccl_manager=ccl,
            mesh_config=mesh_config,
            max_seq_len=n_tokens,
            chunk_size=chunk_size,
            num_layers=num_layers,
            tensor_cache_path=model_args.weight_cache_path("bfp8"),
        )
        del state_dict
        # PREFILL_KV_DTYPE is a DIAGNOSTIC knob, not a configuration: the spec binds the cache to
        # bfloat8_b. Its only use is attributing a PCC gap — running bf16 says how much of the loss
        # is cache quantization and how much is accumulated model error. Any number produced with a
        # non-spec dtype must be labelled as a diagnostic.
        kv_dtype_name = os.getenv("PREFILL_KV_DTYPE", SPEC["dataformats"]["kv_cache"]["default"])
        kv_dtype = {"bfloat8_b": ttnn.bfloat8_b, "bfloat16": ttnn.bfloat16, "float32": ttnn.float32}[kv_dtype_name]
        if kv_dtype_name != SPEC["dataformats"]["kv_cache"]["default"]:
            logger.warning(
                f"DIAGNOSTIC: KV cache dtype {kv_dtype_name}, not the spec's "
                f"{SPEC['dataformats']['kv_cache']['default']}. This run is not a bring-up grade."
            )
        kv = model.allocate_kv_cache(cache_dtype=kv_dtype)
        capacity = cache_capacity(n_tokens, chunk_size)
        logger.info(f"KV cache: {num_layers} layers, capacity {capacity} tokens, {kv.n_kv_local} KV heads/chip")

        # Rope tables for the whole sequence, in the device's Meta order, sliced per chunk below.
        head_dim = config.head_dim
        cos_all = ttnn.to_torch(ttnn.get_device_tensors(model.rope_setup.cos_matrix)[0]).reshape(-1, head_dim)
        sin_all = ttnn.to_torch(ttnn.get_device_tensors(model.rope_setup.sin_matrix)[0]).reshape(-1, head_dim)

        sp_dims = [None, None]
        sp_dims[mesh_config.sp_axis] = 2

        def sp_shard(t, dtype):
            return ttnn.from_torch(
                t,
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT if dtype != ttnn.uint32 else ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=sp_dims),
            )

        ids = torch.tensor(token_ids, dtype=torch.int32).reshape(1, 1, n_tokens)
        start = time.time()
        for c in range(n_chunks):
            lo, hi = c * chunk_size, (c + 1) * chunk_size
            # The runtime contract: a chunk must be chunk-aligned and inside the cache.
            assert lo % chunk_size == 0 and hi <= capacity, f"chunk [{lo}, {hi}) is out of contract"
            model.forward(
                sp_shard(ids[:, :, lo:hi], ttnn.uint32),
                rope_mats=(
                    sp_shard(cos_all[lo:hi].reshape(1, 1, chunk_size, head_dim), ttnn.bfloat16),
                    sp_shard(sin_all[lo:hi].reshape(1, 1, chunk_size, head_dim), ttnn.bfloat16),
                ),
                kv_cache=kv,
                cached_len=lo,
                logical_n=hi,
                return_logits=False,
            )
            logger.info(f"chunk {c + 1}/{n_chunks} [{lo}, {hi}) done")
        ttnn.synchronize_device(mesh_device)
        elapsed = time.time() - start
        logger.info(f"prefill: {elapsed:.2f}s for {n_tokens} tokens -> {n_tokens / elapsed:.0f} tok/s")

        concat = [None, None]
        concat[mesh_config.sp_axis] = 2
        concat[mesh_config.tp_axis] = 1

        def readback(cache):
            return ttnn.to_torch(
                cache,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(concat), mesh_shape=(rows, cols)),
            ).to(REF_DTYPE)

        host_k, host_v = readback(kv.k), readback(kv.v)
        # chunk_global == chunk_size: the chunk length is already the GLOBAL token count, and the
        # SP sharding splits it into chunk_size/sp rows per device.
        idx = _natural_order_index(n_tokens, chunk_size, sp, capacity)

        worst_k = worst_v = 1.0
        failures = []
        for layer in range(num_layers):
            slot = 0 * num_layers + layer  # user-major packing
            k_ref, v_ref = golden_layers[layer]
            # Golden K is HF order; the device cache is Meta order. Permute the golden, not the device.
            k_ref = hf_to_meta_head_dim(k_ref[0].to(REF_DTYPE))
            v_ref = v_ref[0].to(REF_DTYPE)
            pcc_k = comp_pcc(k_ref, host_k[slot][:, idx, :])
            pcc_v = comp_pcc(v_ref, host_v[slot][:, idx, :])
            worst_k, worst_v = min(worst_k, pcc_k), min(worst_v, pcc_v)
            logger.info(f"layer {layer:2d}: K {pcc_k:.6f}  V {pcc_v:.6f}")
            if min(pcc_k, pcc_v) < PCC_LOWER_BOUND:
                failures.append((layer, pcc_k, pcc_v))

        label = "CHUNKED" if chunked else "ONE-SHOT"
        suffix = " [REDUCED]" if reduced else ""
        logger.info(
            f"{label}{suffix} {n_tokens} tok, {num_layers} layers, linear fabric: "
            f"min K {worst_k:.6f} / min V {worst_v:.6f} "
            f"(target {PCC_TARGET}, bound {PCC_LOWER_BOUND})"
        )
        if failures:
            for layer, pk, pv in failures:
                logger.error(f"layer {layer} below the bound: K {pk:.6f} V {pv:.6f}")
            return 1
        if min(worst_k, worst_v) < PCC_TARGET:
            logger.warning(
                f"min PCC {min(worst_k, worst_v):.6f} is below pcc_target {PCC_TARGET}; record the "
                "value and the reason in README.md"
            )
        return 0
    finally:
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    raise SystemExit(main())
