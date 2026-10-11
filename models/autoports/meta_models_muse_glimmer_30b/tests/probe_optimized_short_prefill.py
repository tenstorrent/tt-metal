# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Integrated small-M DRAM-sharded prefill candidate, retaining all attention semantics."""

import argparse
import json
import math
from pathlib import Path

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tests.sweep_optimized import OptimizedDecoder, main

linear = OptimizedDecoder._linear
mlp = OptimizedDecoder._mlp


def short(layer, x):
    return not layer._decode and layer.policy.get("short_dram") and x.padded_shape[-2] <= 64


def short_linear(layer, x, name):
    if not short(layer, x):
        return linear(layer, x, name)
    height = x.padded_shape[-2]
    k, n = list(layer.weights[name].shape)[-2:]
    grid = tuple(layer.policy.get(name + ".grid", layer.grid))
    mem = layer._width_mem(k, grid, exact=True, height=height)
    block = mem.shard_spec.shape[1] // 32 if height > 32 else layer.decode_pc[name].in0_block_w
    pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=block,
        per_core_M=height // 32,
        per_core_N=math.ceil(n / (32 * mem.shard_spec.grid.num_cores())),
        num_workers_per_dram_bank=2,
    )
    out = ttnn.linear(
        ttnn.to_memory_config(x, mem),
        layer.weights[name],
        dtype=ttnn.bfloat16,
        compute_kernel_config=layer.mm_ck[name],
        program_config=pc,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
    )
    # Prefill head splitting consumes interleaved packed heads.
    return ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG) if name == "qkv" else out


def short_mlp(layer, x):
    if not short(layer, x):
        return mlp(layer, x)
    packed = short_linear(layer, x, "mlp.gate_up")
    shape = list(packed.shape)
    gate = ttnn.slice(packed, [0, 0, 0, 0], shape[:-1] + [19968])
    up = ttnn.slice(packed, [0, 0, 0, 19968], shape)
    product = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
    return short_linear(layer, product, "mlp.down_proj")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--length", type=int, default=33)
    p.add_argument("--chunk32", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output = args.output.resolve()
    args.batch = 1
    args.candidates = ["default", "short_dram"]
    args.policies = args.output.with_suffix(".policies.json")
    candidate = OptimizedDecoder.POLICY | {"short_dram": True}
    if args.chunk32:
        candidate["chunk_size"] = 32
    args.policies.write_text(json.dumps({"short_dram": candidate}, indent=2))
    OptimizedDecoder._linear = short_linear
    OptimizedDecoder._mlp = short_mlp
    args.output.with_suffix(".probe.py").write_bytes(Path(__file__).read_bytes())
    main(args)
