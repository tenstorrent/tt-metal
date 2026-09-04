# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Measure legal terminal-projection geometries on one Muse-Glimmer topology.

The selected four-chip LM head projects only one quarter of the vocabulary on
each device.  P150 and P150x2 project wider local shards, so a geometry that is
legal on P150x4 can exceed per-core L1 before the first request.  This probe
builds only the real-shape LM head and a 32-row activation, making topology
bring-up iterations much cheaper than rebuilding all 52 decoder layers.
"""

from __future__ import annotations

import argparse
import math
import time

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.generator import close_generator_mesh, open_generator_mesh
from models.autoports.meta_models_muse_glimmer_30b.tt.model import padded_vocab_size
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import (
    dram_sharded_weight_memcfg,
    width_sharded_l1,
)

HIDDEN = 6656
VOCAB = 202048
ROWS = 32
TILE = 32


def _program_config(contract: str, *, local_vocab: int, cores: int, in0_block_w: int, grid):
    per_core_n = math.ceil(local_vocab / (TILE * cores))
    if contract == "dram_sharded":
        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=in0_block_w,
            per_core_M=1,
            per_core_N=per_core_n,
        )
    out_subblock_w = min(per_core_n, 4)
    while out_subblock_w > 1 and per_core_n % out_subblock_w:
        out_subblock_w -= 1
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--contract", choices=("dram_sharded", "mcast1d"), required=True)
    parser.add_argument("--cores", type=int, required=True)
    parser.add_argument("--in0-block-w", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    if args.contract == "dram_sharded":
        if args.cores > 110 or 208 % args.cores:
            raise SystemExit("--cores must be a <=110 divisor of the 208 K tiles")
        if (208 // args.cores) % args.in0_block_w:
            raise SystemExit("--in0-block-w must divide the per-core K tiles")
    elif 208 % args.in0_block_w:
        raise SystemExit("--in0-block-w must divide the 208 K tiles")

    mesh = open_generator_mesh(mesh_shape=(1, args.tp), trace_region_size=0)
    try:
        grid = mesh.compute_with_storage_grid_size()
        if args.contract == "mcast1d" and args.cores != grid.x * grid.y:
            raise ValueError(f"mcast1d uses the full {grid.x}x{grid.y} grid ({grid.x * grid.y} cores)")
        dram_banks = mesh.dram_grid_size().x
        total_vocab = padded_vocab_size(
            VOCAB,
            args.tp,
            cores=dram_banks if args.contract == "dram_sharded" else None,
        )
        local_vocab = total_vocab // args.tp
        print(
            f"PROBE tp={args.tp} contract={args.contract} cores={args.cores} "
            f"in0={args.in0_block_w} local_vocab={local_vocab}",
            flush=True,
        )

        torch.manual_seed(7)
        host_weight = torch.zeros(1, 1, HIDDEN, total_vocab, dtype=torch.bfloat16)
        host_weight[0, 0, :TILE, :TILE] = torch.eye(TILE, dtype=torch.bfloat16)
        weight_memcfg = (
            dram_sharded_weight_memcfg(HIDDEN, local_vocab, mesh)
            if args.contract == "dram_sharded"
            else ttnn.DRAM_MEMORY_CONFIG
        )
        weight = ttnn.from_torch(
            host_weight,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat4_b,
            memory_config=weight_memcfg,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        )
        del host_weight

        host_input = torch.randn(1, 1, ROWS, HIDDEN, dtype=torch.bfloat16)
        input_memcfg = (
            width_sharded_l1(ROWS, HIDDEN, args.cores, grid)
            if args.contract == "dram_sharded"
            else ttnn.DRAM_MEMORY_CONFIG
        )
        output_memcfg = (
            width_sharded_l1(ROWS, local_vocab, args.cores, grid)
            if args.contract == "dram_sharded"
            else ttnn.DRAM_MEMORY_CONFIG
        )
        activation = ttnn.from_torch(
            host_input,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=input_memcfg,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        program_config = _program_config(
            args.contract,
            local_vocab=local_vocab,
            cores=args.cores,
            in0_block_w=args.in0_block_w,
            grid=grid,
        )
        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        def run():
            return ttnn.linear(
                activation,
                weight,
                dtype=ttnn.bfloat16,
                memory_config=output_memcfg,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
            )

        output = run()
        ttnn.synchronize_device(mesh)
        device_zero = ttnn.get_device_tensors(output)[0]
        if device_zero.is_sharded():
            device_zero = ttnn.sharded_to_interleaved(device_zero, ttnn.DRAM_MEMORY_CONFIG)
        got = ttnn.to_torch(device_zero).reshape(ROWS, -1)[:, :TILE].float()
        want = host_input.reshape(ROWS, HIDDEN)[:, :TILE].float()
        max_abs = float((got - want).abs().max())
        if max_abs > 0.125:
            raise RuntimeError(f"identity-tile check failed: max_abs={max_abs}")
        ttnn.deallocate(output)

        started = time.perf_counter()
        for _ in range(args.rounds):
            output = run()
            ttnn.deallocate(output)
        ttnn.synchronize_device(mesh)
        latency_ms = (time.perf_counter() - started) * 1000 / args.rounds
        print(f"PROBE_PASS latency_ms={latency_ms:.4f} identity_max_abs={max_abs:.6f}", flush=True)
        ttnn.deallocate(activation)
        ttnn.deallocate(weight)
        return 0
    finally:
        close_generator_mesh(mesh)


if __name__ == "__main__":
    raise SystemExit(main())
