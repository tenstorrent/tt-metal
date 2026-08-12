# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Can one persistent weight tensor serve both the decode and prefill matmul?

The optimized decode path wants a DRAM **width-sharded** weight for
``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig``; the prefill path wants
whatever ``ttnn.experimental.minimal_matmul`` accepts.  Holding two copies would
double the layer's 315 MB weight footprint (and x52 in a full model), so this
probe answers, on device, which of the four combinations are legal:

1. DRAM-sharded decode matmul with a DRAM **interleaved** weight;
2. DRAM-sharded decode matmul with a DRAM width-sharded weight (the reference
   form, expected to work);
3. ``minimal_matmul`` at prefill rows with a DRAM width-sharded weight;
4. ``ttnn.linear`` at prefill rows with a DRAM width-sharded weight.

Each case prints ``OK`` plus latency and PCC, or ``BLOCKED`` plus the exact
op-contract message.
"""

from __future__ import annotations

import math
import time

import torch

import ttnn

TILE = 32
K, N = 6656, 4608
DECODE_ROWS = 32
PREFILL_ROWS = 8192


def dram_width_sharded(k: int, n: int, mesh) -> ttnn.MemoryConfig:
    dram_grid = mesh.dram_grid_size()
    cores = dram_grid.x
    padded = math.ceil(n / (TILE * cores)) * (TILE * cores)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores - 1, 0))])
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(grid, (k, padded // cores), ttnn.ShardOrientation.ROW_MAJOR),
    )


def l1_width_sharded(rows: int, width: int, cores: int, grid: ttnn.CoreCoord) -> ttnn.MemoryConfig:
    per_core = math.ceil(width / (TILE * cores)) * TILE
    ranges = []
    full = cores // grid.x
    if full:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, full - 1)))
    rest = cores % grid.x
    if rest:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rest - 1, full)))
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(ttnn.CoreRangeSet(ranges), (rows, per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().to(torch.float64), b.flatten().to(torch.float64)
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def timed(fn, mesh, reps=30):
    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - t0) / reps * 1e3


def case(label, fn, ref, mesh):
    try:
        ms = timed(fn, mesh)
        out = fn()
        got = ttnn.to_torch(out)
        print(
            f"OK      {label:52s} {ms:8.4f} ms  pcc={pcc(got[..., : ref.shape[-1]], ref):.6f}  "
            f"out_shape={tuple(got.shape)}"
        )
        ttnn.deallocate(out)
    except Exception as exc:  # noqa: BLE001
        msg = " | ".join(line.strip() for line in str(exc).strip().splitlines() if line.strip())
        print(f"BLOCKED {label:52s} {msg[:400]}")


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    ttnn.SetDefaultDevice(mesh)
    grid = mesh.compute_with_storage_grid_size()
    try:
        torch.manual_seed(0)
        w_t = torch.randn(1, 1, K, N) / math.sqrt(K)
        ck = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        w_inter = ttnn.from_torch(
            w_t, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        w_shard = ttnn.from_torch(
            w_t,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=dram_width_sharded(K, N, mesh),
        )

        cores, in0_bw = 13, 16
        x_d = torch.randn(1, 1, DECODE_ROWS, K)
        ref_d = (x_d.to(torch.float64) @ w_t.to(torch.float64)).float()
        x_dec = ttnn.from_torch(
            x_d, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        x_dec_sh = ttnn.to_memory_config(x_dec, l1_width_sharded(DECODE_ROWS, K, cores, grid))
        cfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=in0_bw, per_core_M=1, per_core_N=math.ceil(N / (TILE * cores)), fused_activation=None
        )
        out_sh = l1_width_sharded(DECODE_ROWS, N, cores, grid)

        case(
            "1 decode dramsharded-cfg + INTERLEAVED weight",
            lambda: ttnn.linear(
                x_dec_sh,
                w_inter,
                dtype=ttnn.bfloat16,
                memory_config=out_sh,
                program_config=cfg,
                compute_kernel_config=ck,
            ),
            ref_d,
            mesh,
        )
        case(
            "2 decode dramsharded-cfg + WIDTH_SHARDED weight",
            lambda: ttnn.linear(
                x_dec_sh,
                w_shard,
                dtype=ttnn.bfloat16,
                memory_config=out_sh,
                program_config=cfg,
                compute_kernel_config=ck,
            ),
            ref_d,
            mesh,
        )
        case(
            "2b decode dramsharded-cfg + L1_INTERLEAVED output",
            lambda: ttnn.linear(
                x_dec_sh,
                w_shard,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                program_config=cfg,
                compute_kernel_config=ck,
            ),
            ref_d,
            mesh,
        )
        case(
            "2c decode dramsharded-cfg + DRAM_INTERLEAVED input",
            lambda: ttnn.linear(
                x_dec,
                w_shard,
                dtype=ttnn.bfloat16,
                memory_config=out_sh,
                program_config=cfg,
                compute_kernel_config=ck,
            ),
            ref_d,
            mesh,
        )

        x_p = torch.randn(1, 1, PREFILL_ROWS, K)
        ref_p = (x_p.to(torch.float64) @ w_t.to(torch.float64)).float()
        x_pre = ttnn.from_torch(
            x_p, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        case(
            "3 prefill minimal_matmul + WIDTH_SHARDED weight",
            lambda: ttnn.experimental.minimal_matmul(
                x_pre, w_shard, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ck
            ),
            ref_p,
            mesh,
        )
        case(
            "3b prefill minimal_matmul + INTERLEAVED weight",
            lambda: ttnn.experimental.minimal_matmul(
                x_pre, w_inter, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ck
            ),
            ref_p,
            mesh,
        )
        case(
            "4 prefill ttnn.linear + WIDTH_SHARDED weight",
            lambda: ttnn.linear(
                x_pre, w_shard, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ck
            ),
            ref_p,
            mesh,
        )
        ttnn.deallocate(x_pre)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
