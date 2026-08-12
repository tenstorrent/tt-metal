# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Where does the DRAM-sharded matmul stop winning as prefill rows grow?

``prefill_matmul_sweep.py`` shows ``minimal_matmul`` with the width-sharded BFP8
weight losing to the fused decoder's short-prefill branch (``ttnn.linear`` with a
BF16 interleaved weight) below about 512 rows -- and ``ttnn.linear`` cannot take
the width-sharded weight at all, so the fused stage's "never slower at any row
count" property needs a different answer there.

The decode matmul is that answer: at 32 rows the DRAM-sharded matmul is ~3x
faster than the BF16 ``ttnn.linear`` branch.  This probe walks it up the row
counts until the width-sharded L1 activation stops fitting, so the dispatch rule
gets a measured threshold instead of a guess.

Per-role core counts are the winners from ``decode_matmul_geometry_bfp*.log``.

    python .../bench/short_prefill_probe.py
"""

from __future__ import annotations

import argparse
import math
import time

import torch

import ttnn

TILE = 32

#: ``name -> (K, N, cores, in0_block_w)``; cores/in0_block_w are the measured
#: 32-row winners for each role at BFP4 (they are also within 1 % at BFP8).
ROLES = {
    "wqkv": (6656, 4608, 13, 16),
    "attn_gate": (6656, 4096, 13, 16),
    "o_proj": (4096, 6656, 8, 16),
    "mlp_gate_up": (6656, 19968, 13, 8),
    "mlp_down": (19968, 6656, 13, 24),
}

ROWS = (32, 64, 128, 256, 512, 1024)


def dram_width_sharded(k: int, n: int, mesh) -> ttnn.MemoryConfig:
    cores = mesh.dram_grid_size().x
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


def timed(fn, mesh, reps: int, rounds: int) -> float:
    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    best = float("inf")
    for _ in range(rounds):
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(mesh)
        best = min(best, (time.perf_counter() - t0) / reps * 1e3)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bfp4", choices=("bfp8", "bfp4"))
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()
    dtype = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[args.dtype]

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    ttnn.SetDefaultDevice(mesh)
    grid = mesh.compute_with_storage_grid_size()
    try:
        ck = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        for name, (k, n, cores, in0_bw) in ROLES.items():
            torch.manual_seed(abs(hash(name)) % 10_000)
            w_t = torch.randn(1, 1, k, n) / math.sqrt(k)
            w_shard = ttnn.from_torch(
                w_t,
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                memory_config=dram_width_sharded(k, n, mesh),
            )
            for rows in ROWS:
                x_t = torch.randn(1, 1, rows, k)
                x = ttnn.from_torch(
                    x_t,
                    device=mesh,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                mm = timed(
                    lambda: ttnn.experimental.minimal_matmul(
                        x,
                        w_shard,
                        dtype=ttnn.bfloat16,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        compute_kernel_config=ck,
                    ),
                    mesh,
                    args.reps,
                    args.rounds,
                )
                cfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=in0_bw,
                    per_core_M=rows // TILE,
                    per_core_N=math.ceil(n / (TILE * cores)),
                    fused_activation=None,
                )
                try:
                    x_sh = ttnn.to_memory_config(x, l1_width_sharded(rows, k, cores, grid))
                    out_mc = l1_width_sharded(rows, n, cores, grid)
                    ds = timed(
                        lambda: ttnn.linear(
                            x_sh,
                            w_shard,
                            dtype=ttnn.bfloat16,
                            memory_config=out_mc,
                            program_config=cfg,
                            compute_kernel_config=ck,
                        ),
                        mesh,
                        args.reps,
                        args.rounds,
                    )
                    ttnn.deallocate(x_sh)
                    print(
                        f"RESULT {name:12s} rows={rows:5d} dtype={args.dtype} minimal={mm:8.4f} ms  "
                        f"dramshard(c{cores},bw{in0_bw})={ds:8.4f} ms  winner="
                        f"{'dramshard' if ds < mm else 'minimal':9s} ratio={mm / ds:6.3f}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    msg = " | ".join(l.strip() for l in str(exc).strip().splitlines() if l.strip())
                    print(
                        f"RESULT {name:12s} rows={rows:5d} dtype={args.dtype} minimal={mm:8.4f} ms  "
                        f"dramshard=BLOCKED {msg[:220]}",
                        flush=True,
                    )
                ttnn.deallocate(x)
            ttnn.deallocate(w_shard)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
