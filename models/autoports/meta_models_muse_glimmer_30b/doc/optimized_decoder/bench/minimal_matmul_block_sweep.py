# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``MinimalMatmulConfig`` sweep for the width-sharded BFP4/BFP8 prefill weights.

Two things changed under the optimized weight layout and invalidate the fused
stage's block table:

* the weight is BFP4/BFP8 instead of BF16, so the same block sizes need a third
  to a half of the L1 circular-buffer budget and larger blocks may now be legal;
* the weight is DRAM **width-sharded** instead of interleaved, and ``ttnn.linear``
  refuses that layout (``matmul_device_operation.cpp:1233``), so
  ``minimal_matmul`` now has to serve the short-prefill row counts the fused
  stage handed to ``ttnn.linear`` -- where the op's own default blocking is weak
  (79 GB/s at 32-512 rows against ``ttnn.linear``'s 350).

So this sweeps the block neighbourhood at *every* row count that matters, not
only the 8192-row chunk, and prints the op's own default as the reference row.

    python .../bench/minimal_matmul_block_sweep.py --dtype bfp4
"""

from __future__ import annotations

import argparse
import itertools
import math
import time

import torch

import ttnn

TILE = 32

SHAPES = {
    "wqkv": (6656, 4608),
    "attn_gate": (6656, 4096),
    "o_proj": (4096, 6656),
    "mlp_gate_up": (6656, 19968),
    "mlp_down": (19968, 6656),
}


def dram_width_sharded(k: int, n: int, mesh) -> ttnn.MemoryConfig:
    cores = mesh.dram_grid_size().x
    padded = math.ceil(n / (TILE * cores)) * (TILE * cores)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores - 1, 0))])
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(grid, (k, padded // cores), ttnn.ShardOrientation.ROW_MAJOR),
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
    ap.add_argument("--shapes", default="all")
    ap.add_argument("--rows", default="128,512,1024,2048,4096,8192")
    ap.add_argument("--m-blocks", default="1,2,4,8,16")
    ap.add_argument("--k-blocks", default="4,8,13,16,26")
    ap.add_argument("--n-blocks", default="4,8,16")
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()
    dtype = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[args.dtype]
    names = list(SHAPES) if args.shapes == "all" else args.shapes.split(",")

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
        for name in names:
            k, n = SHAPES[name]
            torch.manual_seed(abs(hash(name)) % 10_000)
            w_t = torch.randn(1, 1, k, n) / math.sqrt(k)
            w = ttnn.from_torch(
                w_t,
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                memory_config=dram_width_sharded(k, n, mesh),
            )
            for rows in [int(r) for r in args.rows.split(",")]:
                x_t = torch.randn(1, 1, rows, k)
                x = ttnn.from_torch(
                    x_t,
                    device=mesh,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

                def run(cfg=None):
                    return ttnn.experimental.minimal_matmul(
                        x,
                        w,
                        dtype=ttnn.bfloat16,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        compute_kernel_config=ck,
                        config=cfg,
                    )

                base = timed(lambda: run(None), mesh, args.reps, args.rounds)
                print(
                    f"RESULT {name:12s} rows={rows:5d} dtype={args.dtype} default          {base:9.4f} ms", flush=True
                )
                m_tiles, k_tiles = rows // TILE, k // TILE
                for mb, kb, nb in itertools.product(
                    [int(v) for v in args.m_blocks.split(",")],
                    [int(v) for v in args.k_blocks.split(",")],
                    [int(v) for v in args.n_blocks.split(",")],
                ):
                    if mb > m_tiles or kb > k_tiles:
                        continue
                    sh, sw = (2, 4) if n >= rows else (4, 2)
                    cfg = ttnn.MinimalMatmulConfig(
                        M_block_size=mb,
                        K_block_size=kb,
                        N_block_size=nb,
                        subblock_h=sh,
                        subblock_w=sw,
                        compute_with_storage_grid_size=grid,
                    )
                    try:
                        ms = timed(lambda: run(cfg), mesh, args.reps, args.rounds)
                        print(
                            f"RESULT {name:12s} rows={rows:5d} dtype={args.dtype} "
                            f"M{mb:<2d}K{kb:<2d}N{nb:<2d}       {ms:9.4f} ms  vs_default={100 * (base - ms) / base:+7.2f} %",
                            flush=True,
                        )
                    except Exception as exc:  # noqa: BLE001
                        msg = " | ".join(l.strip() for l in str(exc).strip().splitlines() if l.strip())
                        print(
                            f"BLOCKED {name} rows={rows} M{mb}K{kb}N{nb}: {msg[:200]}",
                            flush=True,
                        )
                ttnn.deallocate(x)
            ttnn.deallocate(w)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
