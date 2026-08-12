# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Probe: decode-shaped ``ttnn.rms_norm``, interleaved (1 core) vs width-sharded.

Measures wall time of ``ITERS`` back-to-back rms_norm calls on a
``[1, 1, 32, 6656]`` tile tensor and PCC against a torch reference, for every
legal width-shard core count.

``6656 / 32 = 208 = 2^4 * 13`` tiles, so the core count must divide 208 for the
shard width to stay tile-aligned: 1, 2, 4, 8, 13, 16, 26, 52, 104, 208.  13 does
not fit any rectangle on an 11x10 grid, but the sharded LayerNorm program
factory accepts a **non-rectangular** ``CoreRangeSet`` as long as the whole
height fits on one core (``M == block_h * TILE_HEIGHT``, always true for a
decode step) and the grid is a shard-order prefix of its bounding box
(``layernorm_device_operation.cpp:185-215``).  Both families are swept here.
"""

from __future__ import annotations

import time

import torch

import ttnn
from models.common.utility_functions import comp_pcc

DIM = 6656
ROWS = 32
EPS = 1e-5
ITERS = 200
#: The wall-clock timing includes host dispatch of 3 ops, so a single round has
#: a few us of jitter.  Each config is measured ``ROUNDS`` times and reported by
#: its minimum, which is the least noisy latency estimator.
ROUNDS = 3
MAX_SUBBLOCK_W = 4

#: Rectangles worth trying, plus every core count that divides DIM/32 and needs
#: a non-rectangular (shard-order prefix) core range set.
RECTANGLES = [(2, 1), (2, 2), (4, 1), (4, 2), (2, 4), (8, 1), (4, 4), (8, 2), (2, 8)]
NON_RECTANGULAR = [13, 26, 52, 104]


def subblock_w(block_w: int) -> int:
    for candidate in range(min(MAX_SUBBLOCK_W, block_w), 0, -1):
        if block_w % candidate == 0:
            return candidate
    return 1


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        grid = mesh.compute_with_storage_grid_size()
        torch.manual_seed(0)
        xt = (torch.randn(1, 1, ROWS, DIM) * 0.5).to(torch.bfloat16)
        wt = (1.0 + torch.randn(DIM) * 0.05).to(torch.bfloat16)
        var = xt.float().pow(2).mean(-1, keepdim=True)
        ref = (xt.float() * torch.rsqrt(var + EPS) * wt.float().reshape(1, 1, 1, DIM)).to(torch.bfloat16)

        w_tile = ttnn.from_torch(
            wt.reshape(1, 1, 1, DIM),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        w_rm = ttnn.from_torch(
            wt.reshape(1, 1, DIM // 32, 32),
            device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x_dram = ttnn.from_torch(
            xt, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ck = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        def timed(label, run):
            try:
                out = run(ret=True)
            except Exception as exc:  # noqa: BLE001
                print(f"NORM {label:44s} FAILED {type(exc).__name__}: {str(exc)[:160]}", flush=True)
                return
            pcc = comp_pcc(ref.float(), out.float(), 0.99)[1]
            run()
            rounds = []
            for _ in range(ROUNDS):
                ttnn.synchronize_device(mesh)
                start = time.perf_counter()
                for _ in range(ITERS):
                    run()
                ttnn.synchronize_device(mesh)
                rounds.append((time.perf_counter() - start) / ITERS * 1e6)
            spread = "/".join(f"{r:.1f}" for r in rounds)
            print(f"NORM {label:44s} min {min(rounds):8.1f} us/call  (rounds {spread})  PCC={pcc}", flush=True)

        def interleaved(ret=False):
            out = ttnn.rms_norm(
                x_dram, weight=w_tile, epsilon=EPS, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ck
            )
            if ret:
                t = ttnn.to_torch(out)
                ttnn.deallocate(out)
                return t
            ttnn.deallocate(out)

        timed("interleaved DRAM (baseline, 1 core)", interleaved)

        def sharded_case(core_range_set, cores, grid_size, label):
            block_w = DIM // cores // 32
            memcfg = ttnn.create_sharded_memory_config(
                shape=(ROWS, DIM // cores),
                core_grid=core_range_set,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            pcfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=list(grid_size),
                subblock_w=subblock_w(block_w),
                block_h=ROWS // 32,
                block_w=block_w,
                inplace=False,
            )

            def run(ret=False):
                xs = ttnn.to_memory_config(x_dram, memcfg)
                out = ttnn.rms_norm(
                    xs, weight=w_rm, epsilon=EPS, program_config=pcfg, memory_config=memcfg, compute_kernel_config=ck
                )
                ttnn.deallocate(xs)
                interleaved_out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(out)
                if ret:
                    t = ttnn.to_torch(interleaved_out)
                    ttnn.deallocate(interleaved_out)
                    return t
                ttnn.deallocate(interleaved_out)

            timed(f"{label} {cores:3d}c bw={block_w:3d} sw={subblock_w(block_w)}", run)

        tiles = DIM // 32
        for gx, gy in RECTANGLES:
            cores = gx * gy
            if tiles % cores:
                continue
            sharded_case(ttnn.CoreGrid(y=gy, x=gx), cores, (gx, gy), f"rect {gx}x{gy}")

        device_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        for cores in NON_RECTANGULAR:
            if tiles % cores or cores > grid.x * grid.y:
                print(
                    f"NORM non-rect {cores:3d}c skipped (needs {cores} cores, grid has {grid.x * grid.y})", flush=True
                )
                continue
            crs = ttnn.num_cores_to_corerangeset_in_subcoregrids(ttnn.CoreCoord(0, 0), cores, device_range, True)
            bbox = crs.bounding_box()
            bbox_size = (bbox.end.x - bbox.start.x + 1, bbox.end.y - bbox.start.y + 1)
            sharded_case(crs, cores, bbox_size, f"prefix bbox{bbox_size[0]}x{bbox_size[1]}")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
