# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Is a sharded RMSNorm correct when its program grid is bigger than its shard grid?

At ``block_h = 1`` yes; above it, **no, silently**.

This probe exists because of a chain: the 128-row prefill ``tt-perf-report`` window
shows the four hidden-size ``LayerNormDeviceOperation`` rows costing ~134 us each
on **4 cores** (21 % of the window), because ``ttnn.rms_norm`` on a
DRAM-interleaved input parallelises over tile *rows* and 128 rows is 4 tile rows.
The obvious fix is the width-sharded L1 norm the decode path already uses.  While
measuring that (``short_prefill_layout_probe.py``) some core counts returned
``nan`` PCC, which is what this probe isolates.

The rule it establishes: a width-sharded ``ttnn.rms_norm`` is only correct when the
``LayerNormShardedMultiCoreProgramConfig`` grid rectangle **equals** the tensor's
shard core set, unless ``block_h == 1``.

* ``block_h = 1`` (a decode step: 32 rows): every core count agrees with the
  exact-grid case to ``max|diff| = 0.03182``, which is BF16 rounding against a
  float64 reference.  **The shipped decode norm is therefore correct** even though
  it puts a 16-core shard under an ``11 x 2 = 22``-core program grid, and this
  probe is what proves it rather than assuming it.
* ``block_h = 4`` (128 prefill rows): 16 shards under 22 program cores returns
  **75,155 non-finite elements**; 26 under 33 returns 77,465; 52 under 55 returns
  finite but wrong output (``max|diff| = 1.94``).  No exception is raised in any
  case.

That is also what rules the optimisation out.  A correct sharded prefill norm needs
the core count to divide ``6656 / 32 = 208`` tiles exactly (a norm cannot have a
padded shard: the padding would enter the mean) *and* to be an exact rectangle on
an 11-wide grid.  Those two constraints intersect only at ``{1, 2, 4, 8}``, and at
512 rows even 8 cores is L1 OOM -- so the reachable win is 8 cores at <= 256 rows.
See ``README.md`` "Measured and rejected".

    python .../bench/sharded_norm_grid_probe.py
"""

from __future__ import annotations

import argparse
import math

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.fused_decoder import _norm_subblock_w, norm_compute_kernel_config
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import width_sharded_l1

TILE = 32
HIDDEN = 6656


def rectangle(cores: int, grid) -> tuple[int, int] | None:
    """``(gx, gy)`` with ``gx * gy == cores`` inside ``grid``, widest first.

    A width shard laid out as an exact rectangle can be paired with a
    ``LayerNormShardedMultiCoreProgramConfig`` grid that covers *exactly* the
    shard's cores, which is the only configuration this probe finds correct above
    ``block_h = 1``.  On an 11x10 grid with a 6656-wide (208-tile) tensor the core
    count must divide 208 *and* factor inside the grid, which admits
    ``{1, 2, 4, 8, 16}`` -- 16 as ``8x2``.  13, 26 and 52 divide 208 but have no
    rectangle (13 > 11).
    """
    for gx in range(min(cores, grid.x), 0, -1):
        if cores % gx == 0 and cores // gx <= grid.y:
            return gx, cores // gx
    return None


def rect_memcfg(rows: int, cores: int, grid) -> ttnn.MemoryConfig:
    """``[rows, HIDDEN]`` width-sharded over an exact ``gx x gy`` rectangle."""
    gx, gy = rectangle(cores, grid)
    per_core = HIDDEN // cores
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))]),
            (rows, per_core),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="32,128,256")
    ap.add_argument("--cores", default="1,2,4,8,13,16,26,52")
    ap.add_argument(
        "--rect",
        action="store_true",
        help="lay the shard out as an exact gx x gy rectangle and match the program grid to it",
    )
    args = ap.parse_args()

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    ttnn.SetDefaultDevice(mesh)
    try:
        grid = mesh.compute_with_storage_grid_size()
        ck = norm_compute_kernel_config(mesh.arch())
        torch.manual_seed(7)
        weight_t = (1.0 + torch.randn(HIDDEN) * 0.02).to(torch.bfloat16)
        weight = ttnn.from_torch(
            weight_t.reshape(1, 1, 1, HIDDEN), device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        print(f"device grid {grid.x}x{grid.y}; hidden {HIDDEN} = {HIDDEN // TILE} tiles")
        for rows in [int(r) for r in args.rows.split(",")]:
            x_t = torch.randn(1, 1, rows, HIDDEN)
            ref = (
                x_t.to(torch.float64)
                / torch.sqrt((x_t.to(torch.float64) ** 2).mean(-1, keepdim=True) + 1e-5)
                * weight_t.to(torch.float64)
            ).float()
            x = ttnn.from_torch(
                x_t, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            for cores in [int(c) for c in args.cores.split(",")]:
                if (HIDDEN // TILE) % cores:
                    print(f"NORMCHK rows={rows:4d} cores={cores:3d} skipped: would pad the shard")
                    continue
                rect = rectangle(cores, grid) if args.rect else None
                if args.rect and rect is None:
                    print(f"NORMCHK rows={rows:4d} cores={cores:3d} skipped: no exact rectangle on this grid")
                    continue
                memcfg = rect_memcfg(rows, cores, grid) if rect else width_sharded_l1(rows, HIDDEN, cores, grid)
                gx, gy = rect if rect else (min(cores, grid.x), math.ceil(cores / grid.x))
                block_w = HIDDEN // cores // TILE
                program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=[gx, gy],
                    subblock_w=_norm_subblock_w(block_w),
                    block_h=rows // TILE,
                    block_w=block_w,
                    inplace=False,
                )
                match = "grid==shard" if gx * gy == cores else f"grid={gx}x{gy}={gx * gy}!=shard{cores}"
                try:
                    import time as _time

                    def _run(memcfg=memcfg, program_config=program_config):
                        xs2 = ttnn.interleaved_to_sharded(x, memcfg)
                        ys2 = ttnn.rms_norm(
                            xs2,
                            weight=weight,
                            epsilon=1e-5,
                            memory_config=memcfg,
                            program_config=program_config,
                            compute_kernel_config=ck,
                        )
                        ttnn.deallocate(xs2)
                        y2 = ttnn.sharded_to_interleaved(ys2, ttnn.DRAM_MEMORY_CONFIG)
                        ttnn.deallocate(ys2)
                        return y2

                    for _ in range(3):
                        ttnn.deallocate(_run())
                    ttnn.synchronize_device(mesh)
                    elapsed = float("inf")
                    for _ in range(3):
                        ttnn.synchronize_device(mesh)
                        t0 = _time.perf_counter()
                        for _ in range(16):
                            ttnn.deallocate(_run())
                        ttnn.synchronize_device(mesh)
                        elapsed = min(elapsed, (_time.perf_counter() - t0) / 16 * 1e6)
                    xs = ttnn.interleaved_to_sharded(x, memcfg)
                    ys = ttnn.rms_norm(
                        xs,
                        weight=weight,
                        epsilon=1e-5,
                        memory_config=memcfg,
                        program_config=program_config,
                        compute_kernel_config=ck,
                    )
                    y = ttnn.sharded_to_interleaved(ys, ttnn.DRAM_MEMORY_CONFIG)
                    out = ttnn.to_torch(y).float()
                    nonfinite = int((~torch.isfinite(out)).sum())
                    max_diff = float((out - ref).abs().max())
                    print(
                        f"NORMCHK rows={rows:4d} cores={cores:3d} block_h={rows // TILE:2d} "
                        f"grid={gx}x{gy} {match:24s} nonfinite={nonfinite:7d} max|diff|={max_diff:.5f} "
                        f"{elapsed:8.1f} us"
                    )
                    for tensor in (ys, xs, y):
                        ttnn.deallocate(tensor)
                except Exception as exc:  # noqa: BLE001
                    msg = " | ".join(l.strip() for l in str(exc).strip().splitlines() if l.strip())
                    print(f"NORMCHK rows={rows:4d} cores={cores:3d} {match:24s} BLOCKED: {msg[:200]}")
            ttnn.deallocate(x)
        ttnn.deallocate(weight)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
