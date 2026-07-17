# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Core-grid helpers for dynamic-batch decode attention heads."""

from __future__ import annotations

import ttnn


def decode_head_core_grid(mesh_device, batch_size: int) -> ttnn.CoreRangeSet:
    """Return one height-shard core per active decode user.

    Preserve the conventional rectangular grid whenever the active batch can
    be factored within the worker grid. Prime and otherwise non-factorable
    batches use TTNN's exact row-wise multi-range representation instead.
    """
    compute_grid = mesh_device.compute_with_storage_grid_size()
    grid_capacity = compute_grid.x * compute_grid.y
    if not 1 <= batch_size <= grid_capacity:
        raise ValueError(f"decode batch size {batch_size} is outside the worker-grid capacity 1..{grid_capacity}")

    grid_x = min(batch_size, compute_grid.x)
    if batch_size % grid_x != 0:
        divisors = [
            candidate
            for candidate in range(grid_x, 0, -1)
            if batch_size % candidate == 0 and batch_size // candidate <= compute_grid.y
        ]
        if not divisors:
            return ttnn.num_cores_to_corerangeset(batch_size, compute_grid, row_wise=True)
        grid_x = divisors[0]

    grid_y = batch_size // grid_x
    return ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(grid_x - 1, grid_y - 1),
            )
        }
    )


def decode_head_sub_core_grids(mesh_device, core_grid: ttnn.CoreRangeSet) -> ttnn.CoreRangeSet | None:
    """Select the concat-heads subcore program for a multi-range input grid."""
    ranges = core_grid.ranges()
    if len(ranges) == 1 and ranges[0].start == ttnn.CoreCoord(0, 0):
        return None

    compute_grid = mesh_device.compute_with_storage_grid_size()
    return ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1),
            )
        }
    )
