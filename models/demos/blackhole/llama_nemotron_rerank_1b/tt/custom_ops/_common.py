# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the model-local generic_op kernels (adapted from pplx_embed_4b/tt/custom_ops)."""

import ttnn

TILE_BYTES = {
    ttnn.bfloat16: 2048,
    ttnn.bfloat8_b: 1088,
    ttnn.bfloat4_b: 576,
    ttnn.float32: 4096,
}


def split_work_to_cores(num_units: int, grid_x: int, grid_y: int):
    """Spread ``num_units`` over the fewest cores that keep the same per-core maximum.

    The op lasts as long as its busiest core, and every extra core adds launch cost.
    Returns ``(num_cores, [(cx, cy, n_units), ...])``.
    """
    if num_units <= 0:
        return 0, []
    num_cores = min(grid_x * grid_y, num_units)
    per_core_max = -(-num_units // num_cores)
    num_cores = -(-num_units // per_core_max)
    base, extra = divmod(num_units, num_cores)
    cores = []
    for i in range(num_cores):
        cx, cy = divmod(i, grid_y)
        cores.append((cx, cy, base + (1 if i < extra else 0)))
    return num_cores, cores


def core_ranges(per_core) -> "ttnn.CoreRangeSet":
    """Cover the (cx, cy, n) work list with as few rectangles as possible, so each kernel
    binary is multicast once per launch instead of unicast to every core."""
    cols = {}
    for cx, cy, _ in per_core:
        cols.setdefault(cx, []).append(cy)
    strips = []
    for cx in sorted(cols):
        ys = sorted(cols[cx])
        y0 = prev = ys[0]
        for y in ys[1:]:
            if y != prev + 1:
                strips.append((cx, y0, prev))
                y0 = y
            prev = y
        strips.append((cx, y0, prev))
    ranges = []
    for cx, y0, y1 in strips:
        if ranges and ranges[-1][2] == y0 and ranges[-1][3] == y1 and ranges[-1][1] == cx - 1:
            ranges[-1][1] = cx
        else:
            ranges.append([cx, cx, y0, y1])
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for x0, x1, y0, y1 in ranges]
    )
