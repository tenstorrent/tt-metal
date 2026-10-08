# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Equal-core-count Blackhole placement candidates; KV remains interleaved."""

LAYOUTS = ("native", "row_major_64", "flash_mla_64", "row_major_80", "outer_columns_80")
LONG_CONTEXT_LAYOUTS = ("full_grid_sharded", "row_major_96", "outer_columns_96")


def placement(name, batch, worker_grid):
    if name not in (*LAYOUTS, *LONG_CONTEXT_LAYOUTS) or batch not in (4, 8, 16, 32):
        raise ValueError("Unsupported attention placement or batch")
    if tuple(worker_grid) not in ((11, 10), (12, 10)):
        raise ValueError(f"Unsupported placement worker grid: {worker_grid}")
    width, height = worker_grid
    full = [(x, y) for y in range(height) for x in range(width)]
    if name in LONG_CONTEXT_LAYOUTS and tuple(worker_grid) != (12, 10):
        raise ValueError("Long-context placement controls require the measured 12x10 worker grid")
    if name == "native":
        points, grid, cap = full, (width, height), 16
    elif name == "full_grid_sharded":
        # Same active-work count as native; isolate reducer placement and the
        # required output-sharding/conversion boundary before changing count.
        points, grid, cap = full, (width, height), 16
    elif name == "row_major_96":
        points, grid, cap = full[:96], (12, 8), 32
    elif name == "outer_columns_96":
        # Prefer both outside columns equally, then break ties by row. This
        # still reads interleaved KV: physical proximity is only a hypothesis.
        selected = sorted(full, key=lambda p: (min(p[0], width - 1 - p[0]), p[1], p[0]))[:96]
        points = sorted(selected, key=lambda p: (p[1], p[0]))
        grid, cap = (12, 8), 32
    elif name == "flash_mla_64":
        # DeepSeek FlashMLA NOC0's eight 8-core groups. This experiment reuses
        # their locations, not MLA's bank-sharded KV or custom group ordering.
        blocks = (
            (range(0, 4), (1, 2)),
            (range(0, 4), (3, 4)),
            (range(0, 4), (7, 8)),
            (range(0, 4), (9, 0)),
            (range(7, 11), (1, 2)),
            (range(7, 11), (4, 5)),
            (range(7, 11), (6, 7)),
            (range(7, 11), (9, 0)),
        )
        points = sorted([(x, y) for xs, ys in blocks for y in ys for x in xs], key=lambda p: (p[1], p[0]))
        grid, cap = (8, 8), 32
    elif name == "outer_columns_80":
        points = [(x, y) for x, y in full if x < 4 or x >= width - 4]
        grid, cap = (8, 10), 32
    else:
        count = 64 if name == "row_major_64" else 80
        points, grid, cap = full[:count], (8, count // 8), 32
    count = len(points)
    cores_per_user = min(count, cap * batch) // batch
    return dict(
        name=name,
        worker_grid=list(worker_grid),
        grid=list(grid),
        logical_cores=[list(p) for p in points],
        max_cores_per_head_batch=cap,
        active_cores=cores_per_user * batch,
        cores_per_user=cores_per_user,
        explicit_subgrid=name != "native",
        output_layout="dram_interleaved" if name == "native" else "height_sharded_then_dram_interleaved",
        kv_layout="unchanged_dram_interleaved",
    )


def useful_kv_bytes(positions):
    # Both K and V, one local KV head, 256 elements, BFP8 including exponents.
    return sum(position + 1 for position in positions) * 2 * 256 * 1088 // 1024
