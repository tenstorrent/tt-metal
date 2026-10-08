# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Equal-core-count Blackhole placement candidates; KV remains interleaved."""

LAYOUTS = ("native", "row_major_64", "bank_proximity_64", "row_major_80", "outer_columns_80")


def placement(name, batch):
    if name not in LAYOUTS or batch not in (4, 8, 16, 32):
        raise ValueError("Unsupported attention placement or batch")
    full = [(x, y) for y in range(10) for x in range(11)]
    if name == "native":
        points, grid, cap = full, (11, 10), 16
    elif name == "bank_proximity_64":
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
        points = [(x, y) for x, y in full if x < 4 or x >= 7]
        grid, cap = (8, 10), 32
    else:
        count = 64 if name == "row_major_64" else 80
        points, grid, cap = full[:count], (8, count // 8), 32
    count = len(points)
    cores_per_user = min(count, cap * batch) // batch
    return dict(
        name=name,
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
