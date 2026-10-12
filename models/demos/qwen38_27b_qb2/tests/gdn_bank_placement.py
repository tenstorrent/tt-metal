# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bank-pair-aware work assignment without changing state storage or arithmetic."""

from functools import lru_cache

from models.demos.qwen38_27b_qb2.tt.gdn_step.op import work_items

MODES = ("native", "bank_read", "bank_write", "bank_balanced")


def bank_pair(work):
    # State tiles are head*16 + key_row*4 + value_partition. The pinned
    # eight-bank interleave maps every partition to the same pair of banks.
    partition = work % 4
    return partition, partition + 4


def cost(core, partition, read, write, mode):
    anchors = read if mode == "bank_read" else write if mode == "bank_write" else read + write
    banks = (partition, partition + 4)
    return sum(abs(core[0] - anchors[i][0]) + abs(core[1] - anchors[i][1]) for i in banks) + (
        sum(abs(core[0] - anchors[i + 8][0]) + abs(core[1] - anchors[i + 8][1]) for i in banks)
        if mode == "bank_balanced"
        else 0
    )


@lru_cache(maxsize=16)
def placement(read, write, mode):
    if mode not in MODES:
        raise ValueError("Unknown bank placement mode")
    for anchors in (read, write):
        if len(anchors) != 8 or any(
            len(p) != 2 or any(type(v) is not int for v in p) or not (0 <= p[0] < 12 and 0 <= p[1] < 10)
            for p in anchors
        ):
            raise ValueError("Require eight logical bank anchors within the 12x10 worker grid")
    cores = [(x, y) for y in range(10) for x in range(12)]
    if mode == "native":
        return tuple(cores)
    costs = [[cost(core, partition, read, write, mode) for partition in range(4)] for core in cores]
    # Preserve 30 workers per value partition. Start with the strongest
    # preferences, then improve by pair exchanges that retain those capacities.
    priority = sorted(range(120), key=lambda i: (-(sorted(costs[i])[1] - min(costs[i])), i))
    remaining, owners = [30] * 4, [-1] * 120
    for i in priority:
        owner = min((p for p in range(4) if remaining[p]), key=lambda p: (costs[i][p], p))
        owners[i] = owner
        remaining[owner] -= 1
    for _ in range(120):
        swaps = [
            (costs[i][owners[j]] + costs[j][owners[i]] - costs[i][owners[i]] - costs[j][owners[j]], i, j)
            for i in range(120)
            for j in range(i + 1, 120)
            if owners[i] != owners[j]
        ]
        delta, i, j = min(swaps)
        if delta >= 0:
            break
        owners[i], owners[j] = owners[j], owners[i]
    # Earlier work indices carry the extra item in partial final waves.
    # Place those within each partition nearest the selected bank anchors.
    groups = [sorted((i for i in range(120) if owners[i] == p), key=lambda i: (costs[i][p], i)) for p in range(4)]
    assert all(len(group) == 30 for group in groups)
    result = tuple(cores[groups[i % 4][i // 4]] for i in range(120))
    assert len(set(result)) == 120
    return result


def assignments(heads, grid_x, grid_y, value_splits, *, read, write, mode):
    if (grid_x, grid_y, value_splits) != (12, 10, 4) or heads not in (192, 384):
        raise ValueError("Placement experiment requires B16/B32 with four value partitions on 12x10 workers")
    order = placement(read, write, mode)
    original = work_items(heads, grid_x, grid_y, value_splits)
    return [(*order[first], first, stride, count) for _, _, first, stride, count in original]
