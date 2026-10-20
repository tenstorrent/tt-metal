# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from helpers.llk_params import MathOperation
from helpers.tile_constants import DEFAULT_TILE_R_DIM

VALUE_TILE_COUNT = 2
INDEX_TILE_COUNT = 2
DEST_TILE_COUNT = VALUE_TILE_COUNT + INDEX_TILE_COUNT

TOTAL_DATUMS_TO_COMPARE = VALUE_TILE_COUNT * DEFAULT_TILE_R_DIM

MAX_MERGE_K = 32
MAX_COMPARE_DISTANCE = 32

FIRST_DIRECTIONAL_PHASE = 3

STEP_N_MIN_DISTANCE = 16

FIRST_DIRECTIONAL_REBUILD_K = 16

FULL_REBUILD_MIN_K = 32


def _compare(values, indices, left, right, descending, reverse_operands=False):
    a, b = values[:, left].clone(), values[:, right].clone()
    ia, ib = indices[:, left].clone(), indices[:, right].clone()
    c, d = (b, a) if reverse_operands and not descending else (a, b)
    swap = (c < d) | ((c == d) & torch.signbit(c))
    if not descending and not reverse_operands:
        swap = ~swap
    values[:, left], values[:, right] = torch.where(swap, b, a), torch.where(swap, a, b)
    indices[:, left], indices[:, right] = torch.where(swap, ib, ia), torch.where(
        swap, ia, ib
    )


def _merge_run(values, indices, start, width, descending, phase0=False):
    distance = width // 2
    while distance:
        left = [i for i in range(start, start + width) if not (i - start) & distance]
        right = [i + distance for i in left]
        _compare(
            values,
            indices,
            left,
            right,
            descending,
            phase0 or distance >= STEP_N_MIN_DISTANCE,
        )
        distance //= 2


def topk_golden(call, state, node, operation, config):
    sfpu = node.sfpu
    tiles = [state.dest.get(call.dest + i) for i in range(DEST_TILE_COUNT)]
    values = torch.cat([tile.T for tile in tiles[:VALUE_TILE_COUNT]], dim=1)
    indices = torch.cat(
        [tile.T.to(torch.int64) for tile in tiles[VALUE_TILE_COUNT:]], dim=1
    )
    if not torch.isfinite(values).all():
        raise ValueError("TopK golden supports finite keys only")
    k = sfpu.k
    if sfpu.operation == MathOperation.TopKLocalSort:
        for phase in range(k.bit_length() - 1):
            width = 1 << (phase + 1)
            descending = True if phase < FIRST_DIRECTIONAL_PHASE else sfpu.descending
            for run, start in enumerate(range(0, TOTAL_DATUMS_TO_COMPARE, width)):
                _merge_run(
                    values,
                    indices,
                    start,
                    width,
                    descending != bool(run % 2),
                    phase == 0,
                )
    elif sfpu.operation == MathOperation.TopKMerge:
        width = min(k, MAX_MERGE_K)
        distance = min(width << sfpu.m_iter, MAX_COMPARE_DISTANCE)
        pairs = max(TOTAL_DATUMS_TO_COMPARE >> sfpu.m_iter, 2 * width) // (2 * width)
        for pair in range(pairs):
            left = list(range(2 * distance * pair, 2 * distance * pair + width))
            _compare(
                values,
                indices,
                left,
                [i + distance for i in left],
                sfpu.descending,
                reverse_operands=True,
            )
    else:
        stride = (
            k
            if k >= FULL_REBUILD_MIN_K
            else min(2 * k << sfpu.m_iter, MAX_COMPARE_DISTANCE)
        )
        starts = list(range(0, TOTAL_DATUMS_TO_COMPARE, stride))
        if (
            sfpu.skip_second
            and k < FULL_REBUILD_MIN_K
            and (k == FIRST_DIRECTIONAL_REBUILD_K or sfpu.m_iter == 0)
        ):
            starts = starts[: len(starts) // 2]
        descending = True if k < FIRST_DIRECTIONAL_REBUILD_K else sfpu.descending
        for run, start in enumerate(starts):
            _merge_run(values, indices, start, k, descending != bool(run % 2))
    for i, tile in enumerate(
        (
            *values.split(DEFAULT_TILE_R_DIM, dim=1),
            *indices.split(DEFAULT_TILE_R_DIM, dim=1),
        )
    ):
        state.dest.set(call.dest + i, tile.T.to(tiles[i].dtype))
