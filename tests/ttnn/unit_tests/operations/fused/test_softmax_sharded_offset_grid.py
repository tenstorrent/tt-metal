# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for sharded softmax on offset core grids.

ttnn.softmax_in_place and ttnn.scale_mask_softmax_in_place use
SoftmaxShardedMultiCoreProgramConfig. The host used to build the program's core
range from the origin and enumerate per-core runtime arguments from the origin,
so on a shard grid that does not begin at (0, 0) it launched work on the wrong
cores and left the real shards untouched -- a silent wrong answer.

These tests run the same tensor over the same number of cores, changing only
where the shard grid sits: (0, 0), (1, 0), and (0, 1). Each shard is compared
against the PyTorch result, so a grid that is not fully covered fails.
"""

import pytest
import torch
import torch.nn.functional as F

import ttnn
from tests.ttnn.utils_for_testing import assert_numeric_metrics

TILE = 32
GRID = (4, 4)  # 16 cores
SHAPE = (1, 16, 256, 256)
SHARD = [256, 256]  # one full-width block per core
BLOCK_H = SHARD[0] // TILE
BLOCK_W = SHARD[1] // TILE
SCALE = 0.5

OFFSETS = [(0, 0), (1, 0), (0, 1)]
ORIENTATIONS = [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR]
MASKED = [False, True]


def _grid(offset):
    x0, y0 = offset
    return ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x0 + GRID[0] - 1, y0 + GRID[1] - 1))}
    )


def _sharded_config(offset, orientation):
    return ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=ttnn.ShardSpec(_grid(offset), SHARD, orientation),
    )


def _program_config():
    return ttnn.SoftmaxShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=GRID,
        subblock_w=1,
        block_h=BLOCK_H,
        block_w=BLOCK_W,
    )


@pytest.mark.parametrize("offset", OFFSETS)
@pytest.mark.parametrize("orientation", ORIENTATIONS)
@pytest.mark.parametrize("with_mask", MASKED)
def test_softmax_in_place_sharded_offset_grid(device, offset, orientation, with_mask):
    torch.manual_seed(0)
    x = torch.randn(SHAPE, dtype=torch.bfloat16).float() * 4

    program_config = _program_config()
    memory_config = _sharded_config(offset, orientation)
    input_tensor = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)

    if with_mask:
        mask = torch.zeros(SHAPE, dtype=torch.bfloat16).float()
        mask[..., -8:] = -10.0  # exercise the mask read path, not just the unmasked one
        golden = F.softmax(x * SCALE + mask, dim=-1, dtype=torch.bfloat16)
        mask_t = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = ttnn.scale_mask_softmax_in_place(
            input_tensor, SCALE, mask_t, program_config=program_config, numeric_stable=True
        )
    else:
        golden = F.softmax(x, dim=-1, dtype=torch.bfloat16)
        out = ttnn.softmax_in_place(input_tensor, program_config=program_config, numeric_stable=True)

    assert_numeric_metrics(
        golden,
        ttnn.to_torch(out),
        pcc_threshold=0.999,
        rtol=0.066,
        atol=0.043,
        frobenius_threshold=0.034,
    )


@pytest.mark.parametrize("offset", OFFSETS)
def test_softmax_in_place_offset_grid_matches_origin_grid(device, offset):
    """The offset grid must give the same numbers as the origin grid. This pins
    the fix directly: the bug left offset shards at their input logits."""
    torch.manual_seed(0)
    x = torch.randn(SHAPE, dtype=torch.bfloat16).float() * 4

    results = []
    for off in [(0, 0), offset]:
        tensor = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=_sharded_config(off, ttnn.ShardOrientation.ROW_MAJOR),
        )
        out = ttnn.softmax_in_place(tensor, program_config=_program_config(), numeric_stable=True)
        results.append(ttnn.to_torch(out))

    assert_numeric_metrics(
        results[0],
        results[1],
        pcc_threshold=0.999,
        rtol=0.066,
        atol=0.043,
        frobenius_threshold=0.034,
    )
