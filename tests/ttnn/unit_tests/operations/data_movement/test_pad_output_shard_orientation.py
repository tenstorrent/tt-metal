# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""pad must keep the shard orientation of a requested HEIGHT_SHARDED output.

For a HEIGHT_SHARDED input with width padding, pad ignores the requested output memory config and
recomputes a HEIGHT_SHARDED config with the wider shard. It used to hard-code ROW_MAJOR there, so a
COL_MAJOR request produced a ROW_MAJOR tensor (with correct values) and the caller could not see
that the request was dropped.
"""

import pytest
import torch
import torch.nn.functional as F

import ttnn
from tests.ttnn.utils_for_testing import assert_equal

HEIGHT_SHARDED = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
ROW_MAJOR = ttnn.ShardOrientation.ROW_MAJOR
COL_MAJOR = ttnn.ShardOrientation.COL_MAJOR


def _sharded_config(orientation, shard_width):
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))])
    return ttnn.MemoryConfig(HEIGHT_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(grid, [32, shard_width], orientation))


def _input(device, orientation):
    torch_input = torch.randn(1, 1, 128, 64, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_sharded_config(orientation, 64),
    )
    return torch_input, tt_input


def _pad(device, tt_input):
    return ttnn.pad(tt_input, [(0, 0), (0, 0), (0, 0), (0, 64)], 0.0, memory_config=_sharded_config(COL_MAJOR, 128))


@pytest.mark.parametrize("input_orientation", [ROW_MAJOR, COL_MAJOR], ids=["row_major_input", "col_major_input"])
def test_pad_keeps_requested_col_major_output(device, input_orientation):
    torch_input, tt_input = _input(device, input_orientation)
    output = _pad(device, tt_input)
    assert output.memory_config().shard_spec.orientation == COL_MAJOR
    assert_equal(ttnn.to_torch(output).float(), F.pad(torch_input.float(), (0, 64)))


@pytest.mark.parametrize("input_orientation", [ROW_MAJOR, COL_MAJOR], ids=["row_major_input", "col_major_input"])
def test_pad_keeps_requested_row_major_output(device, input_orientation):
    # Control that passes with and without the fix: ROW_MAJOR is what the recomputed config used.
    torch_input, tt_input = _input(device, input_orientation)
    output = ttnn.pad(tt_input, [(0, 0), (0, 0), (0, 0), (0, 64)], 0.0, memory_config=_sharded_config(ROW_MAJOR, 128))
    assert output.memory_config().shard_spec.orientation == ROW_MAJOR
    assert_equal(ttnn.to_torch(output).float(), F.pad(torch_input.float(), (0, 64)))
