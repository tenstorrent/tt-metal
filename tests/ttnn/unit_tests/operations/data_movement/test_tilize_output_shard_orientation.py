# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""tilize must not ignore the orientation of a requested sharded output memory config.

For a sharded input with a sharded output of the same memory layout, tilize uses a zero-copy
factory that aliases the output buffer to the input shard and replaces the requested output shard
spec with the input one. A request for the other orientation used to succeed with the input's
orientation, so the caller could not see that the request was dropped.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal

HEIGHT_SHARDED = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
ROW_MAJOR = ttnn.ShardOrientation.ROW_MAJOR
COL_MAJOR = ttnn.ShardOrientation.COL_MAJOR


def _sharded_config(orientation):
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))])
    return ttnn.MemoryConfig(HEIGHT_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(grid, [64, 64], orientation))


def _row_major_input(device, orientation):
    torch_input = torch.randn(1, 1, 256, 64, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=_sharded_config(orientation),
    )
    return torch_input, tt_input


@pytest.mark.parametrize(
    "input_orientation, requested_orientation",
    [(ROW_MAJOR, COL_MAJOR), (COL_MAJOR, ROW_MAJOR)],
    ids=["row_major_to_col_major", "col_major_to_row_major"],
)
def test_tilize_rejects_flipped_output_shard_orientation(
    device, input_orientation, requested_orientation, expect_error
):
    _, tt_input = _row_major_input(device, input_orientation)
    # The zero-copy factory keeps the input shard spec, so the other orientation is not honoured.
    with expect_error(RuntimeError, "requested output shard orientation"):
        ttnn.tilize(tt_input, memory_config=_sharded_config(requested_orientation))


@pytest.mark.parametrize("orientation", [ROW_MAJOR, COL_MAJOR], ids=["row_major", "col_major"])
def test_tilize_accepts_matching_output_shard_orientation(device, orientation):
    # Control that passes with and without the fix: the request matches the input shard spec.
    torch_input, tt_input = _row_major_input(device, orientation)
    output = ttnn.tilize(tt_input, memory_config=_sharded_config(orientation))
    assert output.memory_config().shard_spec.orientation == orientation
    assert_equal(ttnn.to_torch(output).float(), torch_input.float())
