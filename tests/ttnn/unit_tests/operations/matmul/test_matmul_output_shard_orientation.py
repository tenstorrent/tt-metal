# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""matmul and linear must not ignore the orientation of a requested sharded output memory config.

The output shard orientation is chosen by the program config (from the input shard orientation or
from transpose_mcast), not by the requested memory config. A request for the other orientation used
to succeed with the computed orientation, so the returned tensor said ROW_MAJOR while the caller had
asked for COL_MAJOR, and the caller could not see that the request was dropped.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

HEIGHT_SHARDED = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
BLOCK_SHARDED = ttnn.TensorMemoryLayout.BLOCK_SHARDED
ROW_MAJOR = ttnn.ShardOrientation.ROW_MAJOR
COL_MAJOR = ttnn.ShardOrientation.COL_MAJOR


def _sharded_config(memory_layout, orientation):
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))])
    return ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, ttnn.ShardSpec(grid, [64, 64], orientation))


def _matmul_inputs(device, memory_layout, shape, weight_shape):
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_weight = torch.randn(weight_shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_sharded_config(memory_layout, ROW_MAJOR),
    )
    tt_weight = ttnn.from_torch(
        torch_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return torch_input, torch_weight, tt_input, tt_weight


@pytest.mark.parametrize(
    "memory_layout, shape, weight_shape",
    [
        (HEIGHT_SHARDED, (1, 1, 256, 64), (1, 1, 64, 64)),
        (BLOCK_SHARDED, (1, 1, 128, 128), (1, 1, 128, 128)),
    ],
    ids=["height_sharded", "block_sharded"],
)
def test_matmul_rejects_flipped_output_shard_orientation(device, memory_layout, shape, weight_shape, expect_error):
    _, _, tt_input, tt_weight = _matmul_inputs(device, memory_layout, shape, weight_shape)
    # The program config produces ROW_MAJOR shards for this input, so COL_MAJOR is not honoured.
    with expect_error(RuntimeError, "requested output shard orientation"):
        ttnn.matmul(tt_input, tt_weight, memory_config=_sharded_config(memory_layout, COL_MAJOR))


def test_linear_rejects_flipped_output_shard_orientation(device, expect_error):
    _, _, tt_input, tt_weight = _matmul_inputs(device, HEIGHT_SHARDED, (1, 1, 256, 64), (1, 1, 64, 64))
    tt_bias = ttnn.from_torch(
        torch.randn(1, 1, 1, 64, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with expect_error(RuntimeError, "requested output shard orientation"):
        ttnn.linear(tt_input, tt_weight, bias=tt_bias, memory_config=_sharded_config(HEIGHT_SHARDED, COL_MAJOR))


@pytest.mark.parametrize(
    "memory_layout, shape, weight_shape",
    [
        (HEIGHT_SHARDED, (1, 1, 256, 64), (1, 1, 64, 64)),
        (BLOCK_SHARDED, (1, 1, 128, 128), (1, 1, 128, 128)),
    ],
    ids=["height_sharded", "block_sharded"],
)
def test_matmul_accepts_matching_output_shard_orientation(device, memory_layout, shape, weight_shape):
    # Control that passes with and without the fix: the request matches what the program config produces.
    torch_input, torch_weight, tt_input, tt_weight = _matmul_inputs(device, memory_layout, shape, weight_shape)
    output = ttnn.matmul(tt_input, tt_weight, memory_config=_sharded_config(memory_layout, ROW_MAJOR))
    assert output.memory_config().shard_spec.orientation == ROW_MAJOR
    assert_with_pcc(torch_input.float() @ torch_weight.float(), ttnn.to_torch(output).float(), 0.99)


def test_matmul_accepts_interleaved_output(device):
    # Control: an interleaved output has no shard orientation to honour.
    torch_input, torch_weight, tt_input, tt_weight = _matmul_inputs(
        device, HEIGHT_SHARDED, (1, 1, 256, 64), (1, 1, 64, 64)
    )
    output = ttnn.matmul(tt_input, tt_weight)
    assert not output.memory_config().is_sharded()
    assert_with_pcc(torch_input.float() @ torch_weight.float(), ttnn.to_torch(output).float(), 0.99)
