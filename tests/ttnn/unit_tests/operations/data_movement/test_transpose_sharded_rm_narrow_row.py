# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from tt_py_test_utils_common.utility_functions import run_for_blackhole


def _height_sharded_rm_input(device, h, w, c=4):
    """[1, c, h, w] bf16 ROW_MAJOR tensor, height-sharded on one core so every H x W slab is in the shard.

    This selects the sharded row-major WH transpose factory, which packs the output rows (h datums wide when
    h < 32) through the narrow-row untilize. The native sharded factory is only chosen when the row-major
    shard holds a tile multiple (1024) of elements, hence c slabs per shard: (c*h) * w and the output's
    (c*w) * h are multiples of 1024 for every h used here.
    """
    torch.manual_seed(2005)
    torch_input = torch.rand((1, c, h, w), dtype=torch.bfloat16)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    shard_spec = ttnn.ShardSpec(shard_grid, (c * h, w), ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=sharded_mem_config,
    )
    return torch_input, tt_input


@pytest.mark.parametrize("h", [8, 16])
@pytest.mark.parametrize("w", [64])
def test_transpose_hw_sharded_rm_narrow_row_aligned(device, h, w):
    """Narrow output rows whose bf16 byte width is a multiple of 16 transpose exactly."""
    torch_input, tt_input = _height_sharded_rm_input(device, h, w)
    tt_output = ttnn.transpose(tt_input, 2, 3)
    tt_output = ttnn.to_torch(ttnn.from_device(ttnn.to_memory_config(tt_output, ttnn.L1_MEMORY_CONFIG)))
    assert torch.equal(torch_input.transpose(2, 3), tt_output)
    assert_with_pcc(torch_input.transpose(2, 3), tt_output, 0.9999)


@run_for_blackhole()
@pytest.mark.parametrize("h", [4, 20])
@pytest.mark.parametrize("w", [64])
def test_transpose_hw_sharded_rm_narrow_row_rejects_unaligned_row_bytes(device, expect_error, h, w):
    """An 8- or 40-byte output row cannot be packed at its pitch on Blackhole (16-byte L1 advance): reject it."""
    _, tt_input = _height_sharded_rm_input(device, h, w)
    with expect_error(RuntimeError, "the row byte width must be a multiple of 16 bytes"):
        ttnn.transpose(tt_input, 2, 3)
