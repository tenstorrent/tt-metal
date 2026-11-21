# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_equal


def make_height_sharded_tensor(device):
    torch_input = torch.randn((1, 1, 64, 64), dtype=torch.bfloat16)
    memory_config = ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.L1,
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        shard_spec=ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
            (32, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    return torch_input, tt_input


@pytest.mark.parametrize("core_id", [0, 1])
def test_extract_shard_valid_core_id(device, core_id):
    torch_input, tt_input = make_height_sharded_tensor(device)

    shard = tt_input.extract_shard(core_id)

    expected = torch_input[:, :, core_id * 32 : (core_id + 1) * 32, :]
    assert_equal(expected, ttnn.to_torch(shard))


@pytest.mark.parametrize("core_id", [2, 3, 100, 2**31, 2**32 - 1])
def test_extract_shard_core_id_out_of_range(device, core_id, expect_error):
    _, tt_input = make_height_sharded_tensor(device)

    with expect_error(RuntimeError, "extract_shard: core_id .* is out of range"):
        tt_input.extract_shard(core_id)
