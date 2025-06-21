# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "shapes",
    [
        ([1, 1, 512, 4096], [1, 1, 512, 256], 2, 4),
        ([1, 1, 512, 4096], [1, 1, 512, 256], 4, 2),
        ([1, 1, 32, 64], [1, 1, 32, 32], 0, 0),
        ([1, 1, 64, 128], [1, 1, 64, 32], 0, 0),
    ],
)
@pytest.mark.parametrize("keepdim", [True, False])
def test_nd_shard(device, shapes, keepdim):
    dim = -2
    input_shape, shard_shape, end_x, end_y = shapes
    torch_input_tensor = torch.rand(input_shape)
    torch_output_tensor = torch.sum(torch_input_tensor, dim, keepdim)

    memory_config = ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.L1,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(end_x, end_y))}),
        ),
    )
    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=ttnn.float32, device=device, layout=ttnn.TILE_LAYOUT, memory_config=memory_config
    )
    op_output_tensor = ttnn.sum(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_torch(op_output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.999)
