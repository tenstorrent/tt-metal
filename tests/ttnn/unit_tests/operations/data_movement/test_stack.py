# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "input_shapes, dim, layout",
    [
        ([(1, 1, 253), (1, 1, 253), (1, 1, 253)], 2, ttnn.ROW_MAJOR_LAYOUT),
        ([(1, 253), (1, 253), (1, 253)], 1, ttnn.ROW_MAJOR_LAYOUT),
        ([(57, 83), (57, 83), (57, 83)], 0, ttnn.TILE_LAYOUT),
        ([(8732,), (8732,), (8732,)], 0, ttnn.ROW_MAJOR_LAYOUT),
        ([(123, 259), (123, 259), (123, 259)], -1, ttnn.TILE_LAYOUT),
        ([(1, 1, 253), (1, 1, 253), (1, 1, 253)], -3, ttnn.ROW_MAJOR_LAYOUT),
        ([(120, 1, 253, 3), (120, 1, 253, 3), (120, 1, 253, 3), (120, 1, 253, 3)], 3, ttnn.ROW_MAJOR_LAYOUT),
        ([(57, 83), (57, 83), (57, 83), (57, 83), (57, 83), (57, 83), (57, 83)], -1, ttnn.TILE_LAYOUT),
        ([(24, 50, 83), (24, 50, 83), (24, 50, 83), (24, 50, 83), (24, 50, 83), (24, 50, 83)], 2, ttnn.TILE_LAYOUT),
    ],
)
def test_stack(device, input_shapes, dim, layout):
    torch_tensors = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]

    ttnn_tensors = [ttnn.from_torch(tensor) for tensor in torch_tensors]
    ttnn_tensors = [ttnn.to_device(tensor, device, memory_config=ttnn.L1_MEMORY_CONFIG) for tensor in ttnn_tensors]
    ttnn_tensors = [ttnn.to_layout(tensor, layout) for tensor in ttnn_tensors]
    torch_result = torch.stack(torch_tensors, dim=dim)
    ttnn_result = ttnn.stack(ttnn_tensors, dim=dim)

    ttnn_result_torch = ttnn.to_torch(ttnn_result)
    assert_with_pcc(torch_result, ttnn_result_torch)

    assert torch_result.shape == ttnn_result_torch.shape
