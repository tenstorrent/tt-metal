# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

from loguru import logger

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "T, B, C, cores_y, cores_x",
    [
        (16384, 4, 8192, 8, 8),  # base case
    ],
)
def test_ema(device, T, B, C, cores_y, cores_x):
    torch.manual_seed(0)

    grid_size = ttnn.CoreGrid(y=cores_y, x=cores_x)
    # torch input tensor
    torch_input_tensor = torch.rand(T * B * C, dtype=torch.bfloat16).reshape(B, C, T // 1024, 1024)
    # input tensor
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output_tensor = ttnn.from_torch(
        torch_input_tensor * 0,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    num_itr = 1  # second iteration to help catch potential runtime args issue.
    for _ in range(num_itr):
        tt_output_tensor = ttnn.ema(ttnn_input_tensor, alpha=1e-2, core_grid=grid_size, out=ttnn_output_tensor)
        ttnn.synchronize_device(device)
    dev_output_tensor = ttnn.from_device(tt_output_tensor)
    torch_output_tensor = ttnn.to_torch(dev_output_tensor)
    assert torch.allclose(2 * torch_input_tensor, torch_output_tensor, atol=1e-3)
