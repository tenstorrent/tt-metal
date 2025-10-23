# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn


@pytest.mark.parametrize(
    "T, B, C, cores_y, cores_x",
    [
        (16384, 4, 8192, 0, 0),  # base case
    ],
)
def test_ema(device, T, B, C, cores_y, cores_x):
    torch.manual_seed(0)

    grid_size = ttnn.CoreGrid(y=cores_y, x=cores_x) if cores_y > 0 and cores_x > 0 else None

    # torch input tensor
    torch_input_tensor = torch.ones(T * B * C, dtype=torch.bfloat16).reshape(1, B, C, T)

    # move to the device
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    alpha = 0.25
    num_itr = 1  # second iteration to help catch potential runtime args issue.
    for _ in range(num_itr):
        tt_output_tensor = ttnn.ema(ttnn_input_tensor, alpha=alpha, core_grid=grid_size)
        ttnn.synchronize_device(device)

    # move to the host
    dev_output_tensor = ttnn.from_device(tt_output_tensor)
    torch_output_tensor = ttnn.to_torch(dev_output_tensor)
    logger.info("Finished OP, comparing outputs")

    # Calculate golden EMA output
    golden_output_tensor = torch.empty_like(torch_input_tensor)

    # Compare with golden output
    prev_value = 0 * torch_input_tensor[0, :, :, 0]
    for t in range(T):
        golden_output_tensor[0, :, :, t] = prev_value * alpha + (1 - alpha) * torch_input_tensor[0, :, :, t]
        prev_value = golden_output_tensor[0, :, :, t]

    assert torch.allclose(golden_output_tensor, torch_output_tensor, atol=1e-3)
