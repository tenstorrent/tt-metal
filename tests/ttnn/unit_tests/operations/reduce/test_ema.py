# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_numeric_metrics

TEST_PADDING_VALUE = -42


@pytest.mark.parametrize(
    "T, B, C, cores_y, cores_x, alpha",
    [
        (2048, 2, 4096, 0, 0, 0.99),
        (2048, 2, 4096, 4, 4, 0.99),
        (10, 3, 18, 0, 0, 0.25),
        (10, 3, 18, 0, 0, 0.9),
        (10, 3, 18, 0, 0, 0.99),
    ],
)
def test_ema(device, T, B, C, cores_y, cores_x, alpha):
    torch.manual_seed(0)

    grid_size = ttnn.CoreGrid(y=cores_y, x=cores_x) if cores_y > 0 and cores_x > 0 else None

    # torch input tensor
    torch_input_tensor = torch.rand((1, B, C, T), dtype=torch.bfloat16)

    # move to the device
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_input_tensor = ttnn.fill_implicit_tile_padding(ttnn_input_tensor, TEST_PADDING_VALUE)

    num_itr = 2  # second iteration to help catch potential runtime args issue.
    for _ in range(num_itr):
        tt_output_tensor = ttnn.ema(ttnn_input_tensor, alpha=alpha, core_grid=grid_size)
        ttnn.synchronize_device(device)

    # move to the host
    dev_output_tensor = ttnn.from_device(tt_output_tensor)
    torch_output_tensor = ttnn.to_torch(dev_output_tensor)
    logger.info("Finished OP, comparing outputs")

    # Calculate golden EMA output
    golden_function = ttnn.get_golden_function(ttnn.ema)
    golden_output_tensor = golden_function(torch_input_tensor.float(), alpha)

    # test for equivalance
    assert_numeric_metrics(
        golden_output_tensor,
        torch_output_tensor,
        pcc_threshold=0.999,
        rtol=0.008,
        atol=0.004,
        frobenius_threshold=0.003,
    )


def test_ema_starts_from_first_input(device):
    torch_input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16).repeat(1, 1, 32, 1)
    expected = torch.tensor([1.0, 1.5, 2.25, 3.125], dtype=torch.bfloat16).repeat(1, 1, 32, 1)

    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    torch_output_tensor = ttnn.to_torch(ttnn.ema(ttnn_input_tensor, alpha=0.5))

    assert torch.equal(torch_output_tensor, expected)
