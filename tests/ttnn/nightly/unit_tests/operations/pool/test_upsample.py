# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

import torch
import torch.nn as nn
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, num_channels, height, width, scale_h, scale_w",
    (
        (1, 1280, 8, 8, 2, 2),
        (1, 256, 16, 16, 8, 8),
        (1, 256, 32, 32, 4, 4),
        (1, 256, 64, 64, 2, 2),
        (1, 256, 64, 64, 3, 3),
        (1, 1024, 8, 8, 2, 2),
        (1, 256, 28, 28, 2, 2),
        (1, 512, 14, 14, 2, 2),
    ),
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("math_approx_mode", [True, False])
def test_bilinear_interleaved_memory(
    device,
    batch_size,
    num_channels,
    height,
    width,
    scale_h,
    scale_w,
    math_fidelity,
    math_approx_mode,
):
    # Performs bilinear upsampling on interleaved inputs
    # Automatically height shards the input tensor

    torch.manual_seed(0)

    input_shape = [batch_size, num_channels, height, width]

    mode = "bilinear"

    torch_input = torch.rand(input_shape, dtype=torch.bfloat16)
    tt_input = torch_input.permute(0, 2, 3, 1)
    input_tensor = ttnn.from_torch(tt_input, device=device)
    scale_factor = (scale_h, scale_w)
    torch_upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
    torch_result = torch_upsample(torch_input)

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=math_fidelity,
        math_approx_mode=math_approx_mode,
        fp32_dest_acc_en=False,
    )

    output_tensor = ttnn.upsample(input_tensor, scale_factor, mode=mode, compute_kernel_config=compute_kernel_config)
    output_tensor = ttnn.to_torch(output_tensor)

    torch_result = torch_result.permute(0, 2, 3, 1)
    pcc_passed, pcc_message = assert_with_pcc(torch_result, output_tensor, pcc=0.999)
    logger.info(pcc_message)
    allclose = torch.allclose(output_tensor, torch_result, atol=1e-1, rtol=1e-1)
    assert allclose
