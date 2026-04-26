# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.rvc.tt_impl.conv2d import Conv2d
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize(
    ("kernel_size", "stride", "padding", "activation"),
    [
        ((3, 3), (1, 1), (1, 1), None),
        ((3, 3), (1, 1), "same", None),
        ((1, 1), (1, 1), (0, 0), None),
        ((3, 3), (1, 1), "same", "relu"),
    ],
)
def test_conv2d(device, batch_size, kernel_size, stride, padding, activation):
    torch.manual_seed(0)

    in_channels = 16
    out_channels = 32
    input_height = 32
    input_width = 32
    dilation = (1, 1)
    groups = 1

    torch_conv = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=True,
    ).eval()

    torch_input = torch.randn(batch_size, in_channels, input_height, input_width, dtype=torch.float32)
    torch_output = torch_conv(torch_input)
    if activation == "relu":
        torch_output = torch.relu(torch_output)

    tt_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1).contiguous().to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    tt_conv = Conv2d(
        device=device,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        activation=activation,
        bias=True,
    )
    tt_conv.load_state_dict(
        {
            "conv.weight": torch_conv.weight.detach().cpu(),
            "conv.bias": torch_conv.bias.detach().cpu(),
        },
        key="conv",
    )

    tt_output = tt_conv(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32).permute(0, 3, 1, 2)

    assert tuple(tt_output_torch.shape) == tuple(torch_output.shape)
    assert_with_pcc(torch_output, tt_output_torch, pcc=0.99)
