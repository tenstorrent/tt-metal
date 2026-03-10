# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.rvc.tt_impl.conv1d import Conv1d
from tests.ttnn.utils_for_testing import assert_with_pcc


def create_conv1d_input_tensor(batch_size: int, input_length: int, in_channels: int, device):
    shape = (batch_size, in_channels, input_length)
    ncl = torch.randn(shape, dtype=torch.bfloat16).float()

    nlc = torch.permute(ncl, (0, 2, 1)).reshape(batch_size, input_length, in_channels)
    nlc = ttnn.from_torch(
        nlc,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        # memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    return ncl, nlc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    ("kernel_size", "padding", "activation"),
    [
        (7, 0, None),
        (7, 3, None),
        (7, "same", None),
        (8, 0, None),
        (8, 4, None),
        (8, "same", None),
        (7, "same", "relu"),
    ],
)
def test_conv1d(device, kernel_size, padding, activation):
    torch.manual_seed(0)

    batch_size = 1
    in_channels = 128
    out_channels = 128
    input_length = 732
    stride = 1
    dilation = 1
    groups = 16

    torch_conv = torch.nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=True,
    ).eval()

    torch_input, tt_input = create_conv1d_input_tensor(
        batch_size=batch_size, input_length=input_length, in_channels=in_channels, device=device
    )
    torch_output = torch_conv(torch_input)
    if activation == "relu":
        torch_output = torch.relu(torch_output)

    tt_conv = Conv1d(
        device,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        dtype=ttnn.bfloat16,
        activation=activation,
    )
    tt_conv.load_parameters(
        {
            "conv.weight": torch_conv.weight.detach().cpu(),
            "conv.bias": torch_conv.bias.detach().cpu(),
        },
        key="conv",
    )

    tt_output = tt_conv(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32)
    tt_output_torch = tt_output_torch.reshape(batch_size, -1, out_channels)
    tt_output_torch = tt_output_torch.permute(0, 2, 1)
    assert_with_pcc(torch_output, tt_output_torch, pcc=0.99)
