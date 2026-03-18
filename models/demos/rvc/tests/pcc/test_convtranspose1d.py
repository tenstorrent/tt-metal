# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.rvc.tt_impl.convtranspose1d import ConvTranspose1d
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_convtranspose1d(device):
    torch.manual_seed(0)

    batch_size = 1
    in_channels = 32
    out_channels = 256
    input_length = 248
    kernel_size = 16
    stride = 10
    padding = 1
    output_padding = 0
    dilation = 1
    groups = 1

    torch_deconv = torch.nn.ConvTranspose1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
        bias=True,
    ).eval()

    torch_input = torch.randn(batch_size, in_channels, input_length, dtype=torch.float32)
    torch_output = torch_deconv(torch_input)

    tt_deconv = ConvTranspose1d(
        device=device,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
        dtype=ttnn.bfloat16,
    )

    parameters = {
        "decoder.deconv.weight": torch_deconv.weight,
        "decoder.deconv.bias": torch_deconv.bias,
    }
    tt_deconv.load_state_dict(parameters=parameters, key="deconv", module_prefix="decoder.")

    tt_input = ttnn.from_torch(
        torch_input.to(torch.bfloat16).permute(0, 2, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    tt_output = tt_deconv(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32)
    tt_output_torch = tt_output_torch.reshape(batch_size, -1, out_channels)
    tt_output_torch = tt_output_torch.permute(0, 2, 1)

    assert_with_pcc(torch_output, tt_output_torch, pcc=0.99)
