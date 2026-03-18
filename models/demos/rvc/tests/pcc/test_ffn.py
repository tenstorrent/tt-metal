# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.rvc.torch_impl.synthesizer.attentions import FFN as TorchFFN
from models.demos.rvc.tt_impl.synthesizer.attentions import FFN as TTFFN
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_ffn(device):
    torch.manual_seed(0)

    batch_size = 1
    in_channels = 16
    out_channels = 16
    filter_channels = 32
    input_length = 64
    kernel_size = 3

    torch_ffn = TorchFFN(
        in_channels=in_channels,
        out_channels=out_channels,
        filter_channels=filter_channels,
        kernel_size=kernel_size,
    ).eval()

    torch_input = torch.randn(batch_size, in_channels, input_length, dtype=torch.float32)
    torch_output = torch_ffn(torch_input)

    tt_ffn = TTFFN(
        device=device,
        in_channels=in_channels,
        out_channels=out_channels,
        filter_channels=filter_channels,
        kernel_size=kernel_size,
    )

    parameters = {
        "encoder.ffn.conv_1.weight": torch_ffn.conv_1.weight,
        "encoder.ffn.conv_1.bias": torch_ffn.conv_1.bias,
        "encoder.ffn.conv_2.weight": torch_ffn.conv_2.weight,
        "encoder.ffn.conv_2.bias": torch_ffn.conv_2.bias,
    }
    tt_ffn.load_state_dict(parameters=parameters, module_prefix="encoder.ffn.")

    tt_input = ttnn.from_torch(
        torch_input.to(torch.bfloat16).permute(0, 2, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    tt_output = tt_ffn(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32)
    tt_output_torch = tt_output_torch.reshape(batch_size, -1, out_channels)
    tt_output_torch = tt_output_torch.permute(0, 2, 1)

    assert_with_pcc(torch_output, tt_output_torch, pcc=0.99)
