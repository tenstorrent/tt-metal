# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.rvc.torch_impl.vc.hubert import ConvolutionModule as TorchConvolutionModule
from models.demos.rvc.tt_impl.vc.hubert import ConvolutionModule as TTConvolutionModule
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_hubert_convolution_module(device):
    torch.manual_seed(0)

    embed_dim = 64
    channels = 64
    depthwise_kernel_size = 7
    activation_fn = "swish"
    bias = False
    b = 2
    t = 48

    torch_module = TorchConvolutionModule(
        embed_dim=embed_dim,
        channels=channels,
        depthwise_kernel_size=depthwise_kernel_size,
        activation_fn=activation_fn,
        bias=bias,
    ).eval()
    tt_module = TTConvolutionModule(
        device=device,
        embed_dim=embed_dim,
        channels=channels,
        depthwise_kernel_size=depthwise_kernel_size,
        activation_fn=activation_fn,
        bias=bias,
    )

    parameters = {f"conv.{k}": v for k, v in torch_module.state_dict().items()}
    tt_module.load_parameters(parameters=parameters, prefix="conv.")

    torch_x = torch.randn(b, t, embed_dim, dtype=torch.float32)
    torch_output = torch_module(torch_x)

    tt_x = ttnn.from_torch(
        torch_x.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    tt_output = tt_module(tt_x)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32)

    assert tuple(tt_output_torch.shape) == tuple(torch_output.shape)
    assert_with_pcc(torch_output, tt_output_torch, pcc=0.97)
