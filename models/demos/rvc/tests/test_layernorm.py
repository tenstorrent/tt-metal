# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.rvc.tt_impl.layernorm import TTLayerNorm
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_layernorm(device):
    torch.manual_seed(0)

    batch_size = 2
    input_length = 48
    channels = 64

    torch_layernorm = torch.nn.LayerNorm(channels, eps=1e-5, elementwise_affine=True).eval()

    torch_input = torch.randn(batch_size, input_length, channels, dtype=torch.float32)
    torch_output = torch_layernorm(torch_input)

    tt_layernorm = TTLayerNorm(
        device=device,
        normalized_shape=channels,
        eps=1e-5,
        dtype=ttnn.bfloat16,
    )
    parameters = {
        "norm.layernorm.weight": torch_layernorm.weight,
        "norm.layernorm.bias": torch_layernorm.bias,
    }
    tt_layernorm.load_parameters(parameters=parameters, key="layernorm", prefix="norm.")

    tt_input = ttnn.from_torch(
        torch_input.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    tt_output = tt_layernorm(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32)

    assert tuple(tt_output_torch.shape) == tuple(torch_output.shape)
    assert_with_pcc(torch_output, tt_output_torch, pcc=0.99)
