# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.rvc.tt_impl.linear import Linear
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("activation", [None, "relu"])
def test_linear(device, activation):
    torch.manual_seed(0)

    batch_size = 1
    input_length = 64
    in_features = 32
    out_features = 48

    torch_linear = torch.nn.Linear(in_features, out_features, bias=True).eval()

    torch_input = torch.randn(batch_size, input_length, in_features, dtype=torch.float32)
    torch_output = torch_linear(torch_input)
    if activation == "relu":
        torch_output = torch.relu(torch_output)

    tt_linear = Linear(
        device=device,
        in_features=in_features,
        out_features=out_features,
        dtype=ttnn.bfloat16,
        activation=activation,
    )
    parameters = {
        "proj.linear.weight": torch_linear.weight,
        "proj.linear.bias": torch_linear.bias,
    }
    tt_linear.load_state_dict(parameters=parameters, key="linear", module_prefix="proj.")

    tt_input = ttnn.from_torch(
        torch_input.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    tt_output = tt_linear(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32)

    assert_with_pcc(torch_output, tt_output_torch, pcc=0.99)
