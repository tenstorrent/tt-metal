# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0


def rms_norm(hidden_states, weight, *, epsilon=1e-6):
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + epsilon)

    if weight.dtype in [torch.float16, torch.bfloat16]:
        hidden_states = hidden_states.to(weight.dtype)

    return weight * hidden_states


@skip_for_wormhole_b0()
@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("h", [32, 384])
@pytest.mark.parametrize("w", [64, 1024])
@pytest.mark.parametrize("input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_rms_norm(device, batch_size, h, w, input_layout):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)
    torch_output_tensor = rms_norm(torch_input_tensor, torch_weight)

    input_tensor = ttnn.from_torch(torch_input_tensor, device=device, layout=input_layout)
    weight = ttnn.from_torch(torch_weight, device=device)
    output_tensor = ttnn.rms_norm(input_tensor, weight)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)
