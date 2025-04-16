# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def run_unary_test_sharded(device, hw, out_channels, vector_mode, approx_mode, ttnn_function, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((hw, out_channels), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.sigmoid(input_tensor, vector_mode=vector_mode, fast_and_approximate_mode=approx_mode)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [2048])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("out_channels", [2])
@pytest.mark.parametrize("vector_mode", [2, 4])
@pytest.mark.parametrize("approx_mode", [True, False])
def test_sigmoid_two_out_channels(device, h, w, out_channels, vector_mode, approx_mode):
    torch.manual_seed(0)
    run_unary_test_sharded(device, h * w, out_channels, vector_mode, approx_mode, ttnn.sigmoid, pcc=0.991)
