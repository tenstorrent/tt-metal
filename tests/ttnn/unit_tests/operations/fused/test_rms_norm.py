# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("h", [32, 384])
@pytest.mark.parametrize("w", [64, 1024])
def test_rms_norm(device, batch_size, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn.rms_norm)
    torch_output_tensor = golden_function(torch_input_tensor, torch_weight)

    input_tensor = ttnn.from_torch(torch_input_tensor, device=device, layout=ttnn.TILE_LAYOUT)
    weight = ttnn.from_torch(torch_weight, device=device, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn.rms_norm(input_tensor, weight=weight)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)
