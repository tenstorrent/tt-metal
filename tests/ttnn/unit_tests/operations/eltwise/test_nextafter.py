# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("shape", [(1, 1, 32, 32)])
def test_nextafter(device, shape):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand(shape, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shape, dtype=torch.bfloat16)

    torch_output_tensor = torch.nextafter(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.nextafter(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
