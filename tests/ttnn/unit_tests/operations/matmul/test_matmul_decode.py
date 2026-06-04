# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "m, k, n",
    [
        (32, 1024, 4096),
    ],
)
def test_matmul_decode(device, m, k, n):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.randn((m, k), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k, n), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a.to(torch.float32) @ torch_input_tensor_b.to(torch.float32)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.matmul_decode(input_tensor_a, input_tensor_b)

    assert output_tensor.shape == (m, n)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.99)
