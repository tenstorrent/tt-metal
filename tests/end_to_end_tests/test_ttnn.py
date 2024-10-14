# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import ttnn
import torch

import ttnn.operations.binary


@pytest.mark.eager_host_side
def test_ttnn_host_tensor(reset_seeds):
    torch_input_tensor = torch.zeros(2, 4, dtype=torch.float32)
    tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16)
    torch_output_tensor = ttnn.to_torch(tensor)

    expected_output = torch_input_tensor.bfloat16()

    assert torch.allclose(expected_output, torch_output_tensor)


@pytest.mark.eager_package_silicon
def test_ttnn_add(reset_seeds):
    with ttnn.manage_device(device_id=0) as device:
        a_torch = torch.ones((5, 7))
        b_torch = torch.ones((1, 7))

        a = ttnn.from_torch(a_torch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        b = ttnn.from_torch(b_torch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        output = a + b
        output = ttnn.to_torch(output)
