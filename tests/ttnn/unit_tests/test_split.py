# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0


@skip_for_wormhole_b0()
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("split_size", [2, 4])
@pytest.mark.parametrize("dim", [-1, -2])
def test_split(device, h, w, split_size, dim):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensors = torch.split(torch_input_tensor, split_size, dim=dim)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensors = ttnn.split(input_tensor, split_size=split_size, dim=dim)

    for torch_output_tensor, output_tensor in zip(torch_output_tensors, output_tensors):
        output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.from_device(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
