# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from models.utility_functions import is_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("c", [1])
@pytest.mark.parametrize("h", [10])
@pytest.mark.parametrize("w", [10])
def test_tensor_unpad(device, n, c, h, w):
    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor[:, :, :6, :6]
    activation_pyt_padded = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    activation_pyt_padded = ttnn.slice(activation_pyt_padded, (0, 0, 0, 0), (n - 1, c - 1, 5, 5))

    activation_pyt_padded_out = ttnn.to_torch(activation_pyt_padded)
    assert_with_pcc(torch_output_tensor, activation_pyt_padded_out, 0.9999)
