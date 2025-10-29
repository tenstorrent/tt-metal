# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def torch_polyval(input_tensor, coeff):
    curVal = 0
    for curValIndex in range(len(coeff) - 1):
        curVal = (curVal + coeff[curValIndex]) * input_tensor[0]
    return curVal + coeff[len(coeff) - 1]


@pytest.mark.parametrize("shape", [(1, 1, 32, 32)])
@pytest.mark.parametrize("coeff", [(1.5, 2.4, 6.7, 9.1)])
def test_polyval(device, shape, coeff):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)

    torch_output_tensor = torch_polyval(torch_input_tensor, coeff)

    input_tensor_a = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.polyval(input_tensor_a, coeff)
    output_tensor = ttnn.to_torch(output_tensor).squeeze(0)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
