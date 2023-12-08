# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from torch.nn import functional as F


# fmt: off
@pytest.mark.parametrize("h", [2 * 32])
@pytest.mark.parametrize("w", [4 * 32])
# fmt: on
def test_tanh_not_4D(device, h, w):
    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.tanh(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)

    output = ttnn.tanh(input_tensor)
    output = ttnn.from_device(output)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.99)
