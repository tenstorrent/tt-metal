# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


# fmt: off
@pytest.mark.parametrize("input_a,scalar", [
        ([1.0,2.0,3.0],3.0)
    ])
# fmt: on
def test_multiply_with_scalar(device, input_a, scalar):
    # pad the lists with zeros to make it 32 so that it fits nicely on the device.
    input_a += [0.0] * (32 - len(input_a))
    torch_input_tensor_a = torch.as_tensor(input_a, dtype=torch.bfloat16).reshape((1, 1, 1, len(input_a)))
    torch_output = torch.mul(torch_input_tensor_a, scalar)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    input_tensor_a = ttnn.to_device(input_tensor_a, device)

    tt_output = ttnn.mul(input_tensor_a, scalar)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.99)
