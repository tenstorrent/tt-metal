# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("shape", [(1, 1, 32, 32)])
@pytest.mark.parametrize(
    "activations",
    [
        [],
        [ttnn.UnaryWithParam(ttnn.UnaryOpType.EQZ)],
        [ttnn.UnaryWithParam(ttnn.UnaryOpType.LTZ)],
        [ttnn.UnaryWithParam(ttnn.UnaryOpType.LEZ)],
        [ttnn.UnaryWithParam(ttnn.UnaryOpType.GTZ)],
        [ttnn.UnaryWithParam(ttnn.UnaryOpType.GEZ)],
        [ttnn.UnaryWithParam(ttnn.UnaryOpType.NEZ)],
    ],
)
@pytest.mark.parametrize(
    "torch_dtype",
    [
        (torch.int32),
        (torch.bfloat16),
    ],
)
def test_add_and_apply_activations(device, shape, activations, torch_dtype):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.Tensor(size=shape).uniform_(-100, 100).to(torch_dtype)
    torch_input_tensor_b = torch.Tensor(size=shape).uniform_(-150, 150).to(torch_dtype)
    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b
    if activations:
        for activation in activations:
            op_type = activation.op_type
            if op_type == ttnn.UnaryOpType.EQZ:
                torch_output_tensor = torch.eq(torch_output_tensor, 0)
            elif op_type == ttnn.UnaryOpType.LTZ:
                torch_output_tensor = torch.lt(torch_output_tensor, 0)
            elif op_type == ttnn.UnaryOpType.LEZ:
                torch_output_tensor = torch.le(torch_output_tensor, 0)
            elif op_type == ttnn.UnaryOpType.GTZ:
                torch_output_tensor = torch.gt(torch_output_tensor, 0)
            elif op_type == ttnn.UnaryOpType.GEZ:
                torch_output_tensor = torch.ge(torch_output_tensor, 0)
            elif op_type == ttnn.UnaryOpType.NEZ:
                torch_output_tensor = torch.ne(torch_output_tensor, 0)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, activations=activations, use_legacy=None)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.99988)
    assert output_tensor.shape == shape
