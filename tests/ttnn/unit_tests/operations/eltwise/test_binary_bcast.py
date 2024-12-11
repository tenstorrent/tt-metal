# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import random
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range, compare_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 1, 1]), torch.Size([5, 3, 32, 32])),
        (torch.Size([5, 1, 64, 1]), torch.Size([1, 3, 1, 128])),
        (torch.Size([5, 1, 1, 64]), torch.Size([1, 3, 128, 1])),
    ),
)
def test_binary_scalar_ops(input_shapes, device):
    a_shape, b_shape = input_shapes
    a_pt = torch.rand(a_shape).bfloat16()
    b_pt = torch.rand(b_shape).bfloat16()

    a_tt = ttnn.from_torch(a_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b_tt = ttnn.from_torch(b_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    cq_id = 0
    out_tt = ttnn.experimental.add(a_tt, b_tt, queue_id=cq_id)
    out_pt = a_pt + b_pt

    comp_pass = compare_pcc([out_tt], [out_pt])
    assert comp_pass


@pytest.mark.parametrize("shape", [(1, 1, 4, 32)])
@pytest.mark.parametrize("activations", [None, [ttnn.UnaryWithParam(ttnn.UnaryOpType.SQRT)]])
def test_add_and_apply_activations(device, shape, activations):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand(shape, dtype=torch.bfloat16) + 2
    torch_input_tensor_b = torch.rand(shape, dtype=torch.bfloat16) + 2
    torch_output_tensor = torch_input_tensor_a + 2
    if activations is not None:
        # for activation in activations:
        #     if activation == "relu":
        torch_output_tensor = torch.sqrt(torch_output_tensor)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.experimental.add(input_tensor_a, input_tensor_b, activations=activations)
    output_tensor = ttnn.to_torch(output_tensor)
    print("torch_output_tensor", torch_output_tensor)
    print("output_tensor", output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.99)
    assert output_tensor.shape == shape
