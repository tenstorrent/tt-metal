# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range, compare_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_grayskull


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("exponent", [0.5, 2.0, 4])
def test_unary_pow_ttnn(input_shapes, exponent, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.pow(input_tensor, exponent, output_tensor=output_tensor, queue_id=cq_id)
    golden_fn = ttnn.get_golden_function(ttnn.pow)
    golden_tensor = golden_fn(in_data, exponent)

    comp_pass = compare_pcc([output_tensor], [golden_tensor], pcc=0.9)
    assert comp_pass


@skip_for_grayskull()
@pytest.mark.parametrize("n", [2])
@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [128])
@pytest.mark.parametrize("w", [128])
def test_binary_sfpu_pow_4D(device, n, c, h, w):
    torch_input_tensor_a = torch.ones((n, c, h, w), dtype=torch.float32) * 10
    torch_input_tensor_b = torch.ones((n, c, h, w), dtype=torch.float32) * 2.5
    torch_output_tensor = torch.pow(torch_input_tensor_b, torch_input_tensor_a)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.pow(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.99)


@skip_for_grayskull()
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("input", [10.0, 5.5])
def test_binary_pow_scalar_input(input_shapes, input, device):
    torch_input_tensor_b = torch.ones(input_shapes, dtype=torch.float32) * 2.5
    torch_output_tensor = torch.pow(input, torch_input_tensor_b)

    golden_fn = ttnn.get_golden_function(ttnn.pow)
    golden_tensor = golden_fn(input, torch_input_tensor_b)

    cq_id = 0
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.pow(input, input_tensor_b, queue_id=cq_id)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.99)
