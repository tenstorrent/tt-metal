# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import data_gen_with_range, compare_pcc
from models.utility_functions import is_grayskull


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_square_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.square(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.square(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_power_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    exponent = 2
    ttnn.pow(input_tensor, exponent, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.pow(in_data, exponent)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_floor_ttnn(input_shapes, device):
    if is_grayskull():
        pytest.skip("Op is only supported on Wormhole")
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.floor(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.floor(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass
