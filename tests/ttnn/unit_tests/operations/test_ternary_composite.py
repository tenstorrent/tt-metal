# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.backward.utility_funcs import data_gen_with_range, compare_pcc


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("value", [1.0, 5.0, 10.0])
def test_ternary_addcmul_ttnn(input_shapes, value, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data3, input_tensor3 = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.addcmul(input_tensor1, input_tensor2, input_tensor3, value)
    golden_tensor = torch.addcmul(in_data1, in_data2, in_data3, value=value)

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
@pytest.mark.parametrize("value", [1.0, 5.0, 10.0])
def test_ternary_addcdiv_ttnn(input_shapes, value, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data3, input_tensor3 = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.addcdiv(input_tensor1, input_tensor2, input_tensor3, value)
    golden_tensor = torch.addcdiv(in_data1, in_data2, in_data3, value=value)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass
