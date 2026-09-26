# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    data_gen_with_val,
    compare_pcc,
)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("value", [0.05, 1.0, 0.5, 0.12])
def test_bw_addcmul(input_shapes, value, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    tensor1_data, tensor1_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    tensor2_data, tensor2_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    tt_output_tensor_on_device = ttnn.addcmul_bw(grad_tensor, input_tensor, tensor1_tensor, tensor2_tensor, value)

    golden_function = ttnn.get_golden_function(ttnn.addcmul_bw)
    golden_tensor = golden_function(grad_data, in_data, tensor1_data, tensor2_data, value)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


def test_bw_addcmul_no_intermediate_overflow(device):
    # Regression test for #57867: addcmul_bw used to compute (grad * tensorN) before applying `value`,
    # so a large grad/tensor pair could overflow to inf even though grad * value * tensorN is finite.
    # Folding `value` into grad first (matching addcdiv_bw) keeps the intermediate in range.
    # grad*tensorN overflows fp32/bf16 range (max ~3.4e38) before value is applied, but grad*value*tensorN
    # (~1e37) is finite -- exercises the fixed evaluation order without also overflowing in exact math.
    input_shapes = torch.Size([1, 1, 32, 32])
    value = 1e-3

    in_data, input_tensor = data_gen_with_range(input_shapes, -1, 1, device, True)
    tensor1_data, tensor1_tensor = data_gen_with_val(input_shapes, device, True, val=1e20)
    tensor2_data, tensor2_tensor = data_gen_with_val(input_shapes, device, True, val=1e20)
    grad_data, grad_tensor = data_gen_with_val(input_shapes, device, val=1e20)

    tt_output_tensor_on_device = ttnn.addcmul_bw(grad_tensor, input_tensor, tensor1_tensor, tensor2_tensor, value)

    golden_function = ttnn.get_golden_function(ttnn.addcmul_bw)
    golden_tensor = golden_function(grad_data, in_data, tensor1_data, tensor2_data, value)

    assert torch.isfinite(golden_tensor[1]).all()
    assert torch.isfinite(golden_tensor[2]).all()

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status
