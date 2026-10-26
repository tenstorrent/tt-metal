# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range
from tests.ttnn.utils_for_testing import assert_with_ulp

# Per output: the grad passthrough is exact; the tensor1 and tensor2 gradients measured at most 3 and 5 ULP on Wormhole.
ULP_THRESHOLDS = [0, 3, 5]


def assert_addcdiv_bw_output(tt_output, golden, ulp_threshold):
    # Where tensor2 is 0 the op returns +inf, -inf or NaN by output while torch signs the infinity by grad * value,
    # so non-finite values are checked by position and ULP is measured on the finite ones.
    output = ttnn.to_torch(tt_output)
    finite = torch.isfinite(golden)
    assert torch.equal(torch.isfinite(output), finite)
    assert_with_ulp(expected_result=golden[finite], actual_result=output[finite], ulp_threshold=ulp_threshold)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("value", [0.05, 1.0, 0.5, 5.0])
def test_bw_addcdiv(input_shapes, value, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True, seed=1)
    tensor1_data, tensor1_tensor = data_gen_with_range(input_shapes, -100, 100, device, True, seed=2)
    tensor2_data, tensor2_tensor = data_gen_with_range(input_shapes, -100, 100, device, True, seed=3)

    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device, False, seed=4)

    tt_output_tensor_on_device = ttnn.addcdiv_bw(grad_tensor, input_tensor, tensor1_tensor, tensor2_tensor, value)

    golden_function = ttnn.get_golden_function(ttnn.addcdiv_bw)
    golden_tensor = golden_function(grad_data, in_data, tensor1_data, tensor2_data, value)

    for i in range(3):
        assert_addcdiv_bw_output(tt_output_tensor_on_device[i], golden_tensor[i], ULP_THRESHOLDS[i])


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("value", [0.05, 1.0])
@pytest.mark.parametrize(
    "are_required_outputs",
    [[True, True, True], [True, False, False], [False, True, False], [False, False, True], [False, True, True]],
)
def test_bw_addcdiv_required_outputs(input_shapes, value, are_required_outputs, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True, seed=1)
    tensor1_data, tensor1_tensor = data_gen_with_range(input_shapes, -100, 100, device, True, seed=2)
    tensor2_data, tensor2_tensor = data_gen_with_range(input_shapes, -50, 50, device, True, seed=3)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device, seed=4)

    tt_output_tensor_on_device = ttnn.addcdiv_bw(
        grad_tensor, input_tensor, tensor1_tensor, tensor2_tensor, value, are_required_outputs=are_required_outputs
    )

    golden_function = ttnn.get_golden_function(ttnn.addcdiv_bw)
    golden_tensor = golden_function(grad_data, in_data, tensor1_data, tensor2_data, value)

    assert len(tt_output_tensor_on_device) == 3
    for i in range(len(are_required_outputs)):
        if are_required_outputs[i]:
            assert_addcdiv_bw_output(tt_output_tensor_on_device[i], golden_tensor[i], ULP_THRESHOLDS[i])
        else:
            assert tt_output_tensor_on_device[i] is None
