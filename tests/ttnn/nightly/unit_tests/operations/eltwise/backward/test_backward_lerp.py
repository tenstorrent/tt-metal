# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range
from tests.ttnn.utils_for_testing import assert_with_ulp

# Per output, measured on Wormhole: at most 3, 0 and 3 ULP with a tensor weight, and 1 ULP with a scalar weight.
TENSOR_WEIGHT_ULP_THRESHOLDS = [3, 0, 3]
SCALAR_WEIGHT_ULP_THRESHOLDS = [1, 1]


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_lerp(input_shapes, device):
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 101, device, seed=4)
    in_data, input_tensor = data_gen_with_range(input_shapes, -200, 201, device, True, seed=1)
    end_data, end_tensor = data_gen_with_range(input_shapes, -199, 199, device, True, seed=2)
    weight_data, weight_tensor = data_gen_with_range(input_shapes, -201, 201, device, True, seed=3)

    tt_output_tensor_on_device = ttnn.lerp_bw(grad_tensor, input_tensor, end_tensor, weight_tensor)

    golden_function = ttnn.get_golden_function(ttnn.lerp_bw)
    golden_tensor = golden_function(grad_data, in_data, end_data, weight_data)
    for i in range(3):
        assert_with_ulp(
            expected_result=golden_tensor[i],
            actual_result=tt_output_tensor_on_device[i],
            ulp_threshold=TENSOR_WEIGHT_ULP_THRESHOLDS[i],
        )


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("weight", [-0.25, -25.0, 0.05, 1.0, 25.0])
def test_bw_lerp_weight_scalar(input_shapes, weight, device):
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 101, device, seed=4)
    in_data, input_tensor = data_gen_with_range(input_shapes, -200, 201, device, True, seed=1)
    end_data, end_tensor = data_gen_with_range(input_shapes, -199, 199, device, True, seed=2)

    tt_output_tensor_on_device = ttnn.lerp_bw(grad_tensor, input_tensor, end_tensor, weight)

    golden_function = ttnn.get_golden_function(ttnn.lerp_bw)
    golden_tensor = golden_function(grad_data, in_data, end_data, weight)

    for i in range(2):
        assert_with_ulp(
            expected_result=golden_tensor[i],
            actual_result=tt_output_tensor_on_device[i],
            ulp_threshold=SCALAR_WEIGHT_ULP_THRESHOLDS[i],
        )


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "are_required_outputs",
    [[True, True, True], [True, False, False], [False, True, False], [False, False, True]],
)
def test_bw_lerp_tensor_weight_required_outputs(input_shapes, are_required_outputs, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True, seed=1)
    end_data, end_tensor = data_gen_with_range(input_shapes, -100, 100, device, True, seed=2)
    weight_data, weight_tensor = data_gen_with_range(input_shapes, -10, 10, device, True, seed=3)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device, seed=4)

    tt_output_tensor_on_device = ttnn.lerp_bw(
        grad_tensor, input_tensor, end_tensor, weight_tensor, are_required_outputs=are_required_outputs
    )

    golden_function = ttnn.get_golden_function(ttnn.lerp_bw)
    golden_tensor = golden_function(grad_data, in_data, end_data, weight_data)

    assert len(tt_output_tensor_on_device) == 3
    for i in range(len(are_required_outputs)):
        if are_required_outputs[i]:
            assert_with_ulp(
                expected_result=golden_tensor[i],
                actual_result=tt_output_tensor_on_device[i],
                ulp_threshold=TENSOR_WEIGHT_ULP_THRESHOLDS[i],
            )
        else:
            assert tt_output_tensor_on_device[i] is None


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("weight", [0.0, 0.25, 1.0])
@pytest.mark.parametrize("are_required_outputs", [[True, True], [True, False], [False, True]])
def test_bw_lerp_scalar_weight_required_outputs(input_shapes, weight, are_required_outputs, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True, seed=1)
    end_data, end_tensor = data_gen_with_range(input_shapes, -100, 100, device, True, seed=2)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device, seed=4)

    tt_output_tensor_on_device = ttnn.lerp_bw(
        grad_tensor, input_tensor, end_tensor, weight, are_required_outputs=are_required_outputs
    )

    golden_function = ttnn.get_golden_function(ttnn.lerp_bw)
    golden_tensor = golden_function(grad_data, in_data, end_data, weight)

    assert len(tt_output_tensor_on_device) == 2
    for i in range(len(are_required_outputs)):
        if are_required_outputs[i]:
            assert_with_ulp(
                expected_result=golden_tensor[i],
                actual_result=tt_output_tensor_on_device[i],
                ulp_threshold=SCALAR_WEIGHT_ULP_THRESHOLDS[i],
            )
        else:
            assert tt_output_tensor_on_device[i] is None
