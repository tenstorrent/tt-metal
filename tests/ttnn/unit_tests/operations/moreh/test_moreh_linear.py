# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose_and_pcc
from tests.ttnn.unit_tests.operations.test_utils import get_compute_kernel_options, to_ttnn

pytestmark = pytest.mark.use_module_device

INPUT_SHAPE = [31, 31]
WEIGHT_SHAPE = [30, 31]
# Every batch dim of the input must be summed away for weight_grad and bias_grad.
RANK_4_INPUT_SHAPE = [4, 4, 2, 31]

# bias_grad [1, 1] is a scalar -> SingleCoreProgramFactory, [1, 30] -> MultiCoreProgramFactory.
BIAS_SHAPES = [[1, 1], [1, 30]]
BIAS_IDS = ["scalar_bias", "vector_bias"]


def run_moreh_linear_test(input_shape, weight_shape, bias_shape, device):
    torch_input = torch.randint(-2, 3, input_shape, dtype=torch.bfloat16)
    torch_weight = torch.randint(-2, 3, weight_shape, dtype=torch.bfloat16)
    torch_bias = torch.randint(-10, 10, bias_shape, dtype=torch.bfloat16) if bias_shape is not None else None
    torch_output = torch.nn.functional.linear(torch_input, torch_weight, torch_bias)

    tt_input = to_ttnn(torch_input, device=device)
    tt_weight = to_ttnn(torch_weight, device=device)
    tt_bias = to_ttnn(torch_bias, device=device) if torch_bias is not None else None
    tt_output = ttnn.to_torch(
        ttnn.moreh_linear(tt_input, tt_weight, bias=tt_bias, compute_kernel_config=get_compute_kernel_options(False))
    )

    passing, output_pcc = comp_allclose_and_pcc(torch_output, tt_output, pcc=0.999, rtol=0.1, atol=0.1)
    assert passing, output_pcc


def run_moreh_linear_backward_test(
    input_shape, weight_shape, bias_shape, device, are_required_outputs=(True, True, True)
):
    torch_input = torch.randint(-2, 3, input_shape, dtype=torch.bfloat16, requires_grad=True)
    torch_weight = torch.randint(-2, 3, weight_shape, dtype=torch.bfloat16, requires_grad=True)
    torch_bias = torch.randint(-10, 10, bias_shape, dtype=torch.bfloat16, requires_grad=True)
    torch_output = torch.nn.functional.linear(torch_input, torch_weight, torch_bias)
    torch_output_grad = torch.randint(-2, 3, torch_output.shape, dtype=torch.bfloat16)
    torch_output.backward(torch_output_grad)

    input_requires_grad, weight_requires_grad, bias_requires_grad = are_required_outputs
    tt_output_grad = to_ttnn(torch_output_grad, device=device)
    tt_input = to_ttnn(torch_input.detach(), device=device)
    tt_weight = to_ttnn(torch_weight.detach(), device=device)
    tt_bias = to_ttnn(torch_bias.detach(), device=device)
    tt_input_grad = to_ttnn(torch.full(input_shape, float("nan")), device=device) if input_requires_grad else None
    tt_weight_grad = to_ttnn(torch.full(weight_shape, float("nan")), device=device) if weight_requires_grad else None
    tt_bias_grad = to_ttnn(torch.full(bias_shape, float("nan")), device=device) if bias_requires_grad else None
    tt_grads = ttnn.moreh_linear_backward(
        tt_output_grad,
        tt_input,
        tt_weight,
        are_required_outputs=are_required_outputs,
        bias=tt_bias,
        input_grad=tt_input_grad,
        weight_grad=tt_weight_grad,
        bias_grad=tt_bias_grad,
        compute_kernel_config=get_compute_kernel_options(False),
    )

    torch_grads = (torch_input.grad, torch_weight.grad, torch_bias.grad)
    for required, torch_grad, tt_grad in zip(are_required_outputs, torch_grads, tt_grads):
        if not required:
            assert tt_grad is None
            continue
        passing, output_pcc = comp_allclose_and_pcc(torch_grad, ttnn.to_torch(tt_grad), pcc=0.999, rtol=0.1, atol=0.1)
        assert passing, output_pcc


@pytest.mark.merge_gate
def test_moreh_linear(device):
    torch.manual_seed(0)
    run_moreh_linear_test(INPUT_SHAPE, WEIGHT_SHAPE, [1, 30], device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape, bias_shape",
    [
        (INPUT_SHAPE, None),
        (INPUT_SHAPE, [1, 1]),
        (RANK_4_INPUT_SHAPE, [1, 30]),
    ],
    ids=["no_bias", "scalar_bias", "rank_4"],
)
def test_moreh_linear_corner_cases(input_shape, bias_shape, device):
    torch.manual_seed(0)
    run_moreh_linear_test(input_shape, WEIGHT_SHAPE, bias_shape, device)


@pytest.mark.merge_gate
def test_moreh_linear_program_cache(device):
    torch.manual_seed(0)
    # Start from an empty cache: the module-scoped device carries entries over from earlier tests.
    device.clear_program_cache()
    run_moreh_linear_test(INPUT_SHAPE, WEIGHT_SHAPE, [1, 30], device)
    num_program_cache_entries = device.num_program_cache_entries()
    # Holding this tensor moves the next allocations, so the cache hit must update the buffer addresses.
    tt_placeholder = to_ttnn(torch.zeros(INPUT_SHAPE), device=device)
    run_moreh_linear_test(INPUT_SHAPE, WEIGHT_SHAPE, [1, 30], device)
    assert device.num_program_cache_entries() == num_program_cache_entries


@pytest.mark.merge_gate
@pytest.mark.parametrize("bias_shape", BIAS_SHAPES, ids=BIAS_IDS)
def test_moreh_linear_backward(bias_shape, device):
    torch.manual_seed(0)
    run_moreh_linear_backward_test(INPUT_SHAPE, WEIGHT_SHAPE, bias_shape, device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape, bias_shape, are_required_outputs",
    [
        (INPUT_SHAPE, [1, 30], (True, False, False)),
        (INPUT_SHAPE, [1, 30], (False, True, False)),
        (INPUT_SHAPE, [1, 30], (True, True, False)),
        (RANK_4_INPUT_SHAPE, [1, 1], (True, True, True)),
        (RANK_4_INPUT_SHAPE, [1, 30], (True, True, True)),
    ],
    ids=["input_grad_only", "weight_grad_only", "no_bias_grad", "rank_4_scalar_bias", "rank_4_vector_bias"],
)
def test_moreh_linear_backward_corner_cases(input_shape, bias_shape, are_required_outputs, device):
    torch.manual_seed(0)
    run_moreh_linear_backward_test(
        input_shape, WEIGHT_SHAPE, bias_shape, device, are_required_outputs=are_required_outputs
    )


@pytest.mark.merge_gate
def test_moreh_linear_backward_program_cache(device):
    torch.manual_seed(0)
    # Start from an empty cache: the module-scoped device carries entries over from earlier tests.
    device.clear_program_cache()
    run_moreh_linear_backward_test(INPUT_SHAPE, WEIGHT_SHAPE, [1, 30], device)
    num_program_cache_entries = device.num_program_cache_entries()
    # Holding this tensor moves the next allocations, so the cache hit must update the buffer addresses.
    tt_placeholder = to_ttnn(torch.zeros(INPUT_SHAPE), device=device)
    run_moreh_linear_backward_test(INPUT_SHAPE, WEIGHT_SHAPE, [1, 30], device)
    assert device.num_program_cache_entries() == num_program_cache_entries
