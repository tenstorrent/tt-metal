# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose_and_pcc
from tests.ttnn.unit_tests.operations.test_utils import (
    TILE_HEIGHT,
    TILE_WIDTH,
    create_ttnn_tilized_tensor,
    get_compute_kernel_options,
)

pytestmark = pytest.mark.use_module_device


def run_moreh_matmul_test(
    input_shape,
    other_shape,
    device,
    transpose_input=False,
    transpose_other=False,
    bias_shape=None,
    fp32_dest_acc_en=False,
):
    torch_input = torch.randint(-2, 3, input_shape, dtype=torch.bfloat16)
    torch_other = torch.randint(-2, 3, other_shape, dtype=torch.bfloat16)
    torch_output = torch.matmul(
        torch_input.transpose(-1, -2) if transpose_input else torch_input,
        torch_other.transpose(-1, -2) if transpose_other else torch_other,
    )
    tt_bias = None
    if bias_shape is not None:
        torch_bias = torch.randint(-10, 10, bias_shape, dtype=torch.bfloat16)
        torch_output = torch_output + torch_bias
        tt_bias = create_ttnn_tilized_tensor(torch_bias, device, ttnn.bfloat16)

    tt_input = create_ttnn_tilized_tensor(torch_input, device, ttnn.bfloat16)
    tt_other = create_ttnn_tilized_tensor(torch_other, device, ttnn.bfloat16)
    tt_output = ttnn.to_torch(
        ttnn.moreh_matmul(
            tt_input,
            tt_other,
            transpose_input=transpose_input,
            transpose_other=transpose_other,
            bias=tt_bias,
            compute_kernel_config=get_compute_kernel_options(fp32_dest_acc_en),
        )
    )

    passing, output_pcc = comp_allclose_and_pcc(torch_output, tt_output, pcc=0.999, rtol=0.1, atol=0.1)
    assert passing, output_pcc


def run_moreh_matmul_backward_test(input_shape, other_shape, device, are_required_outputs=(True, True)):
    torch_input = torch.randint(-2, 3, input_shape, dtype=torch.bfloat16, requires_grad=True)
    torch_other = torch.randint(-2, 3, other_shape, dtype=torch.bfloat16, requires_grad=True)
    torch_output = torch.matmul(torch_input, torch_other)
    torch_output_grad = torch.randint(-2, 3, torch_output.shape, dtype=torch.bfloat16)
    torch_output.backward(torch_output_grad)

    input_requires_grad, other_requires_grad = are_required_outputs
    tt_output_grad = create_ttnn_tilized_tensor(torch_output_grad, device, ttnn.bfloat16)
    tt_input = create_ttnn_tilized_tensor(torch_input.detach(), device, ttnn.bfloat16)
    tt_other = create_ttnn_tilized_tensor(torch_other.detach(), device, ttnn.bfloat16)
    tt_input_grad = (
        create_ttnn_tilized_tensor(torch.full(input_shape, float("nan")), device, ttnn.bfloat16)
        if input_requires_grad
        else None
    )
    tt_other_grad = (
        create_ttnn_tilized_tensor(torch.full(other_shape, float("nan")), device, ttnn.bfloat16)
        if other_requires_grad
        else None
    )
    tt_grads = ttnn.moreh_matmul_backward(
        tt_output_grad,
        tt_input,
        tt_other,
        are_required_outputs=are_required_outputs,
        input_a_grad=tt_input_grad,
        input_b_grad=tt_other_grad,
        compute_kernel_config=get_compute_kernel_options(False),
    )

    for required, torch_grad, tt_grad in zip(are_required_outputs, (torch_input.grad, torch_other.grad), tt_grads):
        if not required:
            assert tt_grad is None
            continue
        passing, output_pcc = comp_allclose_and_pcc(torch_grad, ttnn.to_torch(tt_grad), pcc=0.999, rtol=0.1, atol=0.1)
        assert passing, output_pcc


@pytest.mark.merge_gate
def test_moreh_matmul(device):
    torch.manual_seed(0)
    run_moreh_matmul_test([TILE_HEIGHT, TILE_WIDTH], [TILE_WIDTH, TILE_HEIGHT], device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape, other_shape, transpose_input, transpose_other, bias_shape",
    [
        ([TILE_HEIGHT * 2, TILE_WIDTH * 3], [TILE_HEIGHT * 2, TILE_WIDTH], True, False, None),
        ([TILE_HEIGHT * 3, TILE_WIDTH * 2], [TILE_HEIGHT, TILE_WIDTH * 2], False, True, None),
        ([TILE_HEIGHT * 2, TILE_WIDTH * 3], [TILE_HEIGHT, TILE_WIDTH * 2], True, True, None),
        # Partial last tiles in both H and W of both operands: exercises the input and other masks.
        ([TILE_HEIGHT - 1, TILE_WIDTH * 2 - 1], [TILE_HEIGHT * 2 - 1, TILE_WIDTH + 1], False, False, None),
        ([TILE_HEIGHT * 2 - 1, TILE_WIDTH - 1], [TILE_HEIGHT + 1, TILE_WIDTH * 2 - 1], True, True, None),
        ([3, TILE_HEIGHT * 2, TILE_WIDTH], [TILE_HEIGHT, TILE_WIDTH * 2], False, False, None),
        # Each operand broadcasts along a different batch dim.
        ([2, 1, TILE_HEIGHT, TILE_WIDTH], [1, 3, TILE_HEIGHT, TILE_WIDTH * 2], False, False, None),
        ([TILE_HEIGHT * 2, TILE_WIDTH], [TILE_HEIGHT, TILE_WIDTH * 2], False, False, [1, TILE_WIDTH * 2]),
        ([TILE_HEIGHT * 2, TILE_WIDTH], [TILE_HEIGHT, TILE_WIDTH * 2], False, False, [1, 1]),
    ],
    ids=[
        "transpose_input",
        "transpose_other",
        "transpose_both",
        "unaligned",
        "unaligned_transpose_both",
        "batched_broadcast",
        "rank_4_broadcast",
        "vector_bias",
        "scalar_bias",
    ],
)
def test_moreh_matmul_corner_cases(input_shape, other_shape, transpose_input, transpose_other, bias_shape, device):
    torch.manual_seed(0)
    run_moreh_matmul_test(
        input_shape,
        other_shape,
        device,
        transpose_input=transpose_input,
        transpose_other=transpose_other,
        bias_shape=bias_shape,
    )


@pytest.mark.merge_gate
def test_moreh_matmul_fp32_dest_acc(device):
    torch.manual_seed(0)
    # K = 3100 spans 97 tiles with a partial last tile, accumulated in the fp32 dest/intermediate buffers.
    run_moreh_matmul_test(
        [3100, TILE_WIDTH - 1], [3100, TILE_WIDTH - 1], device, transpose_input=True, fp32_dest_acc_en=True
    )


@pytest.mark.merge_gate
def test_moreh_matmul_program_cache(device):
    torch.manual_seed(0)
    # Start from an empty cache: the module-scoped device carries entries over from earlier tests.
    device.clear_program_cache()
    run_moreh_matmul_test([TILE_HEIGHT, TILE_WIDTH], [TILE_WIDTH, TILE_HEIGHT], device)
    num_program_cache_entries = device.num_program_cache_entries()
    # Holding this tensor moves the next allocations, so the cache hit must update the buffer addresses.
    tt_placeholder = create_ttnn_tilized_tensor(torch.zeros([TILE_HEIGHT, TILE_WIDTH]), device, ttnn.bfloat16)
    run_moreh_matmul_test([TILE_HEIGHT, TILE_WIDTH], [TILE_WIDTH, TILE_HEIGHT], device)
    assert device.num_program_cache_entries() == num_program_cache_entries


@pytest.mark.merge_gate
def test_moreh_matmul_backward(device):
    torch.manual_seed(0)
    # other has no batch dim, so its grad runs moreh_matmul and then moreh_sum over the input's batch dim.
    run_moreh_matmul_backward_test([2, TILE_HEIGHT, TILE_WIDTH], [TILE_WIDTH, TILE_HEIGHT], device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape, other_shape, are_required_outputs",
    [
        ([2, TILE_HEIGHT, TILE_WIDTH], [TILE_WIDTH, TILE_HEIGHT], (True, False)),
        ([2, TILE_HEIGHT, TILE_WIDTH], [TILE_WIDTH, TILE_HEIGHT], (False, True)),
        ([2, TILE_HEIGHT - 1, TILE_WIDTH + 1], [TILE_HEIGHT + 1, TILE_WIDTH * 2 - 1], (True, True)),
        # Both grads need a moreh_sum: input over the leading batch dim, other over its size-1 batch dim.
        ([2, TILE_HEIGHT, TILE_WIDTH], [2, 2, 1, TILE_HEIGHT, TILE_WIDTH], (True, True)),
    ],
    ids=["input_grad_only", "other_grad_only", "unaligned", "batched_broadcast"],
)
def test_moreh_matmul_backward_corner_cases(input_shape, other_shape, are_required_outputs, device):
    torch.manual_seed(0)
    run_moreh_matmul_backward_test(input_shape, other_shape, device, are_required_outputs=are_required_outputs)
