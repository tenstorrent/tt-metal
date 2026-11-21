# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""moreh_linear_backward must reject a ROW_MAJOR output_grad.

Every gradient path sizes its reads from output_grad and reads it as whole tiles, so a row-major
gradient is misread and the bias grad comes back mostly wrong with nothing rejecting the call. The
layout requirement was already imposed on the other operands but never on this one, which is the one
each gradient path is derived from.
"""

import pytest
import torch

import ttnn
from tests.ttnn.unit_tests.operations.test_utils import create_ttnn_tilized_tensor, get_compute_kernel_options

pytestmark = pytest.mark.use_module_device

# input, weight, bias, output_grad. Bias is [1, out_features]: the shape the moreh backward cases use.
INPUT_SHAPE = (32, 64)
WEIGHT_SHAPE = (96, 64)
BIAS_SHAPE = (1, 96)
OUTPUT_GRAD_SHAPE = (32, 96)


def _inputs():
    torch.manual_seed(0)
    # Integers keep every accumulated value inside bfloat16's exact integer range, so the control
    # comparison below is exact rather than a tolerance.
    output_grad = torch.randint(-2, 3, OUTPUT_GRAD_SHAPE, dtype=torch.int32)
    x = torch.randint(-2, 3, INPUT_SHAPE, dtype=torch.int32)
    w = torch.randint(-2, 3, WEIGHT_SHAPE, dtype=torch.int32)
    bias = torch.randint(-10, 10, BIAS_SHAPE, dtype=torch.int32)
    return output_grad, x, w, bias


def _to_tile(tensor, device):
    return create_ttnn_tilized_tensor(
        tensor.to(torch.bfloat16), device=device, dtype=ttnn.bfloat16
    )


def _bias_grad(device):
    # Preallocate with NaN, the way the existing backward cases do, so the check reads a buffer the
    # op has to have written.
    return create_ttnn_tilized_tensor(
        torch.full(BIAS_SHAPE, float("nan"), dtype=torch.bfloat16), device=device, dtype=ttnn.bfloat16
    )


def test_moreh_linear_backward_row_major_output_grad_rejected(device, expect_error):
    """A ROW_MAJOR output_grad used to be misread; it must be rejected instead."""
    output_grad, x, w, bias = _inputs()
    tt_output_grad = ttnn.from_torch(
        output_grad.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    with expect_error(RuntimeError, "requires output_grad in TILE layout"):
        ttnn.moreh_linear_backward(
            tt_output_grad,
            _to_tile(x, device),
            _to_tile(w, device),
            are_required_outputs=[False, False, True],
            bias=_to_tile(bias, device),
            bias_grad=_bias_grad(device),
            compute_kernel_config=get_compute_kernel_options(False),
        )


def test_moreh_linear_backward_tile_output_grad_still_works(device):
    """Control: a TILE output_grad still runs and writes the right bias grad."""
    output_grad, x, w, bias = _inputs()

    tt_bias_grad = _bias_grad(device)

    ttnn.moreh_linear_backward(
        _to_tile(output_grad, device),
        _to_tile(x, device),
        _to_tile(w, device),
        are_required_outputs=[False, False, True],
        bias=_to_tile(bias, device),
        bias_grad=tt_bias_grad,
        compute_kernel_config=get_compute_kernel_options(False),
    )

    # The bias grad is output_grad reduced over every non-output dim.
    expected = output_grad.sum(dim=0).to(torch.bfloat16).reshape(BIAS_SHAPE)
    actual = ttnn.to_torch(tt_bias_grad)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert torch.equal(actual.to(torch.int32), expected.to(torch.int32))
