# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Checks that moreh ops reject arguments that would be used as host-side divisors (zero or
wrongly sized) with a RuntimeError."""

import pytest
import torch

import ttnn


def _tile(device, shape):
    """Returns a random bfloat16 tensor of the given shape in tile layout on device."""
    return ttnn.from_torch(torch.randn(shape).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)


def test_moreh_group_norm_num_groups_zero(device, expect_error):
    """group_norm rejects num_groups == 0."""
    with expect_error(RuntimeError, "num_groups must be greater than 0"):
        ttnn.operations.moreh.group_norm(_tile(device, (2, 32, 32, 32)), 0, 1e-5)


@pytest.mark.parametrize(
    "are_required_outputs", [(True, False, False), (False, True, False)], ids=["input_grad", "gamma_beta_grad"]
)
def test_moreh_group_norm_backward_num_groups_zero(device, expect_error, are_required_outputs):
    """group_norm_backward rejects num_groups == 0 in both the input_grad and gamma/beta_grad device ops."""
    n, c, num_groups = 2, 32, 4
    output_grad = _tile(device, (n, c, 32, 32))
    x = _tile(device, (n, c, 32, 32))
    mean = _tile(device, (1, 1, n, num_groups))
    rstd = _tile(device, (1, 1, n, num_groups))
    with expect_error(RuntimeError, "num_groups must be greater than 0"):
        ttnn.operations.moreh.group_norm_backward(
            output_grad, x, mean, rstd, 0, are_required_outputs=list(are_required_outputs)
        )


def test_moreh_fold_stride_zero(device, expect_error):
    """fold rejects a zero stride."""
    x = ttnn.from_torch(torch.randn(1, 36, 64).bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "stride must be greater than 0"):
        ttnn.operations.moreh.fold(
            x, None, output_size=[8, 8], kernel_size=[3, 3], dilation=[1, 1], padding=[1, 1], stride=[0, 1]
        )


@pytest.mark.parametrize("preallocated_output", [False, True], ids=["no_output", "preallocated_output"])
def test_moreh_fold_kernel_size_zero(device, expect_error, preallocated_output):
    """fold rejects a zero kernel_size, both when computing the output spec and with a preallocated output."""
    x = ttnn.from_torch(torch.randn(1, 36, 64).bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output = (
        ttnn.from_torch(torch.zeros(1, 4, 8, 8).bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        if preallocated_output
        else None
    )
    with expect_error(RuntimeError, "kernel_size must be greater than 0"):
        ttnn.operations.moreh.fold(
            x, output, output_size=[8, 8], kernel_size=[0, 3], dilation=[1, 1], padding=[1, 1], stride=[1, 1]
        )


@pytest.mark.parametrize(
    "output_size, kernel_size, message",
    [([8, 8], [3], "kernel_size takes 2 elements"), ([8], [3, 3], "output_size takes 2 elements")],
    ids=["kernel_size", "output_size"],
)
def test_moreh_fold_size_arity(device, expect_error, output_size, kernel_size, message):
    """fold rejects output_size or kernel_size that does not have exactly 2 elements."""
    x = ttnn.from_torch(torch.randn(1, 36, 64).bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, message):
        ttnn.operations.moreh.fold(
            x, None, output_size=output_size, kernel_size=kernel_size, dilation=[1, 1], padding=[1, 1], stride=[1, 1]
        )
