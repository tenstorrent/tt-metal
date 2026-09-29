# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


# These zero arguments used to reach an integer division on the host, which killed the process
# with SIGFPE instead of raising.


def _tile(device, shape):
    return ttnn.from_torch(torch.randn(shape).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)


def test_moreh_group_norm_num_groups_zero(device, expect_error):
    with expect_error(RuntimeError, "num_groups must be greater than 0"):
        ttnn.operations.moreh.group_norm(_tile(device, (2, 32, 32, 32)), 0, 1e-5)


# input_grad and gamma/beta_grad are separate device operations, each with its own num_groups check.
@pytest.mark.parametrize(
    "are_required_outputs", [(True, False, False), (False, True, False)], ids=["input_grad", "gamma_beta_grad"]
)
def test_moreh_group_norm_backward_num_groups_zero(device, expect_error, are_required_outputs):
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
    x = ttnn.from_torch(torch.randn(1, 36, 64).bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "stride must be greater than 0"):
        ttnn.operations.moreh.fold(
            x, None, output_size=[8, 8], kernel_size=[3, 3], dilation=[1, 1], padding=[1, 1], stride=[0, 1]
        )


def test_moreh_fold_kernel_size_zero(device, expect_error):
    x = ttnn.from_torch(torch.randn(1, 36, 64).bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "kernel_size must be greater than 0"):
        ttnn.operations.moreh.fold(
            x, None, output_size=[8, 8], kernel_size=[0, 3], dilation=[1, 1], padding=[1, 1], stride=[1, 1]
        )


def test_moreh_fold_kernel_size_arity(device, expect_error):
    x = ttnn.from_torch(torch.randn(1, 36, 64).bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "kernel_size takes 2 elements"):
        ttnn.operations.moreh.fold(
            x, None, output_size=[8, 8], kernel_size=[3], dilation=[1, 1], padding=[1, 1], stride=[1, 1]
        )
