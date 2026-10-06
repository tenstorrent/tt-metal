# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.common.utility_functions import comp_allclose
from tests.ttnn.unit_tests.operations.test_utils import (
    TILE_HEIGHT,
    TILE_WIDTH,
    get_compute_kernel_options,
    to_torch,
    to_ttnn,
)

pytestmark = pytest.mark.use_module_device

INPUT_SHAPE = [2, 4, TILE_HEIGHT, TILE_WIDTH]
NUM_GROUPS = 2
EPS = 1e-5


def run_moreh_group_norm_test(input_shape, num_groups, device, affine=True, create_mean_rstd=True):
    N, C, _, _ = input_shape
    torch_input = torch.empty(input_shape).uniform_(-2, 2)
    torch_gamma = torch.empty([C]).uniform_(-2, 2) if affine else None
    torch_beta = torch.empty([C]).uniform_(-2, 2) if affine else None
    torch_output = F.group_norm(torch_input, num_groups, torch_gamma, torch_beta, EPS)
    x_view = torch_input.view(N, num_groups, -1)
    torch_mean = x_view.mean(dim=-1, keepdim=True)
    torch_rstd = (((x_view - torch_mean) ** 2).mean(dim=-1) + EPS).rsqrt()

    tt_output, tt_mean, tt_rstd = ttnn.moreh_group_norm(
        to_ttnn(torch_input, device=device),
        num_groups,
        EPS,
        to_ttnn(torch_gamma, device=device, shape=[1, 1, 1, C]),
        to_ttnn(torch_beta, device=device, shape=[1, 1, 1, C]),
        are_required_outputs=(True, create_mean_rstd, create_mean_rstd),
        compute_kernel_config=get_compute_kernel_options(False),
    )

    passing, out = comp_allclose(torch_output, to_torch(tt_output), rtol=0.1, atol=0.1)
    assert passing, out
    if not create_mean_rstd:
        assert tt_mean is None and tt_rstd is None
        return
    passing, out = comp_allclose(torch_mean, to_torch(tt_mean, shape=torch_mean.shape), rtol=0.1, atol=0.1)
    assert passing, out
    passing, out = comp_allclose(torch_rstd, to_torch(tt_rstd, shape=torch_rstd.shape), rtol=0.1, atol=0.1)
    assert passing, out


def run_moreh_group_norm_backward_test(
    input_shape, num_groups, device, are_required_outputs=(True, True, True), affine=True, group_scale=1.0
):
    N, C, H, W = input_shape
    torch_input = torch.empty(input_shape).uniform_(-2, 2)
    # Scaling all groups but the first pulls the per-group mean/rstd apart, so reading the wrong group's stats shows.
    torch_input[:, C // num_groups :] *= group_scale
    torch_input.requires_grad_()
    torch_gamma = torch.empty([C]).uniform_(-2, 2).requires_grad_() if affine else None
    torch_beta = torch.empty([C]).uniform_(-2, 2).requires_grad_() if affine else None
    torch_output_grad = torch.empty(input_shape).uniform_(-2, 2)
    torch_output = F.group_norm(torch_input, num_groups, torch_gamma, torch_beta, EPS)
    torch_output.backward(torch_output_grad)
    x_view = torch_input.detach().view(N, num_groups, -1)
    torch_mean = x_view.mean(dim=-1, keepdim=True)
    torch_rstd = (((x_view - torch_mean) ** 2).mean(dim=-1) + EPS).rsqrt()

    tt_input_grad, tt_gamma_grad, tt_beta_grad = ttnn.moreh_group_norm_backward(
        to_ttnn(torch_output_grad, device=device),
        to_ttnn(torch_input.detach(), device=device),
        to_ttnn(torch_mean, device=device, shape=[1, 1, N, num_groups]),
        to_ttnn(torch_rstd, device=device, shape=[1, 1, N, num_groups]),
        num_groups,
        are_required_outputs=are_required_outputs,
        gamma=to_ttnn(torch_gamma.detach(), device=device, shape=[1, 1, 1, C]) if affine else None,
        compute_kernel_config=get_compute_kernel_options(False),
    )

    input_grad_required, gamma_grad_required, beta_grad_required = are_required_outputs
    if input_grad_required:
        passing, out = comp_allclose(torch_input.grad, to_torch(tt_input_grad), rtol=0.1, atol=0.1)
        assert passing, out
    else:
        assert tt_input_grad is None
    # As in nightly, gamma/beta grads are divided by N * C * Ht * Wt because the bf16 sum error grows with it.
    divisor = N * C * ((H + TILE_HEIGHT - 1) // TILE_HEIGHT) * ((W + TILE_WIDTH - 1) // TILE_WIDTH)
    if gamma_grad_required:
        passing, out = comp_allclose(
            torch_gamma.grad / divisor, to_torch(tt_gamma_grad, shape=[C]) / divisor, rtol=0.1, atol=0.1
        )
        assert passing, out
    else:
        assert tt_gamma_grad is None
    if beta_grad_required:
        passing, out = comp_allclose(
            torch_beta.grad / divisor, to_torch(tt_beta_grad, shape=[C]) / divisor, rtol=0.1, atol=0.1
        )
        assert passing, out
    else:
        assert tt_beta_grad is None


@pytest.mark.merge_gate
def test_moreh_group_norm(device):
    torch.manual_seed(0)
    run_moreh_group_norm_test(INPUT_SHAPE, NUM_GROUPS, device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape, num_groups, affine",
    [
        (INPUT_SHAPE, NUM_GROUPS, False),
        (INPUT_SHAPE, 4, True),
        (INPUT_SHAPE, 1, True),
        ([2, 4, 23, 23], NUM_GROUPS, True),
    ],
    ids=["no_affine", "num_groups_eq_c", "one_group", "hw_unaligned"],
)
def test_moreh_group_norm_corner_cases(input_shape, num_groups, affine, device):
    torch.manual_seed(0)
    run_moreh_group_norm_test(input_shape, num_groups, device, affine=affine)


@pytest.mark.merge_gate
def test_moreh_group_norm_no_mean_rstd(device):
    torch.manual_seed(0)
    run_moreh_group_norm_test(INPUT_SHAPE, NUM_GROUPS, device, create_mean_rstd=False)


@pytest.mark.merge_gate
def test_moreh_group_norm_program_cache(device):
    torch.manual_seed(0)
    # Start from an empty cache: the module-scoped device carries entries over from earlier tests.
    device.clear_program_cache()
    run_moreh_group_norm_test(INPUT_SHAPE, NUM_GROUPS, device)
    num_program_cache_entries = device.num_program_cache_entries()
    # Holding this tensor moves the next allocations, so the cache hit must update the buffer addresses.
    tt_placeholder = to_ttnn(torch.zeros(INPUT_SHAPE), device=device)
    run_moreh_group_norm_test(INPUT_SHAPE, NUM_GROUPS, device)
    assert device.num_program_cache_entries() == num_program_cache_entries


@pytest.mark.merge_gate
def test_moreh_group_norm_rejects_zero_groups(device, expect_error):
    torch.manual_seed(0)
    with expect_error(RuntimeError, "num_groups must be greater than 0"):
        ttnn.moreh_group_norm(to_ttnn(torch.zeros(INPUT_SHAPE), device=device), 0, EPS)


@pytest.mark.merge_gate
def test_moreh_group_norm_backward(device):
    torch.manual_seed(0)
    # Requesting all three grads runs both the input-grad and the gamma/beta-grad factories in one call.
    run_moreh_group_norm_backward_test(INPUT_SHAPE, NUM_GROUPS, device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape, num_groups, are_required_outputs, affine",
    [
        (INPUT_SHAPE, NUM_GROUPS, (True, False, False), True),
        (INPUT_SHAPE, NUM_GROUPS, (True, False, False), False),
        (INPUT_SHAPE, NUM_GROUPS, (False, True, True), True),
        (INPUT_SHAPE, NUM_GROUPS, (False, True, False), True),
        (INPUT_SHAPE, NUM_GROUPS, (False, False, True), True),
        ([2, 4, 23, 23], NUM_GROUPS, (True, True, True), True),
    ],
    ids=["input_grad_only", "no_affine", "gamma_beta_grad_only", "gamma_grad_only", "beta_grad_only", "hw_unaligned"],
)
def test_moreh_group_norm_backward_corner_cases(input_shape, num_groups, are_required_outputs, affine, device):
    torch.manual_seed(0)
    run_moreh_group_norm_backward_test(
        input_shape, num_groups, device, are_required_outputs=are_required_outputs, affine=affine
    )


@pytest.mark.merge_gate
def test_moreh_group_norm_backward_gamma_grad_group_index(device):
    torch.manual_seed(0)
    run_moreh_group_norm_backward_test(
        [2, 8, TILE_HEIGHT, TILE_WIDTH], 4, device, are_required_outputs=(False, True, False), group_scale=100.0
    )


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "are_required_outputs",
    [(True, False, False), (False, True, False)],
    ids=["input_grad", "gamma_grad"],
)
def test_moreh_group_norm_backward_rejects_invalid_mean_volume(are_required_outputs, device, expect_error):
    torch.manual_seed(0)
    N, C, H, W = 2, 4, 23, 23
    input_shape = [N, C, H, W]

    with expect_error(RuntimeError, "mean must have logical volume"):
        ttnn.moreh_group_norm_backward(
            to_ttnn(torch.zeros(input_shape), device=device),
            to_ttnn(torch.zeros(input_shape), device=device),
            to_ttnn(torch.zeros([1, 1, N + 1, NUM_GROUPS]), device=device),
            to_ttnn(torch.ones([1, 1, N, NUM_GROUPS]), device=device),
            NUM_GROUPS,
            are_required_outputs=are_required_outputs,
        )
