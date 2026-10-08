# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose
from tests.ttnn.nightly.unit_tests.operations.moreh.test_moreh_group_norm import (
    make_input_tensors,
    run_test_moreh_group_norm_backward,
    run_test_moreh_group_norm_backward_gamma_grad_group_index,
    torch_group_norm_backward,
)
from tests.ttnn.unit_tests.operations.test_utils import create_ttnn_tilized_tensor, to_torch, to_ttnn

pytestmark = pytest.mark.use_module_device

# The wrapper runs the input_grad factory when input_grad is requested, and the gamma_beta_grad factory when gamma_grad
# or beta_grad is. Both reuse the layer_norm_backward compute kernels with is_groupnorm set.


def run_moreh_group_norm_backward_nan_pad_test(N, C, num_groups, H, W, device):
    # Not the nightly helpers: they pad tiles with 0, and the zeros in output_grad cancel every padded term, so a
    # broken mask would pass. NaN padding on input and output_grad makes it fail.
    input_shape = (N, C, H, W)
    cpu_input, cpu_gamma, cpu_beta, cpu_output_grad = make_input_tensors(input_shape, True, do_backward=True)
    x_view = cpu_input.view(N, num_groups, -1)
    mean = x_view.mean(dim=-1, keepdim=True)
    rstd = (((x_view - mean) ** 2).mean(dim=-1) + 1e-5).rsqrt()
    expected_input_grad, expected_gamma_grad, expected_beta_grad = torch_group_norm_backward(
        cpu_input.clone(), cpu_output_grad, num_groups, True, True, True, cpu_gamma.clone(), cpu_beta.clone()
    )

    gamma_beta_shape = [1, 1, 1, C]
    mean_rstd_shape = [1, 1, N, num_groups]
    # NaN, so a value the op never writes fails. The checks below read these buffers, not the return value.
    tt_input_grad = to_ttnn(torch.full(input_shape, float("nan")), device=device)
    tt_gamma_grad = to_ttnn(torch.full(gamma_beta_shape, float("nan")), device=device)
    tt_beta_grad = to_ttnn(torch.full(gamma_beta_shape, float("nan")), device=device)
    ttnn.moreh_group_norm_backward(
        create_ttnn_tilized_tensor(cpu_output_grad, device, ttnn.bfloat16),
        create_ttnn_tilized_tensor(cpu_input, device, ttnn.bfloat16),
        to_ttnn(mean, device=device, shape=mean_rstd_shape),
        to_ttnn(rstd, device=device, shape=mean_rstd_shape),
        num_groups,
        are_required_outputs=[True, True, True],
        gamma=to_ttnn(cpu_gamma, device=device, shape=gamma_beta_shape),
        input_grad=tt_input_grad,
        gamma_grad=tt_gamma_grad,
        beta_grad=tt_beta_grad,
    )

    passing, out = comp_allclose(expected_input_grad, to_torch(tt_input_grad, shape=input_shape), rtol=0.1, atol=0.1)
    assert passing, out
    # As in nightly: gamma_grad and beta_grad sum over N * C * Ht * Wt values, so the bfloat16 sum error grows with it.
    divisor = N * C * ((H + 31) // 32) * ((W + 31) // 32)
    for expected, actual in [(expected_gamma_grad, tt_gamma_grad), (expected_beta_grad, tt_beta_grad)]:
        actual = to_torch(actual, shape=gamma_beta_shape).view(C)
        passing, out = comp_allclose(expected / divisor, actual / divisor, rtol=0.1, atol=0.1)
        assert passing, out


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "affine, input_requires_grad, gamma_requires_grad, beta_requires_grad",
    [
        # Both factories, with gamma and both grad defines.
        (True, True, True, True),
        # No gamma: input_grad without GAMMA_HAS_VALUE; the gamma_beta_grad factory is skipped.
        (False, True, False, False),
        # One grad define at a time; the input_grad factory is skipped.
        (True, False, True, False),
        (True, False, False, True),
    ],
    ids=["all_grads", "input_grad_no_affine", "gamma_grad_only", "beta_grad_only"],
)
def test_moreh_group_norm_backward(affine, input_requires_grad, gamma_requires_grad, beta_requires_grad, device):
    torch.manual_seed(0)
    run_test_moreh_group_norm_backward(
        2, [4, 2], [64, 64], 1e-5, affine, input_requires_grad, gamma_requires_grad, beta_requires_grad, device
    )


@pytest.mark.merge_gate
def test_moreh_group_norm_backward_gamma_grad_group_index(device):
    # Regression for #51278 item 5: the gamma_grad reader used the wrong group's mean/rstd.
    run_test_moreh_group_norm_backward_gamma_grad_group_index(2, [4, 2], [32, 32], device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "N, C, num_groups, H, W",
    [
        # 23 x 23 leaves the tile partly filled in H and W, so both factories mask it.
        (2, 4, 2, 23, 23),
        # One group of 4 channels x 16 x 16 tiles is too big for L1: the input_grad large kernels, with masks.
        (2, 4, 1, 500, 500),
    ],
    ids=["small", "large_algorithm"],
)
def test_moreh_group_norm_backward_unaligned(N, C, num_groups, H, W, device):
    torch.manual_seed(0)
    run_moreh_group_norm_backward_nan_pad_test(N, C, num_groups, H, W, device)


@pytest.mark.merge_gate
def test_moreh_group_norm_backward_large_algorithm(device):
    torch.manual_seed(0)
    # One group of 4 channels x 16 x 16 tiles = 1024 tiles: too big for L1, so the input_grad factory runs its large
    # (streaming) kernels.
    run_test_moreh_group_norm_backward(2, [4, 1], [512, 512], 1e-5, True, True, False, False, device)
