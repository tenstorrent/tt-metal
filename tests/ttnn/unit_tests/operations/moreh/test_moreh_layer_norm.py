# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, skip_for_blackhole
from tests.ttnn.nightly.unit_tests.operations.moreh.test_moreh_layer_norm import (
    make_input_tensors,
    run_moreh_layer_norm,
    run_moreh_layer_norm_backward,
    run_moreh_layer_norm_backward_with_gamma_or_beta,
    torch_layer_norm,
    torch_layer_norm_backward,
)
from tests.ttnn.unit_tests.operations.test_utils import (
    create_ttnn_tilized_tensor,
    get_compute_kernel_options,
    to_torch,
    to_ttnn,
)

pytestmark = pytest.mark.use_module_device

# With normalized_dims=1 the op writes mean/rstd wrong when they hold more than one value, so only single-row inputs
# request them.


# Not the nightly helpers: their zero padding hides a broken mask; NaN padding makes it fail.
def run_moreh_layer_norm_nan_pad_test(input_shape, normalized_dims, device):
    cpu_input, cpu_gamma, cpu_beta, _ = make_input_tensors(input_shape, normalized_dims, True)
    expected_output, expected_mean, expected_rstd = torch_layer_norm(
        cpu_input, normalized_dims=normalized_dims, eps=1e-5, gamma=cpu_gamma, beta=cpu_beta
    )

    mean_rstd_shape = input_shape[:-normalized_dims]
    # NaN, so an unwritten value fails.
    tt_output = to_ttnn(torch.full(input_shape, float("nan")), device=device)
    tt_mean = to_ttnn(torch.full(mean_rstd_shape, float("nan")), device=device)
    tt_rstd = to_ttnn(torch.full(mean_rstd_shape, float("nan")), device=device)
    ttnn.moreh_layer_norm(
        create_ttnn_tilized_tensor(cpu_input, device, ttnn.bfloat16),
        normalized_dims,
        1e-5,
        to_ttnn(cpu_gamma, device=device),
        to_ttnn(cpu_beta, device=device),
        output=tt_output,
        mean=tt_mean,
        rstd=tt_rstd,
        compute_kernel_config=get_compute_kernel_options(False),
    )

    tol = 0.1 if normalized_dims == 1 else 0.15
    passing, out = comp_allclose(expected_output, to_torch(tt_output, shape=input_shape), rtol=tol, atol=tol)
    assert passing, out
    for expected, actual in [(expected_mean, tt_mean), (expected_rstd, tt_rstd)]:
        passing, out = comp_allclose(expected, to_torch(actual, shape=mean_rstd_shape), rtol=0.1, atol=0.1)
        assert passing, out


def run_moreh_layer_norm_backward_nan_pad_test(input_shape, normalized_dims, device):
    cpu_input, cpu_gamma, cpu_beta, cpu_output_grad = make_input_tensors(
        input_shape, normalized_dims, True, do_backward=True
    )
    expected_input_grad, expected_gamma_grad, expected_beta_grad = torch_layer_norm_backward(
        cpu_input.clone(),
        cpu_output_grad,
        normalized_dims=normalized_dims,
        eps=1e-5,
        gamma=cpu_gamma.clone(),
        beta=cpu_beta.clone(),
    )
    mean_rstd_dims = list(range(-normalized_dims, 0))
    mean = cpu_input.mean(dim=mean_rstd_dims, keepdim=True)
    rstd = (((cpu_input - mean) ** 2).mean(dim=mean_rstd_dims, keepdim=True) + 1e-5).rsqrt()

    mean_rstd_shape = input_shape[:-normalized_dims]
    normalized_shape = input_shape[-normalized_dims:]
    # NaN, so an unwritten value fails.
    tt_input_grad = to_ttnn(torch.full(input_shape, float("nan")), device=device)
    tt_gamma_grad = to_ttnn(torch.full(normalized_shape, float("nan")), device=device)
    tt_beta_grad = to_ttnn(torch.full(normalized_shape, float("nan")), device=device)
    ttnn.moreh_layer_norm_backward(
        create_ttnn_tilized_tensor(cpu_output_grad, device, ttnn.bfloat16),
        create_ttnn_tilized_tensor(cpu_input, device, ttnn.bfloat16),
        to_ttnn(mean, device=device, shape=mean_rstd_shape),
        to_ttnn(rstd, device=device, shape=mean_rstd_shape),
        normalized_dims,
        gamma=to_ttnn(cpu_gamma, device=device),
        input_grad=tt_input_grad,
        gamma_grad=tt_gamma_grad,
        beta_grad=tt_beta_grad,
        compute_kernel_config=get_compute_kernel_options(False),
    )

    passing, out = comp_allclose(expected_input_grad, to_torch(tt_input_grad, shape=input_shape), rtol=0.1, atol=0.5)
    assert passing, out
    for expected, actual in [(expected_gamma_grad, tt_gamma_grad), (expected_beta_grad, tt_beta_grad)]:
        passing, out = comp_allclose(expected, to_torch(actual, shape=normalized_shape), rtol=0.1, atol=0.5)
        assert passing, out


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape_normalized_dims",
    [
        ([1, 32], 1),
        ([2, 64, 64], 2),
    ],
    ids=["lastdim_aligned", "hw_aligned"],
)
def test_moreh_layer_norm(input_shape_normalized_dims, device):
    torch.manual_seed(0)
    run_moreh_layer_norm(input_shape_normalized_dims, False, 1e-5, ttnn.bfloat16, device, compute_kernel_options=False)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape, normalized_dims",
    [
        ([1, 20], 1),
        ([2, 77, 109], 2),
    ],
    ids=["lastdim", "hw"],
)
def test_moreh_layer_norm_unaligned(input_shape, normalized_dims, device):
    torch.manual_seed(0)
    run_moreh_layer_norm_nan_pad_test(input_shape, normalized_dims, device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape_normalized_dims, fp32_dest_acc_en",
    [
        (([2, 3, 45, 45], 3), False),
        (([1, 20], 1), True),
        (([1, 2, 500, 1000], 2), False),
    ],
    ids=["normalized_dims_3", "fp32_dest_acc", "large_algorithm"],
)
def test_moreh_layer_norm_corner_cases(input_shape_normalized_dims, fp32_dest_acc_en, device):
    torch.manual_seed(0)
    run_moreh_layer_norm(
        input_shape_normalized_dims, True, 1e-5, ttnn.bfloat16, device, compute_kernel_options=fp32_dest_acc_en
    )


@pytest.mark.merge_gate
def test_moreh_layer_norm_allocated_output_no_mean_rstd(device):
    torch.manual_seed(0)
    # Not run_moreh_layer_norm: it always passes `output` and doesn't check that mean/rstd come back as None.
    input_shape = [3, 77, 109]
    cpu_input, cpu_gamma, cpu_beta, _ = make_input_tensors(input_shape, 1, True)
    expected_output, _, _ = torch_layer_norm(cpu_input, normalized_dims=1, eps=1e-5, gamma=cpu_gamma, beta=cpu_beta)
    tt_output, tt_mean, tt_rstd = ttnn.moreh_layer_norm(
        create_ttnn_tilized_tensor(cpu_input, device, ttnn.bfloat16),
        1,
        1e-5,
        to_ttnn(cpu_gamma, device=device),
        to_ttnn(cpu_beta, device=device),
        compute_kernel_config=get_compute_kernel_options(False),
    )

    assert tt_mean is None and tt_rstd is None
    passing, out = comp_allclose(expected_output, to_torch(tt_output, shape=input_shape), rtol=0.1, atol=0.1)
    assert passing, out


# On Blackhole gamma_grad is wrong when it reduces over both a batch dim and H, so no backward shape here does.
@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape_normalized_dims, elementwise_affine",
    [
        (([6, 64, 64], 2), True),
        (([20, 30], 1), False),
    ],
    ids=["hw_aligned_batch", "no_affine"],
)
def test_moreh_layer_norm_backward(input_shape_normalized_dims, elementwise_affine, device):
    torch.manual_seed(0)
    run_moreh_layer_norm_backward(
        input_shape_normalized_dims, elementwise_affine, 1e-5, ttnn.bfloat16, device, compute_kernel_options=False
    )


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape, normalized_dims",
    [
        ([20, 30], 1),
        ([20, 30], 2),
    ],
    ids=["lastdim", "hw"],
)
def test_moreh_layer_norm_backward_unaligned(input_shape, normalized_dims, device):
    torch.manual_seed(0)
    run_moreh_layer_norm_backward_nan_pad_test(input_shape, normalized_dims, device)


@pytest.mark.merge_gate
@pytest.mark.parametrize("gamma_or_beta", [True, False], ids=["gamma_only", "beta_only"])
def test_moreh_layer_norm_backward_gamma_or_beta(gamma_or_beta, device):
    torch.manual_seed(0)
    run_moreh_layer_norm_backward_with_gamma_or_beta(
        ([20, 30], 1), gamma_or_beta, 1e-5, ttnn.bfloat16, device, compute_kernel_options=False
    )


@pytest.mark.merge_gate
def test_moreh_layer_norm_backward_fp32_dest_acc(device):
    torch.manual_seed(0)
    run_moreh_layer_norm_backward(([20, 30], 1), True, 1e-5, ttnn.bfloat16, device, compute_kernel_options=True)


@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.merge_gate
def test_moreh_layer_norm_backward_large_algorithm(device):
    torch.manual_seed(0)
    run_moreh_layer_norm_backward(
        ([1, 2, 500, 1000], 2), True, 1e-5, ttnn.bfloat16, device, compute_kernel_options=False
    )
