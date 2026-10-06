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

EPS = 1e-5


def run_moreh_layer_norm_test(
    input_shape, normalized_dims, device, elementwise_affine=True, create_mean_rstd=True, fp32_dest_acc_en=False
):
    normalized_shape = input_shape[-normalized_dims:]
    mean_rstd_shape = input_shape[:-normalized_dims]
    mean_rstd_dims = list(range(-normalized_dims, 0))

    torch_input = torch.randint(-2, 3, input_shape, dtype=torch.bfloat16)
    torch_gamma = torch.rand(normalized_shape, dtype=torch.bfloat16) * 2 - 1.05 if elementwise_affine else None
    torch_beta = torch.rand(normalized_shape, dtype=torch.bfloat16) * 2 - 1.05 if elementwise_affine else None
    torch_output = F.layer_norm(torch_input, normalized_shape, weight=torch_gamma, bias=torch_beta, eps=EPS)
    torch_mean = torch_input.mean(dim=mean_rstd_dims, keepdim=True)
    torch_rstd = (((torch_input - torch_mean) ** 2).mean(dim=mean_rstd_dims, keepdim=True) + EPS).rsqrt()

    tt_mean = tt_rstd = None
    if create_mean_rstd:
        tt_mean = to_ttnn(torch.full(mean_rstd_shape, float("nan"), dtype=torch.bfloat16), device=device)
        tt_rstd = to_ttnn(torch.full(mean_rstd_shape, float("nan"), dtype=torch.bfloat16), device=device)
    tt_output, tt_mean, tt_rstd = ttnn.moreh_layer_norm(
        to_ttnn(torch_input, device=device),
        normalized_dims,
        EPS,
        to_ttnn(torch_gamma, device=device),
        to_ttnn(torch_beta, device=device),
        mean=tt_mean,
        rstd=tt_rstd,
        compute_kernel_config=get_compute_kernel_options(fp32_dest_acc_en),
    )

    # As in nightly, normalizing over more dims accumulates more bf16 error in the output.
    tolerance = 0.1 if normalized_dims == 1 else 0.15
    passing, out = comp_allclose(torch_output, to_torch(tt_output), rtol=tolerance, atol=tolerance)
    assert passing, out
    if not create_mean_rstd:
        assert tt_mean is None and tt_rstd is None
        return
    passing, out = comp_allclose(torch_mean, to_torch(tt_mean, shape=torch_mean.shape), rtol=0.1, atol=0.1)
    assert passing, out
    passing, out = comp_allclose(torch_rstd, to_torch(tt_rstd, shape=torch_rstd.shape), rtol=0.1, atol=0.1)
    assert passing, out


def run_moreh_layer_norm_backward_test(input_shape, normalized_dims, device, has_gamma=True, has_beta=True):
    normalized_shape = input_shape[-normalized_dims:]
    mean_rstd_shape = input_shape[:-normalized_dims]
    mean_rstd_dims = list(range(-normalized_dims, 0))

    torch_input = torch.randint(-2, 3, input_shape, dtype=torch.bfloat16, requires_grad=True)
    torch_gamma = (torch.rand(normalized_shape, dtype=torch.bfloat16) * 2 - 1.05).requires_grad_()
    torch_beta = (torch.rand(normalized_shape, dtype=torch.bfloat16) * 2 - 1.05).requires_grad_()
    torch_gamma = torch_gamma if has_gamma else None
    torch_beta = torch_beta if has_beta else None
    torch_output_grad = torch.randint(-2, 3, input_shape, dtype=torch.bfloat16)
    torch_output = F.layer_norm(torch_input, normalized_shape, weight=torch_gamma, bias=torch_beta, eps=EPS)
    torch_output.backward(torch_output_grad)
    torch_mean = torch_input.detach().mean(dim=mean_rstd_dims, keepdim=True)
    torch_rstd = (((torch_input.detach() - torch_mean) ** 2).mean(dim=mean_rstd_dims, keepdim=True) + EPS).rsqrt()

    def nan_tensor(shape, enabled=True):
        return to_ttnn(torch.full(shape, float("nan"), dtype=torch.bfloat16), device=device) if enabled else None

    tt_input_grad, tt_gamma_grad, tt_beta_grad = ttnn.moreh_layer_norm_backward(
        to_ttnn(torch_output_grad, device=device),
        to_ttnn(torch_input.detach(), device=device),
        to_ttnn(torch_mean, device=device, shape=mean_rstd_shape),
        to_ttnn(torch_rstd, device=device, shape=mean_rstd_shape),
        normalized_dims,
        gamma=to_ttnn(torch_gamma.detach(), device=device) if has_gamma else None,
        input_grad=nan_tensor(input_shape),
        gamma_grad=nan_tensor(normalized_shape, has_gamma),
        beta_grad=nan_tensor(normalized_shape, has_beta),
        compute_kernel_config=get_compute_kernel_options(False),
    )

    passing, out = comp_allclose(torch_input.grad, to_torch(tt_input_grad), rtol=0.1, atol=0.5)
    assert passing, out
    if has_gamma:
        passing, out = comp_allclose(torch_gamma.grad, to_torch(tt_gamma_grad), rtol=0.1, atol=0.5)
        assert passing, out
    else:
        assert tt_gamma_grad is None
    if has_beta:
        passing, out = comp_allclose(torch_beta.grad, to_torch(tt_beta_grad), rtol=0.1, atol=0.5)
        assert passing, out
    else:
        assert tt_beta_grad is None


@pytest.mark.merge_gate
def test_moreh_layer_norm(device):
    torch.manual_seed(0)
    run_moreh_layer_norm_test([1, 20], 1, device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape, normalized_dims, elementwise_affine",
    [
        ([1, 20], 1, False),
        ([2, 2 * TILE_HEIGHT + 13, 3 * TILE_WIDTH + 13], 2, True),
        ([2, 3, TILE_HEIGHT + 13, TILE_WIDTH + 13], 3, True),
    ],
    ids=["no_affine", "normalized_dims_2", "normalized_dims_3"],
)
def test_moreh_layer_norm_corner_cases(input_shape, normalized_dims, elementwise_affine, device):
    torch.manual_seed(0)
    run_moreh_layer_norm_test(input_shape, normalized_dims, device, elementwise_affine=elementwise_affine)


# lastdim_multi_tile leaves out mean/rstd: with normalized_dims=1 the op writes them wrong whenever they hold more
# than one value per row (only every 16th value is correct).
@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape",
    [[1, 20], [3, 2 * TILE_HEIGHT + 13, 3 * TILE_WIDTH + 13]],
    ids=["single_tile", "lastdim_multi_tile"],
)
def test_moreh_layer_norm_no_mean_rstd(input_shape, device):
    torch.manual_seed(0)
    run_moreh_layer_norm_test(input_shape, 1, device, create_mean_rstd=False)


@pytest.mark.merge_gate
def test_moreh_layer_norm_fp32_dest_acc(device):
    torch.manual_seed(0)
    # fp32_dest_acc_en switches the intermediate buffers to Float32.
    run_moreh_layer_norm_test([1, 20], 1, device, fp32_dest_acc_en=True)


@pytest.mark.merge_gate
def test_moreh_layer_norm_program_cache(device):
    torch.manual_seed(0)
    # Start from an empty cache: the module-scoped device carries entries over from earlier tests.
    device.clear_program_cache()
    run_moreh_layer_norm_test([1, 20], 1, device)
    num_program_cache_entries = device.num_program_cache_entries()
    # Holding this tensor moves the next allocations, so the cache hit must update the buffer addresses.
    tt_placeholder = to_ttnn(torch.zeros([1, 20]), device=device)
    run_moreh_layer_norm_test([1, 20], 1, device)
    assert device.num_program_cache_entries() == num_program_cache_entries


@pytest.mark.merge_gate
def test_moreh_layer_norm_backward(device):
    torch.manual_seed(0)
    # Passing input_grad plus gamma_grad/beta_grad runs both the input-grad and the gamma/beta-grad factories.
    # Not ([2, 20, 30], 1): gamma_grad is wrong on Blackhole when it reduces over both a batch dim and H.
    run_moreh_layer_norm_backward_test([6, 2 * TILE_HEIGHT, 2 * TILE_WIDTH], 2, device)


# Every shape reduces gamma_grad/beta_grad over H only or over nothing, to avoid the Blackhole bug above.
@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape, normalized_dims, has_gamma, has_beta",
    [
        ([20, 30], 1, True, True),
        ([20, 30], 2, True, True),
        ([20, 30], 1, True, False),
        ([20, 30], 1, False, True),
        ([20, 30], 1, False, False),
    ],
    ids=["lastdim_unaligned", "hw_unaligned", "gamma_only", "beta_only", "no_affine"],
)
def test_moreh_layer_norm_backward_corner_cases(input_shape, normalized_dims, has_gamma, has_beta, device):
    torch.manual_seed(0)
    run_moreh_layer_norm_backward_test(input_shape, normalized_dims, device, has_gamma=has_gamma, has_beta=has_beta)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "mean_shape",
    [[2, 33], [1, 64]],
    ids=["wrong_volume", "same_volume_wrong_shape"],
)
def test_moreh_layer_norm_backward_rejects_wrong_mean_shape(mean_shape, device, expect_error):
    torch.manual_seed(0)
    input_shape = [2, TILE_HEIGHT, 2 * TILE_WIDTH]
    tt_input = to_ttnn(torch.randint(-2, 3, input_shape, dtype=torch.bfloat16), device=device)

    with expect_error(RuntimeError, "mean must have logical shape"):
        ttnn.moreh_layer_norm_backward(
            to_ttnn(torch.randint(-2, 3, input_shape, dtype=torch.bfloat16), device=device),
            tt_input,
            to_ttnn(torch.zeros(mean_shape, dtype=torch.bfloat16), device=device),
            to_ttnn(torch.ones(input_shape[:-1], dtype=torch.bfloat16), device=device),
            1,
            input_grad=to_ttnn(torch.zeros(input_shape, dtype=torch.bfloat16), device=device),
        )
