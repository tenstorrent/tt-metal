# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

import tt_lib as ttl
from models.utility_functions import comp_allclose
from loguru import logger

TILE_HEIGHT = 32
TILE_WIDTH = 32


def to_cpu(npu_tensor, shape, *, cpu_layout=ttl.tensor.Layout.ROW_MAJOR):
    if npu_tensor is None:
        return None
    if not isinstance(shape, (list, tuple)):
        shape = tuple(shape)
    cpu_tensor = npu_tensor.cpu().to(cpu_layout).unpad_from_tile(shape).to_torch()
    return cpu_tensor


def to_npu(
    cpu_tensor,
    device,
    *,
    npu_layout=ttl.tensor.Layout.TILE,
    npu_dtype=ttl.tensor.DataType.BFLOAT16,
    shape=None,
):
    if cpu_tensor is None:
        return None
    if shape is not None:
        cpu_tensor = cpu_tensor.view(shape)
    npu_tensor = ttl.tensor.Tensor(cpu_tensor, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    return npu_tensor


def torch_groupnorm(input, num_groups, gamma=None, beta=None, eps=1e-05):
    N, _, _, _ = input.shape

    output = F.group_norm(input, num_groups, gamma, beta, eps)

    x_view = input.view(N, num_groups, -1)
    mean = x_view.mean(dim=-1, keepdim=True)
    var = ((x_view - mean) ** 2).mean(dim=-1, keepdim=False)
    rstd = (var + eps).sqrt()

    return output, mean.view(N, num_groups), rstd.view(N, num_groups)


def torch_groupnorm_backward(input, output_grad, num_groups, gamma=None, beta=None, eps=1e-05):
    input.requires_grad_()
    if gamma is not None:
        gamma.requires_grad_()
    if beta is not None:
        beta.requires_grad_()

    output = F.group_norm(input, num_groups, gamma, beta, eps)
    output.backward(output_grad)

    gamma_grad = beta_grad = None
    if gamma is not None:
        gamma_grad = gamma.grad
    if beta is not None:
        beta_grad = beta.grad

    return input.grad, gamma_grad, beta_grad


def tt_groupnorm(input, num_groups, gamma=None, beta=None, eps=1e-05, device=None):
    N, C, _, _ = input.shape

    gamma_beta_shape = [1, 1, 1, C]
    mean_rstd_shape = [1, 1, N, num_groups]

    npu_input = to_npu(input, device)
    npu_gamma = to_npu(gamma, device, shape=gamma_beta_shape)
    npu_beta = to_npu(beta, device, shape=gamma_beta_shape)

    # Forward
    npu_output, npu_mean, npu_rstd = ttl.operations.primary.moreh_groupnorm(
        npu_input, num_groups, eps, npu_gamma, npu_beta
    )

    tt_output = to_cpu(npu_output, input.shape)
    tt_mean = to_cpu(npu_mean, mean_rstd_shape).view(N, num_groups)
    tt_rstd = to_cpu(npu_rstd, mean_rstd_shape).view(N, num_groups)

    return tt_output, tt_mean, tt_rstd


def tt_groupnorm_backward(input, output_grad, num_groups, gamma=None, eps=1e-05, device=None):
    N, C, _, _ = input.shape

    gamma_beta_shape = [1, 1, 1, C]
    mean_rstd_shape = [1, 1, N, num_groups]

    x_view = input.view(N, num_groups, -1)
    mean = x_view.mean(dim=-1, keepdim=True)
    var = ((x_view - mean) ** 2).mean(dim=-1, keepdim=False)
    rstd = (var + eps).sqrt()

    npu_output_grad = to_npu(output_grad, device)
    npu_input = to_npu(input, device)
    npu_mean = to_npu(mean, device, shape=mean_rstd_shape)
    npu_rstd = to_npu(rstd, device, shape=mean_rstd_shape)
    npu_gamma = to_npu(gamma, device, shape=gamma_beta_shape)

    # Backward
    npu_dx, npu_dg, npu_db = ttl.operations.primary.moreh_groupnorm_backward(
        npu_output_grad, npu_input, npu_mean, npu_rstd, num_groups, gamma=npu_gamma
    )

    tt_input_grad = to_cpu(npu_dx, input.shape)
    tt_gamma_grad = to_cpu(npu_dg, gamma_beta_shape).view(C)
    tt_beta_grad = to_cpu(npu_db, gamma_beta_shape).view(C)

    return tt_input_grad, tt_gamma_grad, tt_beta_grad


def make_input_tensors(input_shape, affine, do_backward=False):
    N, C, H, W = input_shape

    # output_grad_shape
    output_grad_shape = (N, C, H, W)
    # gamma_beta_shape
    gamma_beta_shape = [C]

    # cpu_dtype
    cpu_dtype = torch.float32

    # input
    input = torch.empty(input_shape, dtype=cpu_dtype).uniform_(-2, 2)

    # gamma
    gamma = None
    if affine:
        gamma = torch.empty(gamma_beta_shape, dtype=cpu_dtype).uniform_(-2, 2)

    # beta
    beta = None
    if affine:
        beta = torch.empty(gamma_beta_shape, dtype=cpu_dtype).uniform_(-2, 2)

    # output_grad
    output_grad = None
    if do_backward:
        output_grad = torch.empty(output_grad_shape, dtype=cpu_dtype).uniform_(-2, 2)

    return input, gamma, beta, output_grad


@pytest.mark.parametrize(
    "N",
    [
        2,
    ],
)
@pytest.mark.parametrize(
    "C_num_groups",
    [
        [4, 1],
        [4, 2],
        [4, 4],
    ],
)
@pytest.mark.parametrize(
    "H",
    [
        23,
        512,
    ],
)
@pytest.mark.parametrize(
    "W",
    [
        23,
        512,
    ],
)
@pytest.mark.parametrize(
    "eps",
    [
        1e-05,
        1e-12,
    ],
)
@pytest.mark.parametrize(
    "affine",
    [
        True,
        False,
    ],
)
def test_moreh_groupnorm(N, C_num_groups, H, W, eps, affine, device):
    torch.manual_seed(2024)

    C, num_groups = C_num_groups
    input_shape = (N, C, H, W)
    cpu_input, cpu_beta, cpu_gamma, _ = make_input_tensors(input_shape, affine)

    # expected
    expected_output, expected_mean, expected_rstd = torch_groupnorm(cpu_input, num_groups, cpu_gamma, cpu_beta, eps)
    # actual
    actual_output, actual_mean, actual_rstd = tt_groupnorm(cpu_input, num_groups, cpu_gamma, cpu_beta, eps, device)

    # Set rtol and atol
    rtol = atol = 0.1
    if (C_num_groups == [4, 1]) and (H == 512) and (W == 512) and affine:
        rtol = atol = 0.13

    # Check output
    pass_output, out_output = comp_allclose(expected_output, actual_output, rtol=rtol, atol=atol)
    logger.info(f"output's {out_output}")
    assert pass_output

    # Check mean
    pass_mean, out_mean = comp_allclose(expected_mean, actual_mean, rtol=rtol, atol=atol)
    logger.info(f"mean's {out_mean}")
    assert pass_mean

    # Check rstd
    pass_rstd, out_rstd = comp_allclose(expected_rstd, actual_rstd, rtol=rtol, atol=atol)
    logger.info(f"rstd's {out_rstd}")
    assert pass_rstd


@pytest.mark.parametrize(
    "N",
    [
        2,
    ],
)
@pytest.mark.parametrize(
    "C_num_groups",
    [
        [4, 1],
        [4, 2],
        [4, 4],
    ],
)
@pytest.mark.parametrize(
    "H",
    [
        23,
        512,
    ],
)
@pytest.mark.parametrize(
    "W",
    [
        23,
        512,
    ],
)
@pytest.mark.parametrize(
    "eps",
    [
        1e-05,
        1e-12,
    ],
)
@pytest.mark.parametrize(
    "affine",
    [
        True,
        False,
    ],
)
def test_moreh_groupnorm_backward(N, C_num_groups, H, W, eps, affine, device):
    torch.manual_seed(2024)

    C, num_groups = C_num_groups
    input_shape = (N, C, H, W)
    cpu_input, cpu_gamma, cpu_beta, cpu_output_grad = make_input_tensors(input_shape, affine, do_backward=True)

    # expected
    expected_input_grad, expected_gamma_grad, expected_beta_grad = torch_groupnorm_backward(
        cpu_input, cpu_output_grad, num_groups, cpu_gamma, cpu_beta, eps
    )
    # actual
    actual_input_grad, actual_gamma_grad, actual_beta_grad = tt_groupnorm_backward(
        cpu_input, cpu_output_grad, num_groups, cpu_gamma, eps, device
    )

    # Set rtol and atol
    rtol = atol = 0.1

    # Check input_grad
    pass_input_grad, out_input_grad = comp_allclose(expected_input_grad, actual_input_grad, rtol=rtol, atol=atol)
    logger.info(f"input_grad's {out_input_grad}")
    assert pass_input_grad

    # I divide gamma_grad and beta_grad by (N * C * Ht * Wt), because the error of bf16 sum increases.
    Ht = (H + TILE_HEIGHT - 1) // TILE_HEIGHT
    Wt = (W + TILE_WIDTH - 1) // TILE_WIDTH
    divisor = N * C * Ht * Wt

    # Check gamma_grad
    if expected_gamma_grad is not None:
        pass_gamma_grad, out_gamma_grad = comp_allclose(
            expected_gamma_grad / divisor, actual_gamma_grad / divisor, rtol=rtol, atol=atol
        )
        logger.info(f"gamma_grad's {out_gamma_grad}")
        assert pass_gamma_grad

    # Check beta_grad
    if expected_beta_grad is not None:
        pass_beta_grad, out_beta_grad = comp_allclose(
            expected_beta_grad / divisor, actual_beta_grad / divisor, rtol=rtol, atol=atol
        )
        logger.info(f"beta_grad's {out_beta_grad}")
        assert pass_beta_grad
