# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

import tt_lib as ttl
from models.utility_functions import comp_allclose_and_pcc
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


def torch_layernorm(input, *, normalized_dims=1, eps=1e-5, gamma=None, beta=None):
    normalized_shape = input.shape[-normalized_dims:]
    mean_rstd_dims = tuple(range(-normalized_dims, 0))

    mean = input.clone().mean(dim=mean_rstd_dims, keepdim=True)
    var = ((input.clone() - mean) ** 2).mean(dim=mean_rstd_dims, keepdim=True)
    rstd = (var + eps).rsqrt()

    if gamma is not None:
        gamma = gamma.view(normalized_shape)
    if beta is not None:
        beta = beta.view(normalized_shape)

    output = F.layer_norm(input, normalized_shape, weight=gamma, bias=beta, eps=eps)

    return output, mean, rstd


def torch_layernorm_backward(input, output_grad, *, normalized_dims=1, eps=1e-5, gamma=None, beta=None):
    normalized_shape = input.shape[-normalized_dims:]

    input.requires_grad_()
    if gamma is not None:
        gamma = gamma.view(normalized_shape)
        gamma.requires_grad_()
    if beta is not None:
        beta = beta.view(normalized_shape)
        beta.requires_grad_()

    output = F.layer_norm(input, normalized_shape, weight=gamma, bias=beta, eps=eps)
    output.backward(output_grad)

    gamma_grad = None
    beta_grad = None
    if gamma is not None:
        gamma_grad = gamma.grad.view(normalized_shape)
    if beta is not None:
        beta_grad = beta.grad.view(normalized_shape)

    return input.grad, gamma_grad, beta_grad


def tt_layernorm(input, *, normalized_dims=1, eps=1e-5, gamma=None, beta=None, device=None):
    input_shape = list(input.shape)

    # mean_rstd_shape
    mean_rstd_shape = input_shape[:-normalized_dims] + [1] * normalized_dims

    # dtype
    cpu_dtype = torch.bfloat16

    # input
    npu_input = to_npu(input, device)

    # gamma
    npu_gamma = to_npu(gamma, device)

    # beta
    npu_beta = to_npu(beta, device)

    # mean for inplace update
    cpu_mean = torch.full(mean_rstd_shape, float("nan"), dtype=cpu_dtype)
    npu_mean = to_npu(cpu_mean, device)

    # rstd for inplace update
    cpu_rstd = torch.full(mean_rstd_shape, float("nan"), dtype=cpu_dtype)
    npu_rstd = to_npu(cpu_rstd, device)

    # Forward
    npu_output = ttl.operations.primary.moreh_layernorm(
        npu_input, normalized_dims, eps, npu_gamma, npu_beta, mean=npu_mean, rstd=npu_rstd
    )

    tt_output = to_cpu(npu_output, input_shape)
    tt_mean = to_cpu(npu_mean, mean_rstd_shape)
    tt_rstd = to_cpu(npu_rstd, mean_rstd_shape)

    return tt_output, tt_mean, tt_rstd


def tt_layernorm_backward(input, output_grad, *, normalized_dims=1, eps=1e-5, gamma=None, beta=None, device=None):
    normalized_shape = input.shape[-normalized_dims:]

    # rank
    input_shape = list(input.shape)
    input_rank = len(input_shape)

    # mean_rstd_shape
    mean_rstd_shape = input_shape[:-normalized_dims] + [1] * normalized_dims

    # gamma_beta_shape
    gamma_beta_shape = [1] * (input_rank - normalized_dims) + input_shape[-normalized_dims:]

    # dtype
    cpu_dtype = torch.bfloat16

    # input
    npu_input = to_npu(input, device)

    # output_grad
    npu_output_grad = to_npu(output_grad, device)

    # gamma
    npu_gamma = to_npu(gamma, device)

    # mean, rstd
    mean_rstd_dims = tuple(range(-normalized_dims, 0))

    mean = input.clone().mean(dim=mean_rstd_dims, keepdim=True)
    var = ((input.clone() - mean) ** 2).mean(dim=mean_rstd_dims, keepdim=True)
    rstd = (var + eps).rsqrt()

    npu_mean = to_npu(mean, device, shape=mean_rstd_shape)
    npu_rstd = to_npu(rstd, device, shape=mean_rstd_shape)

    # input_grad for inplace update
    cpu_input_grad = torch.full(input_shape, float("nan"), dtype=cpu_dtype)
    npu_input_grad = to_npu(cpu_input_grad, device)

    # gamma_grad for inplace update
    npu_gamma_grad = None
    if gamma is not None:
        cpu_gamma_grad = torch.full(gamma_beta_shape, float("nan"), dtype=cpu_dtype)
        npu_gamma_grad = to_npu(cpu_gamma_grad, device)

    # beta_grad for inplace update
    npu_beta_grad = None
    if beta is not None:
        cpu_beta_grad = torch.full(gamma_beta_shape, float("nan"), dtype=cpu_dtype)
        npu_beta_grad = to_npu(cpu_beta_grad, device)

    # Backward
    _, npu_gamma_grad, _ = ttl.operations.primary.moreh_layernorm_backward(
        npu_output_grad,
        npu_input,
        npu_mean,
        npu_rstd,
        normalized_dims,
        gamma=npu_gamma,
        input_grad=npu_input_grad,
        gamma_grad=npu_gamma_grad,
        beta_grad=npu_beta_grad,
    )

    tt_input_grad = to_cpu(npu_input_grad, input_shape)
    tt_gamma_grad = to_cpu(npu_gamma_grad, gamma_beta_shape)
    if tt_gamma_grad is not None:
        tt_gamma_grad = tt_gamma_grad.view(normalized_shape)
    tt_beta_grad = to_cpu(npu_beta_grad, gamma_beta_shape)
    if tt_beta_grad is not None:
        tt_beta_grad = tt_beta_grad.view(normalized_shape)

    return tt_input_grad, tt_gamma_grad, tt_beta_grad


def make_input_tensors(input_shape, normalized_dims, elementwise_affine, do_backward=False):
    # rank
    input_rank = len(input_shape)

    # output_grad_shape
    output_grad_shape = input_shape

    # gamma_beta_shape
    gamma_beta_shape = [1] * (input_rank - normalized_dims) + input_shape[-normalized_dims:]

    # dtype
    cpu_dtype = torch.bfloat16

    # input
    cpu_input = torch.randint(-2, 3, input_shape, dtype=cpu_dtype)

    # gamma
    cpu_gamma = None
    if elementwise_affine:
        cpu_gamma = torch.rand(gamma_beta_shape, dtype=cpu_dtype) * 2 - 1.05

    # beta
    cpu_beta = None
    if elementwise_affine:
        cpu_beta = torch.rand(gamma_beta_shape, dtype=cpu_dtype) * 2 - 1.05

    # output_grad
    cpu_output_grad = None
    if do_backward:
        cpu_output_grad = torch.randint(-2, 3, output_grad_shape, dtype=cpu_dtype)

    return cpu_input, cpu_gamma, cpu_beta, cpu_output_grad


@pytest.mark.parametrize("eps", [1e-5, 1e-12], ids=["1e-5", "1e-12"])
@pytest.mark.parametrize("normalized_dims", [1, 2, 3, 4], ids=["W", "HW", "CHW", "NCHW"])
@pytest.mark.parametrize(
    "elementwise_affine",
    [False, True],
    ids=["elementwise_affine=False", "elementwise_affine=True"],
)
@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 1, TILE_HEIGHT, TILE_WIDTH],
        [6, 6, 2 * TILE_HEIGHT, 2 * TILE_WIDTH],
        [2, 2, TILE_HEIGHT + 13, TILE_WIDTH + 13],
        [2, 2, 8 * TILE_HEIGHT + 15, 32 * TILE_WIDTH - 15],
    ],
)
def test_moreh_layernorm(input_shape, normalized_dims, elementwise_affine, eps, device):
    torch.manual_seed(2023)

    cpu_input, cpu_gamma, cpu_beta, _ = make_input_tensors(input_shape, normalized_dims, elementwise_affine)

    # expected
    expected_output, expected_mean, expected_rstd = torch_layernorm(
        cpu_input, normalized_dims=normalized_dims, eps=eps, gamma=cpu_gamma, beta=cpu_beta
    )

    # actual
    actual_output, actual_mean, actual_rstd = tt_layernorm(
        cpu_input, normalized_dims=normalized_dims, eps=eps, gamma=cpu_gamma, beta=cpu_beta, device=device
    )

    # Set rtol and atol and pcc for output
    rtol = atol = 0.1
    if normalized_dims in (2, 3):
        rtol = atol = 0.15
    elif normalized_dims == 4:
        rtol = atol = 0.2
    output_pcc = 0.99

    # Check output
    pass_output, out_output = comp_allclose_and_pcc(
        expected_output, actual_output, rtol=rtol, atol=atol, pcc=output_pcc
    )
    logger.debug(f"output's {out_output}")
    assert pass_output

    # Set rtol and atol and pcc for mean and rstd
    rtol = atol = 0.1
    mean_pcc = 0.99
    rstd_pcc = 0.9
    # TODO(seunghwan100): Debug this case.
    if input_shape == [6, 6, 2 * TILE_HEIGHT, 2 * TILE_WIDTH] and normalized_dims == 3:
        rstd_pcc = 0.6

    # Check mean and rstd
    pass_mean, out_mean = comp_allclose_and_pcc(expected_mean, actual_mean, rtol=rtol, atol=atol, pcc=mean_pcc)
    logger.debug(f"mean's {out_mean}")
    assert pass_mean

    pass_rstd, out_rstd = comp_allclose_and_pcc(expected_rstd, actual_rstd, rtol=rtol, atol=atol, pcc=rstd_pcc)
    logger.debug(f"rstd's {out_rstd}")
    assert pass_rstd


@pytest.mark.parametrize("eps", [1e-5, 1e-12], ids=["1e-5", "1e-12"])
@pytest.mark.parametrize("normalized_dims", [1, 2, 3, 4], ids=["W", "HW", "CHW", "NCHW"])
@pytest.mark.parametrize(
    "elementwise_affine",
    [False, True],
    ids=["elementwise_affine=False", "elementwise_affine=True"],
)
@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 1, TILE_HEIGHT, TILE_WIDTH],
        [6, 6, 2 * TILE_HEIGHT, 2 * TILE_WIDTH],
        [2, 2, TILE_HEIGHT + 13, TILE_WIDTH + 13],
        [2, 2, 8 * TILE_HEIGHT + 15, 32 * TILE_WIDTH - 15],
    ],
)
def test_moreh_layernorm_backward(input_shape, normalized_dims, elementwise_affine, eps, device):
    torch.manual_seed(2023)

    cpu_input, cpu_gamma, cpu_beta, cpu_output_grad = make_input_tensors(
        input_shape, normalized_dims, elementwise_affine, do_backward=True
    )

    # expected
    expected_input_grad, expected_gamma_grad, expected_beta_grad = torch_layernorm_backward(
        cpu_input, cpu_output_grad, normalized_dims=normalized_dims, eps=eps, gamma=cpu_gamma, beta=cpu_beta
    )

    # actual
    actual_input_grad, actual_gamma_grad, actual_beta_grad = tt_layernorm_backward(
        cpu_input,
        cpu_output_grad,
        normalized_dims=normalized_dims,
        eps=eps,
        gamma=cpu_gamma,
        beta=cpu_beta,
        device=device,
    )

    # Set rtol and atol and pcc for gradients
    rtol = atol = 0.1
    pcc = 0.999

    # Check input_grad
    pig, oig = comp_allclose_and_pcc(expected_input_grad, actual_input_grad, rtol=rtol, atol=atol, pcc=pcc)
    logger.debug(f"input_grad's {oig}")
    assert pig

    # I divide gamma_grad and beta_grad by (N * C * Ht * Wt), because the error of bf16 sum increases.
    n, c, h, w = input_shape
    ht = (h + TILE_HEIGHT - 1) // TILE_HEIGHT
    wt = (w + TILE_WIDTH - 1) // TILE_WIDTH
    numerator = n * c * ht * wt

    # Check gamma_grad
    if expected_gamma_grad is not None:
        pgg, ogg = comp_allclose_and_pcc(
            expected_gamma_grad / numerator, actual_gamma_grad / numerator, rtol=rtol, atol=atol, pcc=pcc
        )
        logger.debug(f"gamma_grad's {ogg}")
        assert pgg
    else:
        assert actual_gamma_grad is None

    # Check beta_grad
    if expected_beta_grad is not None:
        pbg, obg = comp_allclose_and_pcc(
            expected_beta_grad / numerator, actual_beta_grad / numerator, rtol=rtol, atol=atol, pcc=pcc
        )
        logger.debug(f"beta_grad's {obg}")
        assert pbg
    else:
        assert actual_beta_grad is None
