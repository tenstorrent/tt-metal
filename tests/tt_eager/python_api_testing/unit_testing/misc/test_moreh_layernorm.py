# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
import copy

import ttnn
from models.utility_functions import comp_allclose
from loguru import logger

from tests.tt_eager.python_api_testing.unit_testing.misc.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    TILE_HEIGHT,
    TILE_WIDTH,
)
from models.utility_functions import skip_for_grayskull


def to_cpu(npu_tensor, shape, *, cpu_layout=ttnn.ROW_MAJOR_LAYOUT):
    if npu_tensor is None:
        return None
    if not isinstance(shape, (list, tuple)):
        shape = tuple(shape)

    unpad_shape = copy.copy(shape)
    if shape == []:
        unpad_shape = [1, 1]
    if len(shape) == 1:
        unpad_shape = [1] + shape

    cpu_tensor = npu_tensor.cpu().to(cpu_layout).unpad_from_tile(unpad_shape).to_torch().reshape(shape)

    return cpu_tensor


def to_npu(
    cpu_tensor,
    device,
    *,
    npu_layout=ttnn.TILE_LAYOUT,
    npu_dtype=ttnn.bfloat16,
    shape=None,
):
    if cpu_tensor is None:
        return None
    if shape is not None:
        cpu_tensor = cpu_tensor.view(shape)

    if len(cpu_tensor.shape) == 1:
        cpu_tensor = cpu_tensor.reshape([1, len(cpu_tensor)])

    if len(cpu_tensor.shape) == 0:
        cpu_tensor = cpu_tensor.reshape([1, 1])

    npu_tensor = ttnn.Tensor(cpu_tensor, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    return npu_tensor


def torch_layernorm(input, *, normalized_dims=1, eps=1e-5, gamma=None, beta=None):
    normalized_shape = input.shape[-normalized_dims:]
    mean_rstd_dims = tuple(range(-normalized_dims, 0))

    mean = input.clone().mean(dim=mean_rstd_dims, keepdim=True)
    var = ((input.clone() - mean) ** 2).mean(dim=mean_rstd_dims)
    rstd = (var + eps).rsqrt()

    mean = torch.squeeze(mean, mean_rstd_dims)

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


def tt_layernorm(
    input,
    *,
    normalized_dims=1,
    eps=1e-5,
    gamma=None,
    beta=None,
    device=None,
    compute_kernel_config=None,
    create_mean_rstd=True,
):
    input_shape = list(input.shape)

    # mean_rstd_shape
    mean_rstd_shape = input_shape[:-normalized_dims]

    # dtype
    cpu_dtype = torch.bfloat16

    # input
    npu_input = to_npu(input, device)

    # output
    output = torch.empty_like(input)
    npu_output = to_npu(output, device)

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
    npu_output, npu_mean, npu_rstd = ttnn.experimental.operations.primary.moreh_layernorm(
        npu_input,
        normalized_dims,
        eps,
        npu_gamma,
        npu_beta,
        output=npu_output,
        mean=npu_mean if create_mean_rstd else None,
        rstd=npu_rstd if create_mean_rstd else None,
        compute_kernel_config=compute_kernel_config,
    )

    tt_output = to_cpu(npu_output, input_shape)
    tt_mean = to_cpu(npu_mean, mean_rstd_shape) if create_mean_rstd else None
    tt_rstd = to_cpu(npu_rstd, mean_rstd_shape) if create_mean_rstd else None

    return tt_output, tt_mean, tt_rstd


def tt_layernorm_backward(
    input, output_grad, *, normalized_dims=1, eps=1e-5, gamma=None, beta=None, device=None, compute_kernel_config=None
):
    normalized_shape = input.shape[-normalized_dims:]

    # rank
    input_shape = list(input.shape)
    input_rank = len(input_shape)

    # mean_rstd_shape
    mean_rstd_shape = input_shape[:-normalized_dims]

    # gamma_beta_shape
    gamma_beta_shape = input_shape[-normalized_dims:]

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
    _, npu_gamma_grad, _ = ttnn.experimental.operations.primary.moreh_layernorm_backward(
        npu_output_grad,
        npu_input,
        npu_mean,
        npu_rstd,
        normalized_dims,
        gamma=npu_gamma,
        input_grad=npu_input_grad,
        gamma_grad=npu_gamma_grad,
        beta_grad=npu_beta_grad,
        compute_kernel_config=compute_kernel_config,
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
    # output_grad_shape
    output_grad_shape = input_shape

    # gamma_beta_shape
    gamma_beta_shape = input_shape[-normalized_dims:]

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


def run_moreh_layernorm(
    input_shape_normalized_dims, elementwise_affine, eps, device, create_mean_rstd=True, compute_kernel_options=None
):
    input_shape, normalized_dims = input_shape_normalized_dims

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    cpu_input, cpu_gamma, cpu_beta, _ = make_input_tensors(input_shape, normalized_dims, elementwise_affine)

    # expected
    expected_output, expected_mean, expected_rstd = torch_layernorm(
        cpu_input, normalized_dims=normalized_dims, eps=eps, gamma=cpu_gamma, beta=cpu_beta
    )

    # actual
    actual_output, actual_mean, actual_rstd = tt_layernorm(
        cpu_input,
        normalized_dims=normalized_dims,
        eps=eps,
        gamma=cpu_gamma,
        beta=cpu_beta,
        device=device,
        compute_kernel_config=compute_kernel_config,
        create_mean_rstd=create_mean_rstd,
    )

    # Set rtol and atol and pcc for output
    rtol = atol = 0.1
    if normalized_dims in (2, 3):
        rtol = atol = 0.15
    elif normalized_dims == 4:
        rtol = atol = 0.2

    # Check output
    pass_output, out_output = comp_allclose(expected_output, actual_output, rtol=rtol, atol=atol)
    logger.debug(f"output's {out_output}")
    assert pass_output

    # Set rtol and atol and pcc for mean and rstd
    rtol = atol = 0.1

    # Check mean and rstd
    if create_mean_rstd:
        pass_mean, out_mean = comp_allclose(expected_mean, actual_mean, rtol=rtol, atol=atol)

        logger.debug(f"mean's {out_mean}")
        assert pass_mean

        pass_rstd, out_rstd = comp_allclose(expected_rstd, actual_rstd, rtol=rtol, atol=atol)

        logger.debug(f"rstd's {out_rstd}")
        assert pass_rstd


def run_moreh_layernorm_backward(
    input_shape_normalized_dims, elementwise_affine, eps, device, compute_kernel_options=None
):
    input_shape, normalized_dims = input_shape_normalized_dims

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

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
        compute_kernel_config=compute_kernel_config,
    )

    # Set rtol and atol and pcc for gradients
    rtol = 0.1
    atol = 0.5

    # Check input_grad
    pig, oig = comp_allclose(expected_input_grad, actual_input_grad, rtol=rtol, atol=atol)
    logger.debug(f"input_grad's {oig}")
    assert pig

    # Check gamma_grad
    if expected_gamma_grad is not None:
        pgg, ogg = comp_allclose(expected_gamma_grad, actual_gamma_grad, rtol=rtol, atol=atol)
        logger.debug(f"gamma_grad's {ogg}")
        assert pgg
    else:
        assert actual_gamma_grad is None

    # Check beta_grad
    if expected_beta_grad is not None:
        pbg, obg = comp_allclose(expected_beta_grad, actual_beta_grad, rtol=rtol, atol=atol)
        logger.debug(f"beta_grad's {obg}")
        assert pbg
    else:
        assert actual_beta_grad is None


@skip_for_grayskull("Using the transpose function in copy_tile causes a hang.")
@pytest.mark.parametrize("eps", [1e-5], ids=["1e-5"])
@pytest.mark.parametrize(
    "elementwise_affine",
    [False, True],
    ids=["elementwise_affine=False", "elementwise_affine=True"],
)
@pytest.mark.parametrize(
    "input_shape_normalized_dims",
    [
        ([1, 20], 1),  # test 2d
        ([10, 20], 2),  # test 2d
        ([3, 3, 4 * TILE_HEIGHT, 5 * TILE_WIDTH], 4),  # test 4d
        ([5, 2, 3, 4, 2 * TILE_HEIGHT + 13, 3 * TILE_WIDTH + 13], 4),  # test 6d
    ],
)
def test_moreh_layernorm(input_shape_normalized_dims, elementwise_affine, eps, device):
    torch.manual_seed(2023)
    run_moreh_layernorm(input_shape_normalized_dims, elementwise_affine, eps, device)


@skip_for_grayskull("Using the transpose function in copy_tile causes a hang.")
@pytest.mark.parametrize("eps", [1e-5], ids=["1e-5"])
@pytest.mark.parametrize(
    "elementwise_affine",
    [False, True],
    ids=["elementwise_affine=False", "elementwise_affine=True"],
)
@pytest.mark.parametrize(
    "input_shape_normalized_dims",
    [
        ([20, 30], 2),  # test 2d
        ([2, 20, 30], 1),  # test 3d
        ([6, 2 * TILE_HEIGHT, 2 * TILE_WIDTH], 2),  # test 3d
        ([5, 2, 3, 4, TILE_HEIGHT + 13, TILE_WIDTH + 13], 3),  # test 6d
    ],
)
def test_moreh_layernorm_backward(input_shape_normalized_dims, elementwise_affine, eps, device):
    torch.manual_seed(2023)
    run_moreh_layernorm_backward(input_shape_normalized_dims, elementwise_affine, eps, device)


@skip_for_grayskull("Using the transpose function in copy_tile causes a hang.")
@pytest.mark.parametrize("eps", [0.05], ids=["0.05"])
@pytest.mark.parametrize(
    "elementwise_affine",
    [False, True],
    ids=["elementwise_affine=False", "elementwise_affine=True"],
)
@pytest.mark.parametrize(
    "input_shape_normalized_dims",
    [
        ([TILE_HEIGHT - 15, TILE_WIDTH + 2], 2),
        ([4, 8, 3 * TILE_HEIGHT + 15, 4 * TILE_WIDTH - 15], 4),
    ],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_layernorm_compute_kernel_options(
    input_shape_normalized_dims, elementwise_affine, eps, compute_kernel_options, device
):
    torch.manual_seed(2023)
    run_moreh_layernorm(input_shape_normalized_dims, elementwise_affine, eps, device, compute_kernel_options)


@skip_for_grayskull("Using the transpose function in copy_tile causes a hang.")
@pytest.mark.parametrize("eps", [0.05], ids=["0.05"])
@pytest.mark.parametrize(
    "elementwise_affine",
    [False, True],
    ids=["elementwise_affine=False", "elementwise_affine=True"],
)
@pytest.mark.parametrize(
    "input_shape_normalized_dims",
    [
        ([TILE_HEIGHT, TILE_WIDTH], 2),  # test 2d
        ([6, 2 * TILE_HEIGHT, 2 * TILE_WIDTH], 2),  # test 3d
    ],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_layernorm_backward_compute_kernel_options(
    input_shape_normalized_dims, elementwise_affine, eps, compute_kernel_options, device
):
    torch.manual_seed(2023)
    run_moreh_layernorm_backward(input_shape_normalized_dims, elementwise_affine, eps, device, compute_kernel_options)


@skip_for_grayskull("Using the transpose function in copy_tile causes a hang.")
@pytest.mark.parametrize("eps", [0.05], ids=["0.05"])
@pytest.mark.parametrize(
    "elementwise_affine",
    [False, True],
    ids=["elementwise_affine=False", "elementwise_affine=True"],
)
@pytest.mark.parametrize(
    "input_shape_normalized_dims",
    [
        ([6, 2 * TILE_HEIGHT, 2 * TILE_WIDTH], 2),  # test 3d
    ],
)
def test_moreh_layernorm_callback(input_shape_normalized_dims, elementwise_affine, eps, device, use_program_cache):
    torch.manual_seed(2023)
    for _ in range(2):
        run_moreh_layernorm(input_shape_normalized_dims, elementwise_affine, eps, device)


@skip_for_grayskull("Using the transpose function in copy_tile causes a hang.")
@pytest.mark.parametrize("eps", [0.05], ids=["0.05"])
@pytest.mark.parametrize(
    "elementwise_affine",
    [False, True],
    ids=["elementwise_affine=False", "elementwise_affine=True"],
)
@pytest.mark.parametrize(
    "input_shape_normalized_dims",
    [
        ([6, 2 * TILE_HEIGHT, 2 * TILE_WIDTH], 2),  # test 3d
    ],
)
def test_moreh_layernorm_backward_callback(
    input_shape_normalized_dims, elementwise_affine, eps, device, use_program_cache
):
    torch.manual_seed(2023)
    for _ in range(2):
        run_moreh_layernorm_backward(input_shape_normalized_dims, elementwise_affine, eps, device)


@skip_for_grayskull("Using the transpose function in copy_tile causes a hang.")
@pytest.mark.parametrize("eps", [1e-5], ids=["1e-5"])
@pytest.mark.parametrize(
    "elementwise_affine",
    [False, True],
    ids=["elementwise_affine=False", "elementwise_affine=True"],
)
@pytest.mark.parametrize(
    "input_shape_normalized_dims",
    [
        ([1, 20], 1),  # test 2d
        ([5, 2, 3, 4, 2 * TILE_HEIGHT + 13, 3 * TILE_WIDTH + 13], 4),  # test 6d
    ],
)
def test_moreh_layernorm_no_mean_rstd(input_shape_normalized_dims, elementwise_affine, eps, device):
    torch.manual_seed(2023)
    run_moreh_layernorm(input_shape_normalized_dims, elementwise_affine, eps, device, create_mean_rstd=False)
