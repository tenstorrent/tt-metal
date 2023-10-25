# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

import tt_lib as ttl
from tests.tt_eager.python_api_testing.sweep_tests.common import skip_for_wormhole_b0
from models.utility_functions import comp_allclose_and_pcc
from loguru import logger


def torch_layernorm_backward(input,
                             output_grad,
                             *,
                             normalized_dims=(3, ),
                             gamma=None,
                             beta=None,
                             eps=1e-5):
    normalized_shape = input.shape[-len(normalized_dims):]

    input.requires_grad_()
    if gamma is not None:
        gamma = gamma.reshape(normalized_shape)
        gamma.requires_grad_()
    if beta is not None:
        beta = beta.reshape(normalized_shape)
        beta.requires_grad_()

    output = F.layer_norm(input,
                          normalized_shape,
                          weight=gamma,
                          bias=beta,
                          eps=eps)
    output.backward(output_grad)

    gamma_grad = None
    beta_grad = None
    if gamma is not None:
        gamma_grad = gamma.grad
    if beta is not None:
        beta_grad = beta.grad

    return input.grad, gamma_grad, beta_grad


def make_tensors(input_shape, normalized_dims, elementwise_affine, eps, device,
                 cpu_layout, npu_layout):
    # rank
    input_rank = len(input_shape)
    normalized_rank = len(normalized_dims)

    # output_grad_shape
    output_grad_shape = input_shape

    # mean_rstd_shape
    mean_rstd_shape = input_shape[:-normalized_rank] + [1] * normalized_rank

    # gamma_beta_shape
    gamma_beta_shape = [1] * (input_rank -
                              normalized_rank) + input_shape[-normalized_rank:]

    # dtype
    npu_dtype = ttl.tensor.DataType.BFLOAT16
    cpu_dtype = torch.bfloat16

    # output_grad
    cpu_output_grad = torch.randint(-2, 3, output_grad_shape, dtype=cpu_dtype)
    npu_output_grad = ttl.tensor.Tensor(
        cpu_output_grad.flatten().tolist(),
        output_grad_shape,
        npu_dtype,
        cpu_layout,
    ).pad_to_tile(float('nan')).to(npu_layout).to(device)

    # input
    cpu_input = torch.randint(-2, 3, input_shape, dtype=cpu_dtype)
    npu_input = ttl.tensor.Tensor(
        cpu_input.flatten().tolist(),
        input_shape,
        npu_dtype,
        cpu_layout,
    ).pad_to_tile(float('nan')).to(npu_layout).to(device)

    # mean
    cpu_mean = cpu_input.mean(dim=normalized_dims, keepdim=True)
    npu_mean = ttl.tensor.Tensor(
        cpu_mean.flatten().tolist(),
        mean_rstd_shape,
        npu_dtype,
        cpu_layout,
    ).pad_to_tile(float('nan')).to(npu_layout).to(device)

    # rstd
    cpu_var = ((cpu_input - cpu_mean)**2).mean(dim=normalized_dims,
                                               keepdim=True)
    cpu_rstd = (cpu_var + eps).sqrt()
    npu_rstd = ttl.tensor.Tensor(
        cpu_rstd.flatten().tolist(),
        mean_rstd_shape,
        npu_dtype,
        cpu_layout,
    ).pad_to_tile(float('nan')).to(npu_layout).to(device)

    # gamma
    cpu_gamma, npu_gamma = None, None
    if elementwise_affine:
        cpu_gamma = torch.rand(gamma_beta_shape, dtype=cpu_dtype) * 2 - 1.05
        npu_gamma = ttl.tensor.Tensor(
            cpu_gamma.flatten().tolist(),
            gamma_beta_shape,
            npu_dtype,
            cpu_layout,
        ).pad_to_tile(float('nan')).to(npu_layout).to(device)

    # beta
    cpu_beta = None
    if elementwise_affine:
        cpu_beta = torch.rand(gamma_beta_shape, dtype=cpu_dtype) * 2 - 1.05

    # input_grad
    cpu_input_grad = torch.full(input_shape, float('nan'), dtype=cpu_dtype)
    npu_input_grad = ttl.tensor.Tensor(
        cpu_input_grad.flatten().tolist(),
        input_shape,
        npu_dtype,
        cpu_layout,
    ).pad_to_tile(float('nan')).to(npu_layout).to(device)

    # gamma_grad
    cpu_gamma_grad, npu_gamma_grad = None, None
    if elementwise_affine:
        cpu_gamma_grad = torch.full(gamma_beta_shape,
                                    float('nan'),
                                    dtype=cpu_dtype)
        npu_gamma_grad = ttl.tensor.Tensor(
            cpu_gamma_grad.flatten().tolist(),
            gamma_beta_shape,
            npu_dtype,
            cpu_layout,
        ).pad_to_tile(float('nan')).to(npu_layout).to(device)

    # beta_grad
    cpu_beta_grad, npu_beta_grad = None, None
    if elementwise_affine:
        cpu_beta_grad = torch.full(gamma_beta_shape,
                                   float('nan'),
                                   dtype=cpu_dtype)
        npu_beta_grad = ttl.tensor.Tensor(
            cpu_beta_grad.flatten().tolist(),
            gamma_beta_shape,
            npu_dtype,
            cpu_layout,
        ).pad_to_tile(float('nan')).to(npu_layout).to(device)

    return cpu_input, cpu_output_grad, cpu_gamma, cpu_beta, npu_input, npu_output_grad, npu_mean, npu_rstd, npu_gamma, npu_input_grad, npu_gamma_grad, npu_beta_grad


TILE_HEIGHT = 32
TILE_WIDTH = 32


@pytest.mark.parametrize('eps', [1e-5, 1e-6, 1e-4],
                         ids=['1e-5', '1e-6', '1e-4'])
@pytest.mark.parametrize('normalized_dims',
                         [[3], [2, 3], [1, 2, 3], [0, 1, 2, 3]],
                         ids=['W', 'HW', 'CHW', 'NCHW'])
@pytest.mark.parametrize(
    'elementwise_affine',
    [False, True],
    ids=['elementwise_affine=False', 'elementwise_affine=True'],
)
@pytest.mark.parametrize(
    'input_shape',
    [
        [1, 1, TILE_HEIGHT, TILE_WIDTH],
        [6, 6, 2 * TILE_HEIGHT, 2 * TILE_WIDTH],
        [2, 2, 32 * TILE_HEIGHT, 10 * TILE_WIDTH],
        [2, 2, 10 * TILE_HEIGHT, 32 * TILE_WIDTH],
        [1, 1, TILE_HEIGHT - 19, TILE_WIDTH],
        [1, 1, TILE_HEIGHT, TILE_WIDTH - 19],
        [2, 2, TILE_HEIGHT - 9, TILE_WIDTH - 9],
        [1, 1, TILE_HEIGHT + 13, TILE_WIDTH],
        [2, 2, TILE_HEIGHT + 13, TILE_WIDTH + 13],
        [2, 2, 8 * TILE_HEIGHT + 15, 32 * TILE_WIDTH - 15],
        # Test large algorithm.
        [2, 2, TILE_HEIGHT + 13, 1000 * TILE_WIDTH + 17],
    ])
@skip_for_wormhole_b0
def test_moreh_layernorm_backward(input_shape, normalized_dims,
                                  elementwise_affine, eps, device):
    torch.manual_seed(2023)

    # layout
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    npu_layout = ttl.tensor.Layout.TILE

    cpu_input, cpu_output_grad, cpu_gamma, cpu_beta, npu_input, npu_output_grad, npu_mean, npu_rstd, npu_gamma, npu_input_grad, npu_gamma_grad, npu_beta_grad = make_tensors(
        input_shape, normalized_dims, elementwise_affine, eps, device,
        cpu_layout, npu_layout)

    # expected
    expected_input_grad, expected_gamma_grad, expected_beta_grad = torch_layernorm_backward(
        cpu_input,
        cpu_output_grad,
        normalized_dims=normalized_dims,
        gamma=cpu_gamma,
        beta=cpu_beta)

    # actual
    _, npu_gamma_grad, _ = ttl.operations.primary.moreh_layernorm_backward(
        npu_output_grad,
        npu_input,
        npu_mean,
        npu_rstd,
        normalized_dims,
        gamma=npu_gamma,
        input_grad=npu_input_grad,
        gamma_grad=npu_gamma_grad,
        beta_grad=npu_beta_grad)

    actual_input_grad = npu_input_grad.cpu().to(cpu_layout).unpad_from_tile(
        input_shape).to_torch()

    # Set rtol and atol and pcc
    rtol = atol = 0.1
    pcc = 0.999

    # Check input_grad
    pig, oig = comp_allclose_and_pcc(expected_input_grad,
                                     actual_input_grad,
                                     rtol=rtol,
                                     atol=atol,
                                     pcc=pcc)
    logger.info(f'input_grad\'s {oig}')
    assert pig

    # gamma_beta_shape
    input_rank = len(input_shape)
    normalized_rank = len(normalized_dims)
    gamma_beta_shape = [1] * (input_rank -
                              normalized_rank) + input_shape[-normalized_rank:]

    # I divide gamma_grad and beta_grad by (N * C * Ht), because the error increases.
    N, C, H, _ = input_shape
    Ht = (H + TILE_HEIGHT - 1) // TILE_HEIGHT
    numerator = N * C * Ht

    # Check gamma_grad
    if npu_gamma_grad is not None:
        actual_gamma_grad = npu_gamma_grad.cpu().to(
            cpu_layout).unpad_from_tile(gamma_beta_shape).to_torch()
        broadcasted_expected_gamma_grad, broadcasted_actual_gamma_grad = torch.broadcast_tensors(
            expected_gamma_grad, actual_gamma_grad)
        pgg, ogg = comp_allclose_and_pcc(
            broadcasted_expected_gamma_grad / numerator,
            broadcasted_actual_gamma_grad / numerator,
            rtol=rtol,
            atol=atol,
            pcc=pcc)
        logger.info(f'gamma_grad\'s {ogg}')
        assert pgg
    else:
        assert expected_gamma_grad is None

    # Check beta_grad
    if npu_beta_grad is not None:
        actual_beta_grad = npu_beta_grad.cpu().to(cpu_layout).unpad_from_tile(
            gamma_beta_shape).to_torch()
        broadcasted_expected_beta_grad, broadcasted_actual_beta_grad = torch.broadcast_tensors(
            expected_beta_grad, actual_beta_grad)
        pbg, obg = comp_allclose_and_pcc(
            broadcasted_expected_beta_grad / numerator,
            broadcasted_actual_beta_grad / numerator,
            rtol=rtol,
            atol=atol,
            pcc=pcc)
        logger.info(f'beta_grad\'s {obg}')
        assert pbg
    else:
        assert expected_beta_grad is None
