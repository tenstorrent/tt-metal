# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

import tt_lib as ttl


def torch_layernorm(input,
                    normalized_dims=(3, ),
                    gamma=None,
                    beta=None,
                    epsilon=1e-5):
    """Applies Layer Normalization over a mini-batch of inputs.
    Args:
        input (torch.Tensor): The input tensor of shape (N, C, H, W).
        normalized_dims (tuple of ints): The dimensions to be normalized. Default: (3,).
        gamma (torch.Tensor): The scale tensor. Default: None.
        beta (torch.Tensor): The bias tensor. Default: None.
        epsilon (float): A value added to the denominator for numerical stability. Default: 1e-5.
    Returns:
        torch.Tensor: The output tensor of the same shape as input.
    """
    normalized_shape = input.shape[-len(normalized_dims):]
    if gamma is not None:
        gamma = gamma.reshape(normalized_shape)
    if beta is not None:
        beta = beta.reshape(normalized_shape)
    output = F.layer_norm(input,
                          normalized_shape,
                          weight=gamma,
                          bias=beta,
                          eps=epsilon)
    return output


TILE_HEIGHT = 32
TILE_WIDTH = 32


@pytest.mark.parametrize('eps', (1e-5, 1e-6, 1e-4),
                         ids=['1e-5', '1e-6', '1e-4'])
@pytest.mark.parametrize(
    'normalized_dims',
    (
        [3],
        [2, 3],
        [1, 2, 3],
        [0, 1, 2, 3],
    ),
    ids=['W', 'HW', 'CHW', 'NCHW'],
)
@pytest.mark.parametrize(
    'elementwise_affine',
    (False, True),
    ids=['elementwise_affine=False', 'elementwise_affine=True'],
)
@pytest.mark.parametrize('input_shape', (
    [1, 1, TILE_HEIGHT, TILE_WIDTH],
    [6, 6, 2 * TILE_HEIGHT, 2 * TILE_WIDTH],
    [2, 2, 32 * TILE_HEIGHT, 10 * TILE_WIDTH],
    [2, 2, 10 * TILE_HEIGHT, 32 * TILE_WIDTH],
    [2, 2, TILE_HEIGHT - 9, TILE_WIDTH - 9],
    [2, 2, TILE_HEIGHT + 13, TILE_WIDTH + 13],
    [2, 2, 8 * TILE_HEIGHT + 15, 32 * TILE_WIDTH - 15],
))
def test_moreh_layernorm(input_shape, normalized_dims, elementwise_affine, eps,
                         device):
    torch.manual_seed(2023)

    dtype = ttl.tensor.DataType.BFLOAT16
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    npu_layout = ttl.tensor.Layout.TILE

    # input
    cpu_input = torch.rand(input_shape, dtype=torch.bfloat16) * 2 - 1.05
    npu_input = ttl.tensor.Tensor(
        cpu_input.flatten().tolist(),
        input_shape,
        dtype,
        cpu_layout,
    ).pad_to_tile(float('nan')).to(npu_layout).to(device)

    normalized_shape = list(cpu_input.shape)[-len(normalized_dims):]
    gamma_beta_shape = [1] * (4 - len(normalized_dims)) + normalized_shape

    cpu_gamma = None
    cpu_beta = None
    npu_gamma = None
    npu_beta = None
    if elementwise_affine:
        # gamma
        cpu_gamma = torch.rand(gamma_beta_shape,
                               dtype=torch.bfloat16) * 2 - 1.05
        npu_gamma = ttl.tensor.Tensor(
            cpu_gamma.flatten().tolist(),
            gamma_beta_shape,
            dtype,
            cpu_layout,
        ).pad_to_tile(float('nan')).to(npu_layout).to(device)
        # beta
        cpu_beta = torch.rand(gamma_beta_shape,
                              dtype=torch.bfloat16) * 2 - 1.05
        npu_beta = ttl.tensor.Tensor(
            cpu_beta.flatten().tolist(),
            gamma_beta_shape,
            dtype,
            cpu_layout,
        ).pad_to_tile(float('nan')).to(npu_layout).to(device)

    # output
    expected = torch_layernorm(cpu_input, normalized_dims, cpu_gamma, cpu_beta,
                               eps)
    npu_output = ttl.operations.primary.moreh_layernorm(
        npu_input, eps, normalized_dims, npu_gamma, npu_beta)
    actual = npu_output.cpu().to(cpu_layout).unpad_from_tile(
        input_shape).to_torch()

    rtol = atol = 0.1
    if len(normalized_dims) in (2, 3):
        rtol = atol = 0.16
    elif len(normalized_dims) == 4:
        rtol = atol = 0.2

    assert torch.allclose(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize('eps', (1e-5, 1e-6, 1e-4),
                         ids=['1e-5', '1e-6', '1e-4'])
@pytest.mark.parametrize(
    'normalized_dims',
    (
        [3],
        [2, 3],
        [1, 2, 3],
        [0, 1, 2, 3],
    ),
    ids=['W', 'HW', 'CHW', 'NCHW'],
)
@pytest.mark.parametrize(
    'elementwise_affine',
    (False, True),
    ids=['elementwise_affine=False', 'elementwise_affine=True'],
)
@pytest.mark.parametrize('input_shape', (
    [1, 1, TILE_HEIGHT, TILE_WIDTH],
    [2, 2, TILE_HEIGHT - 9, TILE_WIDTH - 9],
    [2, 2, TILE_HEIGHT + 13, TILE_WIDTH + 13],
))
def test_moreh_layernorm_with_autoformat(input_shape, normalized_dims,
                                         elementwise_affine, eps, device):
    torch.manual_seed(2023)

    dtype = ttl.tensor.DataType.BFLOAT16
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    npu_layout = ttl.tensor.Layout.ROW_MAJOR

    # input
    cpu_input = torch.rand(input_shape, dtype=torch.bfloat16) * 2 - 1.05
    npu_input = ttl.tensor.Tensor(
        cpu_input.flatten().tolist(),
        input_shape,
        dtype,
        cpu_layout,
    ).pad_to_tile(float('nan')).to(npu_layout).to(device)

    normalized_shape = list(cpu_input.shape)[-len(normalized_dims):]
    gamma_beta_shape = [1] * (4 - len(normalized_dims)) + normalized_shape

    cpu_gamma = None
    cpu_beta = None
    npu_gamma = None
    npu_beta = None
    if elementwise_affine:
        # gamma
        cpu_gamma = torch.rand(gamma_beta_shape,
                               dtype=torch.bfloat16) * 2 - 1.05
        npu_gamma = ttl.tensor.Tensor(
            cpu_gamma.flatten().tolist(),
            gamma_beta_shape,
            dtype,
            cpu_layout,
        ).pad_to_tile(float('nan')).to(device)
        # beta
        cpu_beta = torch.rand(gamma_beta_shape,
                              dtype=torch.bfloat16) * 2 - 1.05
        npu_beta = ttl.tensor.Tensor(
            cpu_beta.flatten().tolist(),
            gamma_beta_shape,
            dtype,
            cpu_layout,
        ).pad_to_tile(float('nan')).to(npu_layout).to(device)

    # output
    expected = torch_layernorm(cpu_input, normalized_dims, cpu_gamma, cpu_beta,
                               eps)
    npu_output = ttl.tensor.moreh_layernorm(npu_input, eps, normalized_dims,
                                            npu_gamma, npu_beta)
    actual = npu_output.cpu().to(cpu_layout).unpad_from_tile(
        input_shape).to_torch()

    rtol = atol = 0.1
    if len(normalized_dims) in (2, 3):
        rtol = atol = 0.16
    elif len(normalized_dims) == 4:
        rtol = atol = 0.2

    assert torch.allclose(actual, expected, rtol=rtol, atol=atol)
