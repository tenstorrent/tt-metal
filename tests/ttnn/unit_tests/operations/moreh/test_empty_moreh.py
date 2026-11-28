# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


def test_moreh_mean_empty_reduced_dim(device):
    # The mean over an empty dim is NaN in torch, and the W factory divides by that dim.
    x = torch.zeros((1, 1, 32, 0), dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.moreh_mean(ttnn_input, dim=3, keepdim=True)
    expected = torch.mean(x.float(), dim=3, keepdim=True)
    assert tuple(output.shape) == tuple(expected.shape)
    assert torch.isnan(ttnn.to_torch(output).float()).all()


@pytest.mark.parametrize("shape", [(1, 1, 0, 32), (1, 1, 32, 0)])
def test_moreh_sgd_empty(device, shape):
    # An empty param has nothing to update; the factory divided by its dims.
    torch_param = torch.zeros(shape, dtype=torch.bfloat16)
    ttnn_param = ttnn.from_torch(torch_param, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.moreh_sgd(ttnn_param, ttnn_param, lr=0.1, momentum_initialized=False)
    assert tuple(output[0].shape) == shape
    assert output[0].logical_volume() == 0
    assert output[1] is None


@pytest.mark.parametrize("shape", [(1, 4, 0, 32), (1, 4, 32, 0)])
def test_moreh_group_norm_empty(device, shape):
    # An empty input has an empty output and NaN mean/rstd, as in torch; the program never completed on it.
    x = torch.zeros(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
    output, mean, rstd = ttnn.moreh_group_norm(ttnn_input, 2, are_required_outputs=[True, True, True])
    assert tuple(output.shape) == shape
    assert tuple(mean.shape) == (1, 1, 1, 2)
    assert tuple(rstd.shape) == (1, 1, 1, 2)
    assert torch.isnan(ttnn.to_torch(mean).float()).all()
    assert torch.isnan(ttnn.to_torch(rstd).float()).all()


def test_moreh_mean_rank0(device):
    # The mean of a rank 0 tensor is the scalar itself; the dim loop indexed an empty dim list.
    x = torch.tensor(3.0, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output = ttnn.moreh_mean(ttnn_input)
    assert tuple(output.shape) == ()
    assert float(ttnn.to_torch(output).float().reshape(-1)[0]) == 3.0


def test_moreh_sum_rank0(device):
    # The sum of a rank 0 tensor is the scalar itself; the dim loop indexed an empty dim list.
    x = torch.tensor(3.0, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output = ttnn.moreh_sum(ttnn_input)
    assert tuple(output.shape) == ()
    assert float(ttnn.to_torch(output).float().reshape(-1)[0]) == 3.0


def test_moreh_empty_guards_nonempty_control(device):
    # Non-empty inputs keep going through the device programs and match torch.
    x = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
    mean = ttnn.to_torch(ttnn.moreh_mean(ttnn_input)).float()
    assert tuple(mean.shape) == (1, 1)
    assert float(mean.reshape(-1)[0]) == 3.0
    total = ttnn.to_torch(ttnn.moreh_sum(ttnn_input)).float()
    assert tuple(total.shape) == (1, 1)
    # 32 x 32 elements of 3.0: the full reduction is 3072, as in torch.
    assert float(total.reshape(-1)[0]) == 3072.0

    param = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    grad = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    sgd_output = ttnn.moreh_sgd(
        ttnn.from_torch(param, layout=ttnn.TILE_LAYOUT, device=device),
        ttnn.from_torch(grad, layout=ttnn.TILE_LAYOUT, device=device),
        lr=0.5,
        momentum_initialized=False,
    )
    assert torch.allclose(
        ttnn.to_torch(sgd_output[0]).float(), param.float() - 0.5 * grad.float(), atol=0.05, rtol=0.05
    )

    norm_input = torch.randn(1, 4, 32, 32, dtype=torch.bfloat16)
    norm_output = ttnn.moreh_group_norm(ttnn.from_torch(norm_input, layout=ttnn.TILE_LAYOUT, device=device), 2)[0]
    reference = torch.nn.functional.group_norm(norm_input.float(), 2, eps=1e-5)
    assert torch.allclose(ttnn.to_torch(norm_output).float(), reference, atol=0.1, rtol=0.1)
