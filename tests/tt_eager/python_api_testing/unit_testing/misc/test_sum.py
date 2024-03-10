# SPDX-FileCopyrightText: © 2023-24 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib as ttl


@pytest.mark.parametrize(
    "shape_dim",
    (
        ((1, 1, 32, 32), 3),
        ((1, 1, 32, 32), 2),
        ((32, 32, 32, 32), 1),
        ((32, 32, 32, 32), 0),
    ),  # single tile
)
def test_sum_for_dim_hw(device, use_program_cache, shape_dim):
    shape, dim = shape_dim
    torch.manual_seed(0)

    N = shape[0]
    C = shape[1]
    H = shape[2]
    W = shape[3]

    input_shape = (N, C, H, W)
    x = 1.0 + torch.arange(0, N * C * H * W).reshape(input_shape).bfloat16()

    value = x.sum(dim=dim, keepdim=True)[0, 0, 0, 0]
    # print(f"x.sum = {value}")

    dev_x = ttl.tensor.Tensor(x, ttl.tensor.DataType.BFLOAT16).to(ttl.tensor.Layout.TILE).to(device)
    tt_npu = ttl.tensor.sum(dev_x, dim)
    tt_dev = tt_npu.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    assert torch.equal(tt_dev[0, 0, 0, 0], torch.Tensor([value]).bfloat16()[0])


@pytest.mark.parametrize(
    "shape_dim",
    (
        ((1, 1, 32, 32), 3),
        ((1, 1, 32, 32), 2),
        ((32, 32, 32, 32), 1),
        ((32, 32, 32, 32), 0),
    ),  # single tile
)
def test_sum_global(device, use_program_cache, shape_dim):
    shape, dim = shape_dim
    torch.manual_seed(0)

    N = shape[0]
    C = shape[1]
    H = shape[2]
    W = shape[3]

    input_shape = (N, C, H, W)
    x = 1.0 + torch.ones(input_shape).bfloat16()

    value = x.sum()

    dev_x = ttl.tensor.Tensor(x, ttl.tensor.DataType.BFLOAT16).to(ttl.tensor.Layout.TILE).to(device)
    tt_npu = ttl.tensor.global_sum(dev_x)
    tt_dev = tt_npu.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    assert torch.equal(tt_dev[0, 0, 0, 0].bfloat16(), torch.Tensor([value]).bfloat16()[0])
