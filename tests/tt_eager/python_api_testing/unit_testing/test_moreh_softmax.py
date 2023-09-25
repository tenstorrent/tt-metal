# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
import sys

import torch

import tt_lib as ttl
from models.utility_functions import print_diff_argmax
import pytest

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

@pytest.mark.parametrize(
    "shape_dim",
    (
        ((1, 1, 32, 32), 3), # single tile
        ((1, 1, 32, 32 * 5), 3), # mutiple tile with dim
        ((5, 6, 32, 32), 3), # multiple cores
        ((10, 20, 32 * 3, 32 * 5), 3), # complex test
        ((1, 1, 32, 32), 2), # single tile
        ((1, 1, 32 * 5, 32), 2), # mutiple tile with dim
        ((5, 6, 32, 32), 2), # multiple cores
        ((10, 20, 32 * 3, 32 * 5), 2), # complex test
    ),
)
def test_softmax_for_dim_hw(shape_dim, device):
    shape, dim = shape_dim
    torch.manual_seed(0)

    N = shape[0]
    C = shape[1]
    H = shape[2]
    W = shape[3]

    x = torch.randint(low=0, high=4, size=(N * C * H * W,)).reshape((N, C, H, W)).to(torch.bfloat16)

    dev_x = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(ttl.tensor.Layout.TILE).to(device)

    tt_cpu = torch.softmax(x, dim)
    tt_npu = ttl.operations.primary.moreh_softmax(dev_x, dim)

    assert tt_npu.shape() == list(tt_cpu.shape)
    tt_dev = tt_npu.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch().to(torch.bfloat16)

    assert torch.allclose(tt_cpu, tt_dev, rtol = 0.07, atol = 0.01)


# TODO: uncomment after implement softmax for dim n and c.
# @pytest.mark.parametrize(
#     "shape_dim",
#     (
#         # ((32, 32, 1, 1), 0), # single tile n
#         # ((32 * 5, 32, 1, 1), 0), # mutiple tile with dim
#         # ((32, 32, 5, 6), 0), # multiple coresa
#         # ((1, 32, 1, 1), 1), # single tile c
#         # ((32, 32 * 5, 1, 1), 1), # mutiple tile with dim
#         # ((32, 32, 5, 6), 1), # multiple coresa

#         ((1, 32, 1, 1), 1), # single tile c
#         # ((1, 1, 32, 1024), 2), # single tile c
#     ),
# )
# def test_softmax_for_dim_nc(shape_dim, device):
#     shape, dim = shape_dim
#     torch.manual_seed(0)

#     N = shape[0]
#     C = shape[1]
#     H = shape[2]
#     W = shape[3]

#     # x = torch.randint(low=0, high=4, size=(N * C * H * W,)).reshape((N, C, H, W)).to(torch.bfloat16)
#     x = torch.randint(low=1, high=2, size=(N * C * H * W,)).reshape((N, C, H, W)).to(torch.bfloat16)

#     # dev_x = ttl.tensor.Tensor(
#     #     x.reshape(-1).tolist(),
#     #     x.shape,
#     #     ttl.tensor.DataType.BFLOAT16,
#     #     ttl.tensor.Layout.ROW_MAJOR,
#     # ).pad_to_tile(float("5")).to(ttl.tensor.Layout.TILE).to(device)
#     # dev_x = ttl.tensor.Tensor(
#     #     x.reshape(-1).tolist(),
#     #     x.shape,
#     #     ttl.tensor.DataType.BFLOAT16,
#     #     ttl.tensor.Layout.ROW_MAJOR,
#     # ).pad_to_tile(float("5")).reshape(1,1,32,1024).to(ttl.tensor.Layout.TILE).to(device)

#     tt_cpu = torch.softmax(x, dim)
#     tt_npu = ttl.operations.primary.moreh_softmax(dev_x, dim)
#     # tt_npu = tt_npu.cpu().to(ttl.tensor.Layout.ROW_MAJOR).unpad_from_tile((N, C, H, W))
#     # tt_npu = tt_npu.cpu().to(ttl.tensor.Layout.ROW_MAJOR).reshape(1,32,32,32).unpad_from_tile((N, C, H, W))

#     assert tt_npu.shape() == list(tt_cpu.shape)
#     tt_dev = tt_npu.to_torch().to(torch.bfloat16)

#     print("cpu")
#     print(tt_cpu.reshape(1,1,1,32))
#     print("")
#     print("npu")
#     print(tt_dev.reshape(1,1,1,32))

#     assert torch.allclose(tt_cpu, tt_dev, rtol = 0.07, atol = 0.01)
