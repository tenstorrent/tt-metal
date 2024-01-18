# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

import tt_lib as ttl
from models.utility_functions import comp_allclose_and_pcc, skip_for_wormhole_b0
from loguru import logger

TILE_HEIGHT = 32
TILE_WIDTH = 32


def to_cpu(npu_tensor, shape, *, cpu_layout=ttl.tensor.Layout.ROW_MAJOR):
    if npu_tensor is None:
        return None
    cpu_tensor = npu_tensor.cpu().to(cpu_layout).unpad_from_tile(shape).to_torch()
    return cpu_tensor


def to_npu(
    cpu_tensor,
    device,
    *,
    npu_layout=ttl.tensor.Layout.TILE,
    npu_dtype=ttl.tensor.DataType.BFLOAT16,
):
    if cpu_tensor is None:
        return None
    npu_tensor = ttl.tensor.Tensor(cpu_tensor, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    return npu_tensor


if __name__ == "__main__":
    print("test_moreh_norm.py")
    device = ttl.device.CreateDevice(0)

    torch.manual_seed(2024)
    torch.set_printoptions(edgeitems=32, linewidth=1000)

    # input_shape = (1, 1, TILE_HEIGHT, TILE_WIDTH)

    # input_shape = (1, 1, TILE_HEIGHT, 2 * TILE_WIDTH)

    # input_shape = (1, 1, 2 * TILE_HEIGHT, TILE_WIDTH)

    # input_shape = (1, 1, TILE_HEIGHT + 15, 2 * TILE_WIDTH)

    input_shape = (1, 1, TILE_HEIGHT, TILE_WIDTH - 15)

    input_shape = (1, 1, TILE_HEIGHT - 15, TILE_WIDTH)

    # input_shape = (1, 1, TILE_HEIGHT, TILE_WIDTH + 15)

    # input_shape = (2, 2, 2 * TILE_HEIGHT, 2 * TILE_WIDTH)

    input_shape = (2, 2, 2 * TILE_HEIGHT + 13, 2 * TILE_WIDTH + 13)

    a = torch.randint(0, 5, input_shape, dtype=torch.float32)
    aa = to_npu(a.bfloat16(), device)

    print(a.shape)

    dim = 2

    p = 2.0
    # p = 2.5
    # p = -2.0
    # p = -3.3
    # p = -1.5
    # print(a)

    b = torch.norm(a, p=p, dim=dim, keepdim=True)

    bb = ttl.operations.primary.moreh_norm(aa, p=p, dim=dim)

    # print(to_cpu(bb, list(b.shape)))
    # print()
    # print(b)
    print(b.shape)

    atol = rtol = 0.1
    print(torch.allclose(to_cpu(bb, list(b.shape)).float(), b, atol=atol, rtol=rtol))

    ttl.device.CloseDevice(device)
