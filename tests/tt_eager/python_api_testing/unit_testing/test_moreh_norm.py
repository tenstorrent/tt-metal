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


# if __name__ == "__main__":
#     print("test_moreh_norm.py")
#     device = ttl.device.CreateDevice(0)

#     torch.manual_seed(2024)
#     torch.set_printoptions(edgeitems=32, linewidth=1000, precision=2)

#     input_shape = (1, 1, TILE_HEIGHT, TILE_WIDTH)

#     input_shape = (2, 2, 2 * TILE_HEIGHT + 13, 2 * TILE_WIDTH + 13)

#     # cpu_x = torch.randint(0, 3, input_shape, dtype=torch.float32).requires_grad_()
#     cpu_x = torch.randint(-9, 10, input_shape, dtype=torch.float32).requires_grad_()
#     # aa = to_npu(a.bfloat16(), device)

#     # print("a")
#     # print(a)
#     # print(a.shape)

#     # dim = 0
#     # dim = 1
#     # dim = 2
#     # dim = 3

#     # dim = [2, 3]
#     # dim = (0, 1, 2, 3)
#     # dim = [0, 2, 3]
#     dim = [0, 2]
#     # dim = (2,)
#     # dim = [2]
#     # dim = 2
#     # dim = 3
#     # dim = None

#     p = 2.0
#     # p = 2.5
#     # p = -2.0
#     # p = -3.3
#     # p = -1.5
#     # print(a)

#     cpu_y = torch.norm(cpu_x, p=p, dim=dim, keepdim=True)

#     # cpu_dy = torch.randint(0, 3, cpu_y.shape, dtype=torch.float32)
#     cpu_dy = torch.randint(-9, 10, cpu_y.shape, dtype=torch.float32)
#     print("cpu_dy")
#     # print(cpu_dy)
#     print(cpu_dy.shape)

#     cpu_y.backward(cpu_dy)
#     # print("cpu_x.grad")
#     # print(cpu_x.grad)

#     npu_x = to_npu(cpu_x.bfloat16(), device)
#     npu_y = to_npu(cpu_y.bfloat16(), device)
#     npu_dy = to_npu(cpu_dy.bfloat16(), device)

#     npu_dx = to_npu(torch.zeros(input_shape, dtype=torch.bfloat16), device)
#     ttl.operations.primary.moreh_norm_backward(npu_x, npu_y, npu_dy, p=p, input_grad=npu_dx)

#     # npu_dx = ttl.operations.primary.moreh_norm_backward(npu_x, npu_y, npu_dy, p=p)

#     # print("npu_y")
#     # print(npu_y)

#     # print("npu_dy")
#     # print(npu_dy)

#     cpu_dx = cpu_x.grad
#     npu_dx = to_cpu(npu_dx, list(cpu_dx.shape))

#     # print("cpu_dx")
#     # print(cpu_dx)
#     # print(cpu_dx.shape)

#     # print("npu_dx")
#     # print(npu_dx)
#     # print(npu_dx.shape)

#     # aa = to_npu(a.bfloat16(), device)

#     # a_grad = (x.pow(p - 1.0) * y * dy) / y.pow(p)
#     # print(a_grad)

#     # print(torch.allclose(a_grad, a.grad))

#     atol = rtol = 0.1
#     print(torch.allclose(npu_dx.float(), cpu_dx, atol=atol, rtol=rtol))

#     ttl.device.CloseDevice(device)

# input_grad = (input.pow(p - 1.0) * output.unsqueeze(dim) * output_grad.unsqueeze(dim)) / (output.unsqueeze(dim).pow(p))

# input_grad = (input.pow(p - 1.0) * output * output_grad) / (output.pow(p))

if __name__ == "__main__":
    print("test_moreh_norm.py")
    device = ttl.device.CreateDevice(0)

    torch.manual_seed(2024)
    torch.set_printoptions(edgeitems=32, linewidth=1000)

    # input_shape = (1, 1, TILE_HEIGHT, TILE_WIDTH)

    # input_shape = (1, 1, TILE_HEIGHT, 2 * TILE_WIDTH)

    # input_shape = (2, 1, TILE_HEIGHT, TILE_WIDTH)

    # input_shape = (1, 2, TILE_HEIGHT, TILE_WIDTH)

    # input_shape = (1, 2, 2 * TILE_HEIGHT, TILE_WIDTH)

    # input_shape = (1, 2, 2 * TILE_HEIGHT, 2 * TILE_WIDTH)

    # input_shape = (2, 2, TILE_HEIGHT, TILE_WIDTH)

    # input_shape = (1, 1, 2 * TILE_HEIGHT, TILE_WIDTH)

    # input_shape = (1, 1, TILE_HEIGHT + 15, 2 * TILE_WIDTH)

    # input_shape = (1, 1, TILE_HEIGHT, TILE_WIDTH - 15)

    # input_shape = (1, 1, TILE_HEIGHT - 15, TILE_WIDTH)

    # input_shape = (1, 1, TILE_HEIGHT, TILE_WIDTH + 15)

    # input_shape = (2, 2, 2 * TILE_HEIGHT, 2 * TILE_WIDTH)

    input_shape = (2, 2, 2 * TILE_HEIGHT + 13, 2 * TILE_WIDTH + 13)

    cpu_x = torch.randint(-5, 6, input_shape, dtype=torch.float32)
    npu_x = to_npu(cpu_x.bfloat16(), device)

    print(cpu_x.shape)

    # dim = 0
    # dim = 1
    # dim = 2
    # dim = 3

    # dim = [2, 3]
    # dim = (0, 1, 2, 3)
    dim = [0, 2, 3]
    # dim = (2,)
    # dim = [2]
    # dim = 2
    # dim = 3
    dim = None

    p = 2.0
    # p = 2.5
    # p = -2.0
    # p = -3.3
    # p = -1.5
    # print(a)

    cpu_y = torch.norm(cpu_x, p=p, dim=dim, keepdim=True)
    npu_y = to_npu(torch.zeros(cpu_y.shape, dtype=torch.bfloat16), device)
    ttl.operations.primary.moreh_norm(npu_x, p=p, dim=dim, output=npu_y)

    # cpu_y = torch.norm(cpu_x, p=p, dim=dim, keepdim=True)
    # npu_y = ttl.operations.primary.moreh_norm(npu_x, p=p, dim=dim)

    # b = torch.norm(a, p=p, keepdim=True)
    # bb = ttl.operations.primary.moreh_norm(aa, p=p)

    print(cpu_y.shape)
    print()
    npu_y = to_cpu(npu_y, list(cpu_y.shape))
    # print(npu_y)

    atol = rtol = 0.1
    print(torch.allclose(npu_y.float(), cpu_y, atol=atol, rtol=rtol))

    ttl.device.CloseDevice(device)
