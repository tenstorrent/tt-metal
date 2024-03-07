# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import tt_lib as ttl
from models.utility_functions import comp_allclose, skip_for_wormhole_b0
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


def make_cpu_tensors(input_shape, dim):
    if dim is None or dim == []:
        dim = list(range(len(input_shape)))

    if isinstance(dim, int):
        dim = [dim]

    # output_grad_shape
    output_grad_shape = input_shape[:]

    for d in dim:
        output_grad_shape[d] = 1

    # input
    cpu_input = torch.empty(input_shape, dtype=torch.float32).uniform_(-1, 1).requires_grad_()

    # output_grad
    cpu_output_grad = torch.empty(output_grad_shape, dtype=torch.float32).uniform_(-1, 1)

    return cpu_input, cpu_output_grad


def torch_norm(cpu_x, cpu_dy, *, p=2.0, dim=None, do_backward=False):
    cpu_y = torch.norm(cpu_x, p=p, dim=dim, keepdim=True)

    cpu_dx = None
    if do_backward:
        cpu_y.backward(cpu_dy)
        cpu_dx = cpu_x.grad

    return cpu_y, cpu_dx


def tt_norm(cpu_x, cpu_dy, *, p=2.0, dim=None, do_backward=False, device=None):
    npu_x = to_npu(cpu_x.bfloat16(), device)
    if do_backward:
        npu_dy = to_npu(cpu_dy.bfloat16(), device)

    npu_y = None
    if do_backward:
        npu_y = to_npu(torch.norm(cpu_x, p=p, dim=dim, keepdim=True).bfloat16(), device)
    else:
        npu_y = ttl.operations.primary.moreh_norm(npu_x, p=p, dim=dim)

    npu_dx = None
    if do_backward:
        npu_dx = ttl.operations.primary.moreh_norm_backward(npu_x, npu_y, npu_dy, p=p)
        npu_dx = to_cpu(npu_dx, list(cpu_x.shape))

    npu_y = to_cpu(npu_y, list(cpu_dy.shape))

    return npu_y, npu_dx


@skip_for_wormhole_b0()
@pytest.mark.parametrize("p", [2.0, 2.5, -2.5], ids=["p=2.0", "p=2.5", "p=-2.5"])
@pytest.mark.parametrize(
    "dim_rtol_atol",
    [
        [[], 0.2, 0.2],
        [None, 0.2, 0.2],
        [0, 0.1, 0.1],
        [1, 0.1, 0.1],
        [2, 0.1, 0.1],
        [3, 0.1, 0.1],
        [[0, 1], 0.1, 0.1],
        [[0, 1, 2], 0.15, 0.15],
        [[0, 1, 2, 3], 0.2, 0.2],
        [[0, 1, 3], 0.15, 0.15],
        [[0, 2, 3], 0.15, 0.15],
        [[1, 2], 0.1, 0.1],
        [[1, 2, 3], 0.15, 0.15],
        [[1, 3], 0.1, 0.1],
        [[2, 3], 0.1, 0.1],
    ],
    ids=[
        "global_norm(dim=[])",
        "global_norm(dim=None)",
        "N",
        "C",
        "H",
        "W",
        "NC",
        "NCH",
        "NCHW",
        "NCW",
        "NHW",
        "CH",
        "CHW",
        "CW",
        "HW",
    ],
)
@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 1, TILE_HEIGHT, TILE_WIDTH],
        [2, 2, 2 * TILE_HEIGHT + 13, 2 * TILE_WIDTH + 13],
        [16, 16, 8 * TILE_HEIGHT + 13, 8 * TILE_WIDTH + 13],
    ],
)
def test_moreh_norm(input_shape, p, dim_rtol_atol, device):
    torch.manual_seed(2024)

    dim, rtol, atol = dim_rtol_atol

    cpu_x, cpu_dy = make_cpu_tensors(input_shape, dim)

    # expected
    expected_y, _ = torch_norm(cpu_x, cpu_dy, p=p, dim=dim, do_backward=False)

    # actual
    actual_y, _ = tt_norm(cpu_x, cpu_dy, p=p, dim=dim, device=device, do_backward=False)

    # Check output
    pass_y, out_y = comp_allclose(expected_y, actual_y, rtol=rtol, atol=atol)
    logger.debug(f"output's {out_y}")
    assert pass_y


@skip_for_wormhole_b0()
@pytest.mark.parametrize("p", [2.0], ids=["p=2.0"])
@pytest.mark.parametrize(
    "dim_rtol_atol",
    [
        [[], 0.2, 0.2],
        [None, 0.2, 0.2],
        [0, 0.1, 0.1],
        [1, 0.1, 0.1],
        [2, 0.1, 0.1],
        [3, 0.1, 0.1],
        [[0, 1], 0.1, 0.1],
        [[0, 1, 2], 0.15, 0.15],
        [[0, 1, 2, 3], 0.2, 0.2],
        [[0, 1, 3], 0.15, 0.15],
        [[0, 2, 3], 0.15, 0.15],
        [[1, 2], 0.1, 0.1],
        [[1, 2, 3], 0.15, 0.15],
        [[1, 3], 0.1, 0.1],
        [[2, 3], 0.1, 0.1],
    ],
    ids=[
        "global_norm(dim=[])",
        "global_norm(dim=None)",
        "N",
        "C",
        "H",
        "W",
        "NC",
        "NCH",
        "NCHW",
        "NCW",
        "NHW",
        "CH",
        "CHW",
        "CW",
        "HW",
    ],
)
@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 1, TILE_HEIGHT, TILE_WIDTH],
        [2, 2, 2 * TILE_HEIGHT + 13, 2 * TILE_WIDTH + 13],
        [16, 16, 8 * TILE_HEIGHT + 13, 8 * TILE_WIDTH + 13],
    ],
)
def test_moreh_norm_backward(input_shape, p, dim_rtol_atol, device):
    torch.manual_seed(2024)

    dim, rtol, atol = dim_rtol_atol

    cpu_x, cpu_dy = make_cpu_tensors(input_shape, dim)

    # expected
    _, expected_dx = torch_norm(cpu_x, cpu_dy, p=p, dim=dim, do_backward=True)

    # actual
    _, actual_dx = tt_norm(cpu_x, cpu_dy, p=p, dim=dim, device=device, do_backward=True)

    # Check input_grad
    pass_dx, out_dx = comp_allclose(expected_dx, actual_dx, rtol=rtol, atol=atol)
    logger.debug(f"input_grad's {out_dx}")
    assert pass_dx
