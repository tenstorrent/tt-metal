# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.utility_functions import comp_allclose, is_wormhole_b0
from loguru import logger

from tests.tt_eager.python_api_testing.unit_testing.misc.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    to_cpu,
    to_npu,
    compute_output_shape,
    TILE_HEIGHT,
    TILE_WIDTH,
    check_dim,
)


def make_cpu_tensors(input_shape, dim, keepdim=False):
    torch_output_shape, _ = compute_output_shape(input_shape, dim, keepdim=keepdim)

    # input
    cpu_input = torch.empty(input_shape, dtype=torch.float32).uniform_(-1, 1).requires_grad_()

    # output_grad
    cpu_output_grad = torch.empty(torch_output_shape, dtype=torch.float32).uniform_(-1, 1)

    return cpu_input, cpu_output_grad


def torch_norm(cpu_x, cpu_dy, *, p=2.0, dim=None, keepdim=False, do_backward=False):
    cpu_y = torch.norm(cpu_x, p=p, dim=dim, keepdim=keepdim)

    cpu_dx = None
    if do_backward:
        cpu_y.backward(cpu_dy)
        cpu_dx = cpu_x.grad

    return cpu_y, cpu_dx


def tt_norm(
    cpu_x,
    cpu_dy,
    *,
    p=2.0,
    dim=None,
    keepdim=False,
    compute_kernel_options=None,
    do_backward=False,
    device=None,
):
    _, tt_output_shape = compute_output_shape(cpu_x.shape, dim, keepdim=keepdim)

    npu_x = to_npu(cpu_x.bfloat16(), device)
    if do_backward:
        npu_dy = to_npu(cpu_dy.reshape(tt_output_shape).bfloat16(), device)

    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    if do_backward:
        npu_y = to_npu(torch.norm(cpu_x, p=p, dim=dim, keepdim=keepdim).bfloat16().reshape(tt_output_shape), device)
    else:
        npu_y = to_npu(torch.empty(tt_output_shape), device)
        ttnn.experimental.operations.primary.moreh_norm(
            npu_x, p=p, dim=dim, keepdim=keepdim, output=npu_y, compute_kernel_config=compute_kernel_config
        )

    npu_dx = None
    if do_backward:
        npu_dx = to_npu(torch.empty_like(cpu_x), device)

        ttnn.experimental.operations.primary.moreh_norm_backward(
            npu_x,
            npu_y,
            npu_dy,
            p=p,
            dim=dim,
            keepdim=keepdim,
            input_grad=npu_dx,
            compute_kernel_config=compute_kernel_config,
        )
        npu_dx = to_cpu(npu_dx, list(cpu_x.shape))

    npu_y = to_cpu(npu_y, tt_output_shape)

    return npu_y, npu_dx


def run_moreh_norm(input_shape, p, dim, rtol, atol, device, keepdim=False, compute_kernel_options=None):
    if dim in (None, [], [0, 1, 2, 3]) and p == 2.5 and is_wormhole_b0():
        pytest.skip("TODO: Check why comp_allclose result is poor on WH_B0.")

    check_dim(input_shape, dim, keepdim)

    cpu_x, cpu_dy = make_cpu_tensors(input_shape, dim, keepdim=keepdim)

    # expected
    expected_y, _ = torch_norm(cpu_x, cpu_dy, p=p, dim=dim, keepdim=keepdim, do_backward=False)

    # actual
    actual_y, _ = tt_norm(
        cpu_x,
        cpu_dy,
        p=p,
        dim=dim,
        keepdim=keepdim,
        compute_kernel_options=compute_kernel_options,
        device=device,
        do_backward=False,
    )

    # Check output
    actual_y = actual_y if keepdim else actual_y.reshape(expected_y.shape)
    pass_y, out_y = comp_allclose(expected_y, actual_y, rtol=rtol, atol=atol)
    logger.debug(f"output's {out_y}")
    assert pass_y


def run_moreh_norm_backward(input_shape, p, dim, rtol, atol, device, keepdim=False, compute_kernel_options=None):
    check_dim(input_shape, dim, keepdim)

    cpu_x, cpu_dy = make_cpu_tensors(input_shape, dim, keepdim=keepdim)

    # expected
    _, expected_dx = torch_norm(cpu_x, cpu_dy, p=p, dim=dim, keepdim=keepdim, do_backward=True)

    # actual
    _, actual_dx = tt_norm(
        cpu_x,
        cpu_dy,
        p=p,
        dim=dim,
        compute_kernel_options=compute_kernel_options,
        device=device,
        do_backward=True,
    )

    # Check input_grad
    pass_dx, out_dx = comp_allclose(expected_dx, actual_dx, rtol=rtol, atol=atol)
    logger.debug(f"input_grad's {out_dx}")

    assert pass_dx


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
        [TILE_HEIGHT, TILE_WIDTH],
        [2, 2, 2 * TILE_HEIGHT + 13, 2 * TILE_WIDTH + 13],
    ],
)
@pytest.mark.parametrize("keepdim", [True, False], ids=["keepdim-true", "keepdim-flase"])
def test_moreh_norm(input_shape, p, dim_rtol_atol, keepdim, device):
    torch.manual_seed(2024)

    dim, rtol, atol = dim_rtol_atol

    run_moreh_norm(input_shape, p, dim, rtol, atol, device, keepdim=keepdim)


@pytest.mark.parametrize("p", [2.0, 2.5, -2.5], ids=["p=2.0", "p=2.5", "p=-2.5"])
@pytest.mark.parametrize(
    "dim_rtol_atol",
    [
        [0, 0.1, 0.1],
        [1, 0.1, 0.1],
        [2, 0.1, 0.1],
    ],
)
@pytest.mark.parametrize(
    "input_shape",
    [
        [10, TILE_HEIGHT, TILE_WIDTH],
    ],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_norm_compute_kernel_options(input_shape, p, dim_rtol_atol, compute_kernel_options, device):
    torch.manual_seed(2024)

    dim, rtol, atol = dim_rtol_atol

    run_moreh_norm(input_shape, p, dim, rtol, atol, device, compute_kernel_options=compute_kernel_options)


@pytest.mark.parametrize("p", [2.0], ids=["p=2.0"])
@pytest.mark.parametrize(
    "dim_rtol_atol",
    [
        [0, 0.1, 0.1],
        [1, 0.1, 0.1],
        [2, 0.1, 0.1],
    ],
)
@pytest.mark.parametrize(
    "input_shape",
    [
        [10, TILE_HEIGHT, TILE_WIDTH],
    ],
)
def test_moreh_norm_callback(input_shape, p, dim_rtol_atol, device, use_program_cache):
    torch.manual_seed(2024)

    dim, rtol, atol = dim_rtol_atol

    for _ in range(2):
        run_moreh_norm(input_shape, p, dim, rtol, atol, device)


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
        [TILE_HEIGHT, TILE_WIDTH],
        [2, 2, 2 * TILE_HEIGHT + 13, 2 * TILE_WIDTH + 13],
    ],
)
@pytest.mark.parametrize("keepdim", [True, False], ids=["keepdim-true", "keepdim-flase"])
def test_moreh_norm_backward(input_shape, p, dim_rtol_atol, keepdim, device):
    torch.manual_seed(2024)

    dim, rtol, atol = dim_rtol_atol

    run_moreh_norm_backward(input_shape, p, dim, rtol, atol, device, keepdim=keepdim)


@pytest.mark.parametrize("p", [2.0], ids=["p=2.0"])
@pytest.mark.parametrize(
    "dim_rtol_atol",
    [
        [[], 0.2, 0.2],
    ],
)
@pytest.mark.parametrize(
    "input_shape",
    [[2, 2], [32, 2], [2, 32], [32, 32]],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_norm_backward_compute_kernel_options(input_shape, p, dim_rtol_atol, compute_kernel_options, device):
    torch.manual_seed(2024)

    dim, rtol, atol = dim_rtol_atol

    run_moreh_norm_backward(input_shape, p, dim, rtol, atol, device, compute_kernel_options=compute_kernel_options)


@pytest.mark.parametrize("p", [1.5], ids=["p=2.0"])
@pytest.mark.parametrize(
    "dim_rtol_atol",
    [
        [[], 0.2, 0.2],
    ],
)
@pytest.mark.parametrize(
    "input_shape",
    [
        [4, 4],
    ],
)
def test_moreh_norm_backward_callback(input_shape, p, dim_rtol_atol, device, use_program_cache):
    torch.manual_seed(2024)

    dim, rtol, atol = dim_rtol_atol

    for _ in range(2):
        run_moreh_norm_backward(input_shape, p, dim, rtol, atol, device)
