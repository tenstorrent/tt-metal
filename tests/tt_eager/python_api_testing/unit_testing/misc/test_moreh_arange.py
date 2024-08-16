# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
import pytest

from models.utility_functions import comp_allclose_and_pcc
from loguru import logger


def get_tt_dtype(torch_dtype):
    if torch_dtype == torch.int32:
        return ttnn.int32
    if torch_dtype == torch.bfloat16:
        return ttnn.bfloat16
    if torch_dtype == torch.float32:
        return ttnn.float32
    return None


@pytest.mark.parametrize(
    "start_end_step",
    (
        (-5, 27, 1),  # simple
        (2.3, 15.3, 0.5),  # floating point
        (10, 0, -0.3),  # minus step
        (10, 32 * 3, 1),  # multiple cores
    ),
)
def test_arange_row_major_simple(start_end_step, device):
    start, end, step = start_end_step

    tt_cpu = torch.arange(start=start, end=end, step=step).to(torch.bfloat16)

    any_cpu = torch.ones((1024))
    any = ttnn.Tensor(any_cpu, ttnn.bfloat16).to(device)

    untilize_out = True
    tt_npu = ttnn.experimental.operations.primary.moreh_arange(start, end, step, any, None, untilize_out)

    tt_dev = tt_npu.cpu().to_torch()

    assert tt_dev.shape == tt_cpu.shape

    rtol = atol = 0.1
    passing, out = comp_allclose_and_pcc(tt_cpu, tt_dev, rtol=rtol, atol=atol)
    logger.info(out)

    assert passing


@pytest.mark.parametrize(
    "start_end_step",
    ((0, 32 * 10, 1),),  # simple
)
@pytest.mark.parametrize(
    "optional_output",
    (
        True,
        False,
    ),
)
def test_arange_row_major_optioanl_output(start_end_step, optional_output, device):
    start, end, step = start_end_step

    tt_cpu = torch.arange(start=start, end=end, step=step).to(torch.bfloat16)

    any_cpu = torch.ones((1024))
    any = ttnn.Tensor(any_cpu, ttnn.bfloat16).to(device)

    untilize_out = True
    if optional_output:
        output_cpu = torch.empty_like(tt_cpu)
        output = ttnn.Tensor(output_cpu, ttnn.bfloat16).to(device)
        tt_npu = ttnn.experimental.operations.primary.moreh_arange(start, end, step, any, output, untilize_out)
    else:
        tt_npu = ttnn.experimental.operations.primary.moreh_arange(start, end, step, any, None, untilize_out)

    tt_dev = tt_npu.cpu().to_torch()

    assert tt_dev.shape == tt_cpu.shape

    rtol = atol = 0.1
    passing, out = comp_allclose_and_pcc(tt_cpu, tt_dev, rtol=rtol, atol=atol)
    logger.info(out)

    assert passing


@pytest.mark.parametrize(
    "start_end_step",
    ((-10, 22, 1),),  # simple
)
@pytest.mark.parametrize(
    "output_dtype",
    (
        torch.bfloat16,
        torch.int32,
        torch.float32,
    ),
    ids=["bfloat16", "int32", "float32"],
)
def test_arange_row_major_dtype(start_end_step, output_dtype, device):
    start, end, step = start_end_step

    tt_dtype = get_tt_dtype(output_dtype)

    tt_cpu = torch.arange(start=start, end=end, step=step).to(output_dtype)

    any_cpu = torch.ones((1024))
    any = ttnn.Tensor(any_cpu, tt_dtype).to(device)

    untilize_out = True
    tt_npu = ttnn.experimental.operations.primary.moreh_arange(start, end, step, any, None, untilize_out, tt_dtype)

    tt_dev = tt_npu.cpu().to_torch()

    assert tt_dev.shape == tt_cpu.shape

    rtol = atol = 0.1
    passing, out = comp_allclose_and_pcc(tt_cpu, tt_dev, rtol=rtol, atol=atol)
    logger.info(out)

    assert passing


@pytest.mark.parametrize(
    "start_end_step",
    (
        (0, 32, 1),  # simple
        (2.3, 15.7, 0.5),  # floating point
        (10, 0, -0.3),  # minus step
        (10, 32 * 3, 1),  # multiple cores
    ),
)
def test_arange_tilized_simple(start_end_step, device):
    start, end, step = start_end_step

    tt_cpu = torch.arange(start=start, end=end, step=step).to(torch.bfloat16)

    any_cpu = torch.ones((1024))
    any = ttnn.Tensor(any_cpu).to(device)
    tt_npu = ttnn.experimental.operations.primary.moreh_arange(start, end, step, any)

    L = tt_cpu.shape[0]

    tt_dev = tt_npu.cpu().to(ttnn.ROW_MAJOR_LAYOUT).unpad_from_tile((1, L)).to_torch().reshape((L)).to(torch.bfloat16)

    rtol = atol = 0.1
    passing, out = comp_allclose_and_pcc(tt_cpu, tt_dev, rtol=rtol, atol=atol)
    logger.info(out)
    assert passing


@pytest.mark.parametrize(
    "start_end_step",
    ((0, 32 * 10, 1),),  # simple
)
@pytest.mark.parametrize(
    "optional_output",
    (
        True,
        False,
    ),
)
def test_arange_tilized_major_optioanl_output(start_end_step, optional_output, device):
    start, end, step = start_end_step

    tt_cpu = torch.arange(start=start, end=end, step=step).to(torch.bfloat16)

    L = tt_cpu.shape[0]

    any_cpu = torch.ones((1024))
    any = ttnn.Tensor(any_cpu, ttnn.bfloat16).to(device)

    untilize_out = False
    if optional_output:
        output_cpu = torch.empty_like(tt_cpu)
        output = (
            ttnn.Tensor(output_cpu, ttnn.bfloat16)
            .reshape([1, L])
            .pad_to_tile(float("nan"))
            .to(ttnn.TILE_LAYOUT)
            .to(device)
        )
        tt_npu = ttnn.experimental.operations.primary.moreh_arange(start, end, step, any, output, untilize_out)
    else:
        tt_npu = ttnn.experimental.operations.primary.moreh_arange(start, end, step, any, None, untilize_out)

    tt_dev = tt_npu.cpu().to_torch()

    tt_dev = tt_npu.cpu().to(ttnn.ROW_MAJOR_LAYOUT).unpad_from_tile((1, L)).to_torch().reshape((L)).to(torch.bfloat16)

    rtol = atol = 0.1
    passing, out = comp_allclose_and_pcc(tt_cpu, tt_dev, rtol=rtol, atol=atol)
    logger.info(out)

    assert passing


@pytest.mark.parametrize(
    "start_end_step",
    ((-10, 57, 1),),  # simple
)
@pytest.mark.parametrize(
    "output_dtype",
    (
        torch.bfloat16,
        torch.int32,
        torch.float32,
    ),
    ids=["bfloat16", "int32", "float32"],
)
def test_arange_tilized_dtype(start_end_step, output_dtype, device):
    start, end, step = start_end_step

    tt_dtype = get_tt_dtype(output_dtype)

    tt_cpu = torch.arange(start=start, end=end, step=step).to(output_dtype)

    any_cpu = torch.ones((1024))
    any = ttnn.Tensor(any_cpu, tt_dtype).to(device)

    untilize_out = False
    tt_npu = ttnn.experimental.operations.primary.moreh_arange(start, end, step, any, None, untilize_out, tt_dtype)

    tt_dev = tt_npu.cpu().to_torch()

    L = tt_cpu.shape[0]

    tt_dev = tt_npu.cpu().to(ttnn.ROW_MAJOR_LAYOUT).unpad_from_tile((1, L)).to_torch().reshape((L)).to(output_dtype)

    rtol = atol = 0.1
    passing, out = comp_allclose_and_pcc(tt_cpu, tt_dev, rtol=rtol, atol=atol)
    logger.info(out)

    assert passing
