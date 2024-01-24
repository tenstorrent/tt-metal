# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import tt_lib as ttl
import pytest

from models.utility_functions import comp_allclose_and_pcc
from loguru import logger


@pytest.mark.parametrize(
    "start_end_step",
    (
        (0, 32, 1),  # simple
        (2.3, 15.7, 0.5),  # floating point
        (10, 0, -0.3),  # minus step
        (10, 32 * 3, 1),  # multiple cores
    ),
)
def test_arange_simple(start_end_step, device):
    start, end, step = start_end_step

    tt_cpu = torch.arange(start=start, end=end, step=step).to(torch.bfloat16)

    L = tt_cpu.shape[0]

    xt = (
        (
            ttl.tensor.Tensor(
                tt_cpu.reshape(-1).tolist(),
                tt_cpu.reshape(1, 1, 1, L).shape,
                ttl.tensor.DataType.BFLOAT16,
                ttl.tensor.Layout.ROW_MAJOR,
            )
        )
        .pad_to_tile(float("nan"))
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    tt_npu = ttl.operations.primary.moreh_arange(start, end, step, xt)
    tt_dev = (
        tt_npu.cpu()
        .to(ttl.tensor.Layout.ROW_MAJOR)
        .unpad_from_tile((1, 1, 1, L))
        .to_torch()
        .reshape((L))
        .to(torch.bfloat16)
    )

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
def test_arange_inplace_simple(start_end_step, device):
    start, end, step = start_end_step

    tt_cpu = torch.arange(start=start, end=end, step=step).to(torch.bfloat16)

    L = tt_cpu.shape[0]

    tt_npu = (
        (
            ttl.tensor.Tensor(
                tt_cpu.reshape(-1).tolist(),
                tt_cpu.reshape(1, 1, 1, L).shape,
                ttl.tensor.DataType.BFLOAT16,
                ttl.tensor.Layout.ROW_MAJOR,
            )
        )
        .pad_to_tile(float("nan"))
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    ttl.operations.primary.moreh_arange_inplace(tt_npu, start, end, step)
    tt_dev = (
        tt_npu.cpu()
        .to(ttl.tensor.Layout.ROW_MAJOR)
        .unpad_from_tile((1, 1, 1, L))
        .to_torch()
        .reshape((L))
        .to(torch.bfloat16)
    )

    assert tt_dev.shape == tt_cpu.shape

    rtol = atol = 0.1
    passing, out = comp_allclose_and_pcc(tt_cpu, tt_dev, rtol=rtol, atol=atol)
    logger.info(out)
    assert passing
