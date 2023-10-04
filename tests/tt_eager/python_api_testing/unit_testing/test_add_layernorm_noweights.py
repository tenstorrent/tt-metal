# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
import sys
import time
import os
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../../../..")

import pytest
import torch

import tt_lib as ttl

from tt_lib.utils import (
    pad_weight,
    tilize_to_list,
    untilize,
    is_close,
)
from tests.tt_eager.python_api_testing.sweep_tests.common import is_wormhole_b0, skip_for_wormhole_b0

# This ref implementation is only here for debugging
def ref_ln(x, gamma, beta=None, epsilon=1e-5, b=None):
    # prints a tile slice with these tile coord range
    torch.set_printoptions(
        precision=2, threshold=1000, sci_mode=False, edgeitems=8, linewidth=480
    )
    sth = 16
    stw = 16
    ph0, ph1, pw0, pw1 = 1, 2, 0, 1
    # ph0, ph1, pw0, pw1 = 0, 1, 0, 1

    # print("x.shape=", x.shape)
    # print("eps=", epsilon)
    # print(f"slice={ph0}:{ph1}, {pw0}:{pw1}")
    # print("Ref x=\n", x[0, 0, ph0 * 32 : ph1 * 32 : sth, pw0 * 32 : pw1 * 32 : stw])
    # print(
    #     "Ref a=\n", (x - b)[0, 0, ph0 * 32 : ph1 * 32 : sth, pw0 * 32 : pw1 * 32 : stw]
    # )
    # print("Ref b=\n", b[0, 0, ph0 * 32 : ph1 * 32 : sth, pw0 * 32 : pw1 * 32 : stw])
    mean = x.mean(dim=-1, keepdim=True)
    # print("Ref Ex=\n", mean[0, 0, ph0*32 : ph1*32 : sth, 0*32 : 1*32 : stw])
    xmm = x - mean
    # print("Ref xmm=\n", xmm[0, 0, ph0*32 : ph1*32 : st, pw0*32 : pw1*32 : st])
    xmm2 = xmm**2
    # print("Ref xmm2=\n", xmm2[0, 0, ph0*32 : ph1*32 : sth, pw0*32 : pw1*32 : stw])
    exmm2 = xmm2.mean(dim=-1, keepdim=True)
    # print("Ref exmm2=\n", exmm2[0, 0, ph0*32 : ph1*32 : sth, 0*32 : 1*32 : stw])

    std = (exmm2 + epsilon).sqrt()
    # print("Ref sqrt_exmm2=\n", std[0, 0, ph0*32 : ph1*32 : st, 0*32 : 1*32 : st])

    invstd = 1.0 / std
    # print("Ref 1/sqrt_exmm2=\n", invstd[0, 0, ph0*32 : ph1*32 : st, 0*32 : 1*32 : st])
    y1 = xmm * invstd
    # print("Ref y*1+0=\n", y1[0, 0, ph0*32 : ph1*32 : st, pw0*32 : pw1*32 : st])
    y = y1.clone()
    if gamma is not None:
        # print("yshape=", y.shape)
        # print("gshape=", gamma.shape)
        y *= gamma
    if beta is not None:
        y += beta
    # y = gamma.repeat(x.shape[0], x.shape[1], x.shape[2], x.shape[3]//gamma.shape[3]) # Debug gamma
    return y, mean, exmm2, std, invstd, y1


def ref_layernorm(x, eps, gamma, beta, H, W):
    lnorm = torch.nn.LayerNorm((W,), eps)
    lnorm.weight = torch.nn.Parameter(torch.full((W,), gamma))
    lnorm.bias = torch.nn.Parameter(torch.full((W,), beta))
    return lnorm(x)


def run_layernorm_tests(dtype, in0_mem_config, out_mem_config, device):
    torch.manual_seed(1234)


    tensor = ttl.tensor
    dev = device

    epsf = 1e-2

    test_dims = ((1, 9, 384, 1024),)
    for N, C, H, W in test_dims:

        for nrepeat in range(0, 1):
            gamma = torch.ones(1, 1, 1, W)
            beta = torch.zeros(1, 1, 1, W)
            gamma = torch.rand(1, 1, 1, W) * 2 - 1
            gammah32 = tilize_to_list(pad_weight(gamma))
            ttgamma = tensor.Tensor(
                gammah32,
                [1, 1, 32, W],
                dtype,
                tensor.Layout.TILE,
                dev,
                in0_mem_config,
            )

            beta = torch.rand(1, 1, 1, W) * 2.0 - 1.1
            betah32 = tilize_to_list(pad_weight(beta))
            ttbeta = tensor.Tensor(
                betah32,
                [1, 1, 32, W],
                dtype,
                tensor.Layout.TILE,
                dev,
                in0_mem_config,
            )

            x = torch.rand((N, C, H, W)) * 2 - 0.95
            y = torch.rand((N, C, H, W)) * 2 - 0.8

            ttx = tensor.Tensor(
                tilize_to_list(x),
                [N, C, H, W],
                dtype,
                tensor.Layout.TILE,
                dev,
                in0_mem_config,
            )
            tty = tensor.Tensor(
                tilize_to_list(y),
                [N, C, H, W],
                dtype,
                tensor.Layout.TILE,
                dev,
                in0_mem_config,
            )

            logger.info("Running add_LN_GB")
            ttz = tensor.add_layernorm(
                ttx, tty, epsf, None, None, out_mem_config
            )

            logger.info("Done")

            assert ttx.memory_config().buffer_type == in0_mem_config.buffer_type
            assert tty.memory_config().buffer_type == in0_mem_config.buffer_type
            assert ttz.memory_config().buffer_type == out_mem_config.buffer_type

            logger.debug(f"ttx is on: {ttx.memory_config().buffer_type}")
            logger.debug(f"tty is on: {tty.memory_config().buffer_type}")
            logger.debug(f"ttz is on: {ttz.memory_config().buffer_type}")

            t2_data = ttz.cpu().to_torch()

            tt_got_back = torch.Tensor(t2_data).reshape((N, C, H, W))
            tt_got_back = untilize(tt_got_back)

            # ref_lnorm = ref_layernorm(x, epsf, gammaf, betaf, H, W)
            ref_lnorm, _, _, _, _, _ = ref_ln(x + y, gamma, beta, epsf, y)

            time.sleep(0.3)  # sleep to avoid print intermixing with kernel prints

            assert is_close(tt_got_back, ref_lnorm)


@skip_for_wormhole_b0
@pytest.mark.parametrize(
    "out_mem_config",
    (
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    ),
    ids=["in0_DRAM", "in0_L1"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT16,),
    ids=["BFLOAT16"],
)
def test_layernorm_test(
    dtype, in0_mem_config, out_mem_config, request, device
):
    run_layernorm_tests(dtype, in0_mem_config, out_mem_config, device)
