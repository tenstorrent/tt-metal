# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import pytest
import torch

import ttnn


from tt_lib.utils import (
    pad_weight,
    tilize_to_list,
    untilize,
    is_close,
)
from models.utility_functions import is_wormhole_b0


def rmsnorm(x, gamma, beta, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma + beta


def run_rmsnorm_tests(test_id, dtype, in0_mem_config, out_mem_config, device):
    torch.manual_seed(1234)

    dev = device

    epsf = 1e-2

    test_dims = ((1, 9, 384, 1024),)
    for N, C, H, W in test_dims:
        """
        test_id = 0  : rmsn(x)*1+0 path
        test_id = 1  : rmsn(x)*g+0 path
        test_id = 2  : rmsn(x)*gamma+beta path
        """
        if test_id >= 0:
            gamma = torch.ones(1, 1, 1, W)
            beta = torch.zeros(1, 1, 1, W)
        if test_id >= 1:
            gamma = torch.rand(1, 1, 1, W) * 2 - 1
            gammah32 = tilize_to_list(pad_weight(gamma))
            ttgamma = ttnn.Tensor(
                gammah32,
                [1, 1, 32, W],
                dtype,
                ttnn.TILE_LAYOUT,
                dev,
                in0_mem_config,
            )
        if test_id >= 2:
            beta = torch.rand(1, 1, 1, W) * 2.0 - 1.1
            betah32 = tilize_to_list(pad_weight(beta))
            ttbeta = ttnn.Tensor(
                betah32,
                [1, 1, 32, W],
                dtype,
                ttnn.TILE_LAYOUT,
                dev,
                in0_mem_config,
            )

        x = torch.rand((N, C, H, W)) * 2 - 0.95

        ttx = ttnn.Tensor(
            tilize_to_list(x),
            [N, C, H, W],
            dtype,
            ttnn.TILE_LAYOUT,
            dev,
            in0_mem_config,
        )

        if test_id == 0:
            logger.info("Running RMSN_NOGB")
            ttz = ttnn.rms_norm(ttx, epsilon=epsf, memory_config=out_mem_config)
        elif test_id == 1:
            logger.info("Running RMSN_G")
            ttz = ttnn.rms_norm(ttx, epsilon=epsf, weight=ttgamma, memory_config=out_mem_config)
        elif test_id == 2:
            logger.info("Running RMSN_GB")
            ttz = ttnn.rms_norm(ttx, epsilon=epsf, weight=ttgamma, bias=ttbeta, memory_config=out_mem_config)
        else:
            assert False
        logger.info("Done")

        assert ttx.memory_config().buffer_type == in0_mem_config.buffer_type
        assert ttz.memory_config().buffer_type == out_mem_config.buffer_type

        logger.debug(f"ttx is on: {ttx.memory_config().buffer_type}")
        logger.debug(f"ttz is on: {ttz.memory_config().buffer_type}")

        t2_data = ttz.cpu().to_torch()

        tt_got_back = torch.Tensor(t2_data).reshape((N, C, H, W))
        tt_got_back = untilize(tt_got_back)

        # ref_lnorm = ref_layernorm(x, epsf, gammaf, betaf, H, W)
        ref_rmsnorm = rmsnorm(x, gamma.flatten(), beta.flatten(), epsf)

        passing = is_close(tt_got_back, ref_rmsnorm)
        assert passing


@pytest.mark.parametrize(
    "out_mem_config",
    (
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
    ),
    ids=["in0_DRAM", "in0_L1"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16,),
    ids=["BFLOAT16"],
)
@pytest.mark.parametrize(
    "test_id",
    (0, 1, 2),
    ids=["RMSN", "RMSN_G", "RMSN_GB"],
)
def test_rmsnorm_test(test_id, dtype, in0_mem_config, out_mem_config, device):
    run_rmsnorm_tests(test_id, dtype, in0_mem_config, out_mem_config, device)
