# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from loguru import logger
import pytest


import torch

import tt_lib as ttl

from tt_lib.utils import (
    pad_weight,
    tilize_to_list,
    untilize,
)


def ref_groupnorm(x, group_size, eps, **kwargs):
    n_channels = x.shape[1]
    lnorm = torch.nn.GroupNorm(group_size, n_channels, eps, **kwargs)
    return lnorm(x)


def run_groupnorm_tests(test_id, group_size, dtype, in0_mem_config, out_mem_config, device):
    torch.manual_seed(1234)

    tensor = ttl.tensor
    dev = device

    epsf = 1e-2

    test_dims = ((1, 32, 32, 64),)  # 384, 1024),)
    for N, C, H, W in test_dims:
        """
        test_id = 0  : ln(x)*1+0 path
        test_id = 1  : ln(x)*g+0 path
        test_id = 2  : ln(x)*gamma+beta path
        test_id = 3  : ln(a+b)*gamma+beta path
        """
        for nrepeat in range(0, 1):
            if test_id >= 0:
                gamma = torch.ones(1, 1, 1, W)
                beta = torch.zeros(1, 1, 1, W)
            if test_id >= 1:
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
            if test_id >= 2:
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

            if test_id < 3:
                y *= 0.0  # zero out the y to exclude x+y from reference calculation

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

            if test_id == 0:
                logger.info("Running LN_NOGB")
                ttz = tensor.groupnorm(ttx, group_size, epsf, output_mem_config=out_mem_config)
                golden = ref_groupnorm(x, group_size, epsf)
            elif test_id == 1:
                logger.info("Running LN_G")
                ttz = tensor.groupnorm(ttx, group_size, epsf, ttgamma, output_mem_config=out_mem_config)
                golden = ref_groupnorm(x, group_size, epsf, gamma=ttgamma)
            elif test_id == 2:
                logger.info("Running LN_GB")
                ttz = tensor.groupnorm(ttx, group_size, epsf, ttgamma, ttbeta, out_mem_config)
                golden = ref_groupnorm(x, group_size, epsf, gamma=ttgamma, beta=ttbeta)
            else:
                assert False
            logger.info("Done")

            assert ttx.memory_config().buffer_type == in0_mem_config.buffer_type
            assert tty.memory_config().buffer_type == in0_mem_config.buffer_type

            logger.debug(f"ttx is on: {ttx.memory_config().buffer_type}")
            logger.debug(f"tty is on: {tty.memory_config().buffer_type}")

            tt_got_back = ttz.cpu().to_torch()
            tt_got_back = untilize(tt_got_back)

            torch.isclose(golden, tt_got_back)


@pytest.mark.parametrize(
    "out_mem_config",
    (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),),
    ids=["out_DRAM"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),),
    ids=["in0_DRAM"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT16,),
    ids=["BFLOAT16"],
)
@pytest.mark.parametrize(
    "test_id",
    (0,),
    ids=[
        "GN",
    ],
)
def test_groupnorm_test(test_id, dtype, in0_mem_config, out_mem_config, device):
    group_size = 1
    run_groupnorm_tests(test_id, group_size, dtype, in0_mem_config, out_mem_config, device)
