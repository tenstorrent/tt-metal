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
    is_close,
)
from models.utility_functions import is_wormhole_b0


def ref_layernorm(x, eps, gamma, beta, H, W):
    lnorm = torch.nn.LayerNorm((W,), eps)
    lnorm.weight = torch.nn.Parameter(torch.full((W,), gamma))
    lnorm.bias = torch.nn.Parameter(torch.full((W,), beta))
    return lnorm(x)


def run_layernorm_tests(test_id, dtype, in0_mem_config, out_mem_config, device):
    torch.manual_seed(1234)

    tensor = ttl.tensor
    dev = device

    epsf = 1e-2

    test_dims = ((1, 9, 384, 1024),)
    for N, C, H, W in test_dims:
        """
        test_id = 0  : ln(x)*1+0 path
        test_id = 1  : ln(x)*g+0 path
        test_id = 2  : ln(x)*gamma+beta path
        test_id = 3  : ln(a+b)*gamma+beta path
        """
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
            ttz = tensor.layernorm(ttx, epsf, output_mem_config=out_mem_config)
        elif test_id == 1:
            logger.info("Running LN_G")
            ttz = tensor.layernorm(ttx, epsf, ttgamma, output_mem_config=out_mem_config)
        elif test_id == 2:
            logger.info("Running LN_GB")
            ttz = tensor.layernorm(ttx, epsf, ttgamma, ttbeta, out_mem_config)
        elif test_id == 3:
            logger.info("Running add_LN_GB")
            ttz = tensor.add_layernorm(ttx, tty, epsf, ttgamma, ttbeta, out_mem_config)
        else:
            assert False
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
        ref_lnorm = torch.nn.functional.layer_norm(x + y, x.shape[-1:], gamma.flatten(), beta.flatten(), epsf)

        assert is_close(tt_got_back, ref_lnorm)


@pytest.mark.parametrize(
    "out_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=["in0_DRAM", "in0_L1"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT16,),
    ids=["BFLOAT16"],
)
@pytest.mark.parametrize(
    "test_id",
    (0, 1, 2, 3),
    ids=["LN", "LN_G", "LN_GB", "add_LN_GB"],
)
def test_layernorm_test(test_id, dtype, in0_mem_config, out_mem_config, device):
    run_layernorm_tests(test_id, dtype, in0_mem_config, out_mem_config, device)


def run_layernorm_mix_precision_tests(test_id, in_dtype, out_dtype, cb_dtype, in0_mem_config, out_mem_config, device):
    tensor = ttl.tensor
    dev = device

    epsf = 1e-2

    test_dims = ((1, 9, 384, 1024),)
    for N, C, H, W in test_dims:
        """
        test_id = 0  : ln(x)*1+0 path
        test_id = 1  : ln(x)*g+0 path
        test_id = 2  : ln(x)*gamma+beta path
        test_id = 3  : ln(a+b)*gamma+beta path
        """
        if test_id >= 0:
            gamma = torch.ones(1, 1, 1, W)
            beta = torch.zeros(1, 1, 1, W)
        if test_id >= 1:
            gamma = torch.rand(1, 1, 1, W) * 2 - 1
            gammah32 = tilize_to_list(pad_weight(gamma))
            ttgamma = tensor.Tensor(
                gammah32,
                [1, 1, 32, W],
                cb_dtype,
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
                cb_dtype,
                tensor.Layout.TILE,
                dev,
                in0_mem_config,
            )

        x = torch.rand((N, C, H, W)) * 2 - 0.95
        y = torch.rand((N, C, H, W)) * 2 - 0.8

        ttx = tensor.Tensor(
            tilize_to_list(x),
            [N, C, H, W],
            in_dtype,
            tensor.Layout.TILE,
            dev,
            in0_mem_config,
        )
        tty = tensor.Tensor(
            tilize_to_list(y),
            [N, C, H, W],
            in_dtype,
            tensor.Layout.TILE,
            dev,
            in0_mem_config,
        )

        if test_id == 0:
            logger.info("Running add_LN_NOGB")
            ttz = tensor.add_layernorm(ttx, tty, epsf, output_mem_config=out_mem_config)
        elif test_id == 1:
            logger.info("Running add_LN_G")
            ttz = tensor.add_layernorm(ttx, tty, epsf, ttgamma, output_mem_config=out_mem_config)
        elif test_id == 2:
            logger.info("Running add_LN_GB")
            ttz = tensor.add_layernorm(ttx, tty, epsf, ttgamma, ttbeta, out_mem_config)
        else:
            assert False
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

        ref_lnorm = torch.nn.functional.layer_norm(x + y, x.shape[-1:], gamma.flatten(), beta.flatten(), epsf)

        assert is_close(tt_got_back, ref_lnorm)


@pytest.mark.parametrize(
    "out_mem_config",
    (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),),
    ids=[
        "in0_L1",
    ],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),),
    ids=[
        "in0_L1",
    ],
)
@pytest.mark.parametrize(
    "out_dtype",
    (
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.DataType.BFLOAT8_B,
    ),
    ids=["BFLOAT16", "BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "cb_dtype",
    (ttl.tensor.DataType.BFLOAT16,),
    ids=["BFLOAT16"],
)
@pytest.mark.parametrize(
    "in_dtype",
    (
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.DataType.BFLOAT8_B,
    ),
    ids=["BFLOAT16", "BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "test_id",
    (0, 1, 2),
    ids=[
        "add_LN",
        "add_LN_G",
        "add_LN_GB",
    ],
)
def test_layernorm_mix_precision(test_id, in_dtype, out_dtype, cb_dtype, in0_mem_config, out_mem_config, device):
    run_layernorm_mix_precision_tests(test_id, in_dtype, out_dtype, cb_dtype, in0_mem_config, out_mem_config, device)
