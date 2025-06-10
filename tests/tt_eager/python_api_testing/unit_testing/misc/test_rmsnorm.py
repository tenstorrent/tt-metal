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
    (0, 1, pytest.param(2, marks=pytest.mark.xfail(reason="GH Issue #22069"))),
    ids=["RMSN", "RMSN_G", "RMSN_GB"],
)
def test_rmsnorm_test(test_id, dtype, in0_mem_config, out_mem_config, device):
    run_rmsnorm_tests(test_id, dtype, in0_mem_config, out_mem_config, device)


@pytest.mark.parametrize("h", [128, 1024, 8192, 65536])
@pytest.mark.parametrize("w", [2048, 3072, 4096])
def test_llama_4D_rms_norm(device, h, w):
    """
    Llama rms input shape: [1, 1, seqlen, hidden_dim]
    Llama weight shape: [1, 1, hidden_dim/32, 32]
    Hidden dims for Llama: {1B:2048, 3B:3072, 8B:4096}
    """
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((1, 1, 1, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn.rms_norm)
    torch_output_tensor = golden_function(torch_input_tensor, torch_weight)

    input_tensor = ttnn.from_torch(torch_input_tensor, device=device, layout=ttnn.TILE_LAYOUT)
    weight = ttnn.from_torch(torch_weight.reshape(1, 1, w // 32, 32), device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.rms_norm(input_tensor, weight=weight)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    is_close(torch_output_tensor, output_tensor)
