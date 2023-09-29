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
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.common import is_wormhole_b0, skip_for_wormhole_b0


# This ref implementation is only here for debugging

def ref_eltwise_lerp_ternary(x, y, z):
    return torch.lerp(x, y, z)


def run_eltwise_lerp_ternary_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    torch.manual_seed(data_seed)

    # Initialize the device
    tensor = ttl.tensor
    dev = device

    test_dims = (input_shape,)

    for N, C, H, W in test_dims:
        for nrepeat in range(0, 100):
            x = torch.Tensor(size=(N, C, H, W)).uniform_(-100, 100)
            x_ref = x

            if dlayout == ttl.tensor.Layout.TILE:
                x = tilize_to_list(x)
            else:
                x = x.reshape(-1).tolist()

            if in_mem_config == "SYSTEM_MEMORY":
                ttx = tensor.Tensor(
                    x,
                    [N, C, H, W],
                    dtype,
                    dlayout,
                    dev,
                ).cpu()
            else:
                ttx = tensor.Tensor(
                    x,
                    [N, C, H, W],
                    dtype,
                    dlayout,
                    dev,
                    in_mem_config,
                )


            y = torch.Tensor(size=(N, C, H, W)).uniform_(-100, 100)
            y_ref = y

            if dlayout == ttl.tensor.Layout.TILE:
                y = tilize_to_list(y)
            else:
                y = y.reshape(-1).tolist()

            if in_mem_config == "SYSTEM_MEMORY":
                tty = tensor.Tensor(
                    y,
                    [N, C, H, W],
                    dtype,
                    dlayout,
                    dev,
                ).cpu()
            else:
                tty = tensor.Tensor(
                    y,
                    [N, C, H, W],
                    dtype,
                    dlayout,
                    dev,
                    in_mem_config,
                )


            z = torch.Tensor(size=(N, C, H, W)).uniform_(-100, 100)
            z_ref = z

            if dlayout == ttl.tensor.Layout.TILE:
                z = tilize_to_list(z)
            else:
                z = z.reshape(-1).tolist()

            if in_mem_config == "SYSTEM_MEMORY":
                ttz = tensor.Tensor(
                    z,
                    [N, C, H, W],
                    dtype,
                    dlayout,
                    dev,
                ).cpu()
            else:
                ttz = tensor.Tensor(
                    z,
                    [N, C, H, W],
                    dtype,
                    dlayout,
                    dev,
                    in_mem_config,
                )

            logger.info("Running Eltwise lerp ternary test")
            ttw = tensor.lerp(ttx, tty, ttz, output_mem_config=out_mem_config)

            logger.info("Done")

            if in_mem_config != "SYSTEM_MEMORY":
                assert ttx.memory_config().buffer_type == in_mem_config.buffer_type
                logger.debug(f"ttx is on: {ttx.memory_config().buffer_type}")
            else:
                logger.debug(f"ttx is on: SYSTEM_MEMORY")


            if in_mem_config != "SYSTEM_MEMORY":
                assert tty.memory_config().buffer_type == in_mem_config.buffer_type
                logger.debug(f"tty is on: {tty.memory_config().buffer_type}")
            else:
                logger.debug(f"tty is on: SYSTEM_MEMORY")

            if in_mem_config != "SYSTEM_MEMORY":
                assert ttz.memory_config().buffer_type == in_mem_config.buffer_type
                logger.debug(f"ttz is on: {ttz.memory_config().buffer_type}")
            else:
                logger.debug(f"ttz is on: SYSTEM_MEMORY")


            assert ttw.memory_config().buffer_type == out_mem_config.buffer_type
            logger.debug(f"ttw is on: {ttw.memory_config().buffer_type}")

            t2_data = ttw.cpu().to_torch()

            tt_got_back = torch.Tensor(t2_data).reshape((N, C, H, W))

            if dlayout == ttl.tensor.Layout.TILE:
                tt_got_back = untilize(tt_got_back)

            # get referent value
            ref_value = ref_eltwise_lerp_ternary(x_ref, y_ref, z_ref)

            # compare tt and golden outputs
            success, pcc_value = comp_pcc(tt_got_back, ref_value)
            logger.debug(pcc_value)

            assert success


test_sweep_args=[
    ((4, 11, 106, 232), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 19096254),
    ((2, 10, 160, 160), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, "SYSTEM_MEMORY", ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1), 3074662),
    ((1, 6, 256, 160), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 16417740),
    ((6, 7, 192, 224), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, "SYSTEM_MEMORY", ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), 11178160),
]

@skip_for_wormhole_b0
@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (
        test_sweep_args
    ),
)

def test_eltwise_lerp_ternary_test(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device
):
    run_eltwise_lerp_ternary_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
