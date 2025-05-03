# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
import sys
import time
import os
from loguru import logger
from models.utility_functions import torch2tt_tensor, tt2torch_tensor
import random
import pytest
import torch

import ttnn

from tt_lib.utils import (
    pad_weight,
    tilize_to_list,
    untilize,
    is_close,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def ref_rpow(x, factor):
    return torch.pow(x, factor)


def run_rpow_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    torch.manual_seed(data_seed)

    # Initialize the device
    dev = device

    test_dims = (input_shape,)

    for N, C, H, W in test_dims:
        for nrepeat in range(0, 100):
            x = torch.Tensor(size=(N, C, H, W)).uniform_(-100, 100)
            x_ref = x
            factor = random.randint(1, 100)
            if dlayout == ttnn.TILE_LAYOUT:
                x = tilize_to_list(x)
            else:
                x = x.reshape(-1).tolist()

            if in_mem_config == "SYSTEM_MEMORY":
                ttx = ttnn.Tensor(
                    x,
                    [N, C, H, W],
                    dtype,
                    dlayout,
                    dev,
                ).cpu()
            else:
                ttx = ttnn.Tensor(
                    x,
                    [N, C, H, W],
                    dtype,
                    dlayout,
                    dev,
                    in_mem_config,
                )

            logger.info("Running rpow test")
            ttz = ttnn.rpow(ttx, factor, memory_config=out_mem_config)

            logger.info("Done")

            if in_mem_config != "SYSTEM_MEMORY":
                assert ttx.memory_config().buffer_type == in_mem_config.buffer_type
                logger.debug(f"ttx is on: {ttx.memory_config().buffer_type}")
            else:
                logger.debug(f"ttx is on: SYSTEM_MEMORY")

            assert ttz.memory_config().buffer_type == out_mem_config.buffer_type
            logger.debug(f"ttz is on: {ttz.memory_config().buffer_type}")

            output = tt2torch_tensor(ttz)

            # get referent value
            ref_value = ref_rpow(factor, x_ref)

            # compare tt and golden outputs
            success, pcc_value = comp_pcc(output, ref_value)
            logger.debug(pcc_value)
            logger.debug(success)
            assert success


test_sweep_args = [
    (
        (1, 1, 32, 64),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)),
        (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)),
        19096254,
    ),
    (
        (1, 1, 128, 192),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)),
        (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)),
        19096254,
    ),
    (
        (1, 1, 64, 128),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)),
        (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)),
        19096254,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_rpow(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_rpow_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
