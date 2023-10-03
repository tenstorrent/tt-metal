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
from itertools import permutations
import random

import tt_lib as ttl

from tt_lib.utils import (
    pad_weight,
    tilize_to_list,
    untilize,
    is_close,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.common import is_wormhole_b0, skip_for_wormhole_b0

from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import permute as tt_permute


all_permutations = list(permutations([0, 1, 2, 3]))
in_mememory_configs = [ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM), ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1)]
all_layouts = [ttl.tensor.Layout.TILE, ttl.tensor.Layout.ROW_MAJOR]

def run_permute_tests(
    input_shape,
    dtype,
    out_mem_config,
    device
):
    torch.manual_seed(random.random())

    # Initialize the device
    tensor = ttl.tensor
    dev = device

    test_dims = (input_shape,)

    for N, C, H, W in test_dims:
        for nrepeat in range(0, 100):
            x = torch.Tensor(size=(N, C, H, W)).uniform_(-100, 100)
            x_ref = x.detach().clone()

            one_permutation = list(random.choice(all_permutations))
            in_memory_config = random.choice(in_mememory_configs)
            dlayout = random.choice(all_layouts)

            # calculate tt output
            if in_memory_config == "SYSTEM_MEMORY":
                in_memory_config = None

            logger.info(f"Running PERMUTE test: {dlayout}: {in_memory_config}: {one_permutation}")
            tt_result = tt_permute(
                x=x,
                device=device,
                device_id=0,
                dtype=[dtype],
                layout=[dlayout],
                input_mem_config=[in_memory_config],
                output_mem_config=out_mem_config,
                permute_dims=one_permutation
            )
            logger.info("Done")


test_sweep_args=[
    ((1, 10, 192, 160), ttl.tensor.DataType.BFLOAT16, ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM)),
]

@skip_for_wormhole_b0
@pytest.mark.parametrize(
    "input_shape, dtype, out_mem_config",
    (
        test_sweep_args
    ),
)

def test_permute_test(
    input_shape, dtype, out_mem_config, device
):
    run_permute_tests(input_shape, dtype, out_mem_config, device)
