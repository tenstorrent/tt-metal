# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
import sys
import time
import os
from loguru import logger
from models.utility_functions import torch2tt_tensor, tt2torch_tensor

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
def ref_geglu(x, dim):
    a, b = torch.split(x, x.shape[dim] // 2, 3)
    return a * torch.nn.functional.gelu(b)
    # return torch.matmul(a,torch.nn.functional.gelu(b))

def run_geglu_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
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

            logger.info("Running geglu test")
            ttz = tensor.geglu(ttx, -1, output_mem_config=out_mem_config)

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
            ref_value = ref_geglu(x_ref, 3)


            # compare tt and golden outputs
            success, pcc_value = comp_pcc(output, ref_value)
            logger.debug(pcc_value)
            logger.debug(success)
            #assert success


test_sweep_args=[
    ((1, 1, 32, 64), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR,  (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)),  (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)), 19096254),
    ((1, 1, 128, 192), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR,  (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)),  (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)), 19096254),
    ((1, 1, 64, 128), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR,  (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)),  (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)), 19096254),

]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (
        test_sweep_args
    ),
)

def test_geglu(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device
):
    run_geglu_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
