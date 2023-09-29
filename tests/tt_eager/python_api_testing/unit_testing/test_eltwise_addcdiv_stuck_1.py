# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys
import random
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../../../..")

import pytest
import torch

import tt_lib as ttl
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.common import skip_for_wormhole_b0

from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import eltwise_addcdiv as tt_eltwise_addcdiv



def run_addcdiv(input_shape, dtype, dlayout, buffer_type, output_mem_config, data_seed, scalar, device):
    random.seed(0)
    torch.manual_seed(data_seed)

    x = torch.Tensor(size=input_shape).uniform_(-100, 100)
    y = torch.Tensor(size=input_shape).uniform_(-100, 100)
    z = torch.Tensor(size=input_shape).uniform_(-100, 100)

    # ref_value = torch.addcdiv(x, y, z, value=scalar)
    logger.info(f"Running addcdiv with input_shape {input_shape} dtype {dtype} dlayout {dlayout} buffer_type {buffer_type} output_mem_config {output_mem_config} scalar {scalar} data_seed {data_seed}")

    try:
        ttz = tt_eltwise_addcdiv(
            x=x,
            y=y,
            z=z,
            scalar=scalar,
            device=device,
            device_id=0,
            dtype=dtype,
            layout=dlayout,
            buffer_type=buffer_type,
            output_mem_config=output_mem_config)
    except Exception as exc:
        logger.warning(f"run_addcdiv RuntimeError occured {exc}")
        assert False

    logger.info(f"Finished running addcdiv")


test_sweep_args=[
    ((3, 10, 73, 388),
     [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16],
     [ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.Layout.ROW_MAJOR],
     [ttl.tensor.BufferType.DRAM, ttl.tensor.BufferType.DRAM, ttl.tensor.BufferType.L1],
     ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1), 8405597, -61.75),
    ((9, 23, 416, 310),
     [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16],
     [ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.Layout.ROW_MAJOR],
     [ttl.tensor.BufferType.DRAM, ttl.tensor.BufferType.L1, ttl.tensor.BufferType.DRAM],
     ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1), 10406825, -42.25),
    ((2, 24, 39, 462),
     [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16],
     [ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.Layout.ROW_MAJOR],
     [ttl.tensor.BufferType.L1, ttl.tensor.BufferType.DRAM, ttl.tensor.BufferType.L1],
     ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1), 10406825, -42.25),
]

@skip_for_wormhole_b0
@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, buffer_type, output_mem_config, data_seed, scalar",
    (
        test_sweep_args
    ),
)
@skip_for_wormhole_b0
def test_addcdiv_test(
    input_shape, dtype, dlayout, buffer_type, output_mem_config, data_seed, scalar
):
    run_addcdiv(input_shape, dtype, dlayout, buffer_type, output_mem_config, data_seed, scalar, device=None)
