# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import random
from loguru import logger
import pytest
import torch
import tt_lib as ttl

from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import setup_tt_tensor
from models.utility_functions import tt2torch_tensor


def run_addcdiv(input_shape, dtype, dlayout, buffer_type, output_mem_config, data_seed, scalar):
    random.seed(0)
    torch.manual_seed(data_seed)

    x = torch.Tensor(size=input_shape).uniform_(-100, 100)
    y = torch.Tensor(size=input_shape).uniform_(-100, 100)
    z = torch.Tensor(size=input_shape).uniform_(-100, 100)

    # ref_value = torch.addcdiv(x, y, z, value=scalar)
    logger.info(
        f"Running addcdiv with input_shape {input_shape} dtype {dtype} dlayout {dlayout} buffer_type {buffer_type} output_mem_config {output_mem_config} scalar {scalar} data_seed {data_seed}"
    )

    device = ttl.device.CreateDevice(0)

    try:
        t0 = setup_tt_tensor(
            x,
            device,
            dlayout[0],
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, buffer_type[0]),
            dtype[0],
        )
        t1 = setup_tt_tensor(
            y,
            device,
            dlayout[1],
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, buffer_type[1]),
            dtype[1],
        )
        t2 = setup_tt_tensor(
            z,
            device,
            dlayout[2],
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, buffer_type[2]),
            dtype[2],
        )
        t3 = ttl.tensor.addcdiv(t0, t1, t2, scalar, output_mem_config)

        y = tt2torch_tensor(t3)

    except Exception as exc:
        logger.warning(f"run_addcdiv RuntimeError occured {exc}")

    ttl.device.DeallocateBuffers(device)
    ttl.device.CloseDevice(device)

    logger.info(f"Finished running addcdiv")


test_sweep_args = [
    (
        (4, 6, 160, 224),
        [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE],
        [ttl.tensor.BufferType.DRAM, ttl.tensor.BufferType.L1, ttl.tensor.BufferType.L1],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        3514701,
        0.78125,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, buffer_type, output_mem_config, data_seed, scalar",
    (test_sweep_args),
)
def test_addcdiv_test(input_shape, dtype, dlayout, buffer_type, output_mem_config, data_seed, scalar):
    run_addcdiv(input_shape, dtype, dlayout, buffer_type, output_mem_config, data_seed, scalar)
