# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys
from loguru import logger
import random

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../../../..")

import pytest
import torch
import tt_lib as ttl

from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import transpose_nh as tt_transpose_nh
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand


def run_transpose_nh_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    torch.manual_seed(data_seed)
    overall_success = True

    for i in range(10):
        x = gen_rand(size = input_shape, low = -100, high = 100)
        # compute ref value
        x_ref = x.detach().clone()
        ref_value = pytorch_ops.transpose(x_ref, dim0=0, dim1=-2)

        tt_result = tt_transpose_nh(
            x=x,
            device=device,
            device_id=0,
            dtype=[dtype],
            layout=[dlayout],
            input_mem_config=[in_mem_config],
            output_mem_config=out_mem_config
        )

        # compare tt and golden outputs
        success, pcc_value = comp_equal(ref_value, tt_result)
        logger.debug(pcc_value)
        logger.debug(success)
        overall_success = overall_success and success
        ttl.device.DeallocateBuffers(device)

    assert overall_success


test_sweep_args=[
    ((5, 15, 64, 320), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM), ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM), 12915139),
    ((5, 15, 64, 320), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, None, ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM), 10638326),
    ((12, 21, 464, 236), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM), ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM), 14610962),
    ((6, 18, 92, 108), ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM), ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM), 12602337),
]

@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (
        test_sweep_args
    ),
)

def test_transpose_nh_test(
    input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device
):
    random.seed(0)
    run_transpose_nh_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
