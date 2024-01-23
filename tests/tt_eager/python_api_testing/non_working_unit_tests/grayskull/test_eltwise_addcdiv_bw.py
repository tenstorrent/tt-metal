# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import random
from loguru import logger
import pytest
import torch
import tt_lib as ttl


from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc_list
from tests.tt_eager.python_api_testing.sweep_tests import tt_lib_ops, pytorch_ops
from models.utility_functions import tt2torch_tensor


def run_addcdiv_bw(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, scalar, device):
    random.seed(0)
    torch.manual_seed(data_seed)

    x = torch.Tensor(size=input_shape).uniform_(-100, 100)
    y = torch.Tensor(size=input_shape).uniform_(-100, 100)
    z = torch.Tensor(size=input_shape).uniform_(-100, 100)
    w = torch.Tensor(size=input_shape).uniform_(-100, 100)

    ref_value = pytorch_ops.addcdiv_bw(x, y, z, w, scalar)

    tt_result = tt_lib_ops.addcdiv_bw(
        x=x,
        y=y,
        z=z,
        w=w,
        scalar=scalar,
        device=device,
        dtype=dtype,
        layout=dlayout,
        input_mem_config=in_mem_config,
        output_mem_config=out_mem_config,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_pcc_list(ref_value, tt_result)
    logger.debug(pcc_value)

    assert success


test_sweep_args = [
    (
        (6, 8, 224, 160),
        [
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.DataType.BFLOAT16,
        ],
        [ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE],
        [
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
            None,
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        ],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        4645844,
        0,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, scalar",
    (test_sweep_args),
)
def test_addcdiv_bw_test(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, scalar, device):
    run_addcdiv_bw(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, scalar, device)
