# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import tt_lib as ttl

from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import permute as tt_permute
from tests.tt_eager.python_api_testing.sweep_tests.common import set_slow_dispatch_mode


def run_permute_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, permute_dims, data_seed, device):
    torch.manual_seed(data_seed)
    # prev_dispatch_mode = set_slow_dispatch_mode("1")

    if in_mem_config == "SYSTEM_MEMORY":
        in_mem_config = None

    x = torch.Tensor(size=input_shape).uniform_(-100, 100)
    x_ref = x.detach().clone()

    # get ref result
    ref_value = pytorch_ops.permute(x_ref, permute_dims=permute_dims)

    tt_result = tt_permute(
        x=x,
        permute_dims=permute_dims,
        device=device,
        dtype=[dtype],
        layout=[dlayout],
        input_mem_config=[in_mem_config],
        output_mem_config=out_mem_config,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)

    assert success


# permute,"[[3, 4, 98, 124]]","{'dtype': [<DataType.BFLOAT16: 0>], 'layout': [<Layout.ROW_MAJOR: 0>], 'input_mem_config': [None], 'output_mem_config': tt::tt_metal::MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM), 'permute_dims': (1, 2, 3, 0)}",18244914,"(('TT_METAL_SLOW_DISPATCH_MODE', ''),)",error,Unexpected index,fail
# permute,"[[2, 3, 82, 126]]","{'dtype': [<DataType.BFLOAT16: 0>], 'layout': [<Layout.ROW_MAJOR: 0>], 'input_mem_config': [None], 'output_mem_config': tt::tt_metal::MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM), 'permute_dims': (0, 1, 2, 3)}",4054436,"(('TT_METAL_SLOW_DISPATCH_MODE', ''),)",error,Unexpected index,fail
# permute,"[[2, 6, 6, 2]]","{'dtype': [<DataType.BFLOAT16: 0>], 'layout': [<Layout.ROW_MAJOR: 0>], 'input_mem_config': [None], 'output_mem_config': tt::tt_metal::MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM), 'permute_dims': (1, 3, 0, 2)}",13369045,"(('TT_METAL_SLOW_DISPATCH_MODE', ''),)",error,Unexpected index,fail
# permute,"[[3, 5, 92, 52]]","{'dtype': [<DataType.BFLOAT16: 0>], 'layout': [<Layout.ROW_MAJOR: 0>], 'input_mem_config': [None], 'output_mem_config': tt::tt_metal::MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM), 'permute_dims': (3, 0, 1, 2)}",3423616,"(('TT_METAL_SLOW_DISPATCH_MODE', ''),)",error,Unexpected index,fail

test_sweep_args = [
    (
        (3, 4, 98, 124),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        "SYSTEM_MEMORY",
        (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)),
        (1, 2, 3, 0),
        18244914,
    ),
    (
        (2, 3, 82, 126),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        "SYSTEM_MEMORY",
        (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)),
        (0, 1, 2, 3),
        4054436,
    ),
    (
        (2, 6, 6, 2),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        "SYSTEM_MEMORY",
        (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)),
        (1, 3, 0, 2),
        13369045,
    ),
    (
        (3, 5, 92, 52),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        "SYSTEM_MEMORY",
        (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)),
        (3, 0, 1, 2),
        3423616,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, permute_dims, data_seed",
    (test_sweep_args),
)
def test_permute(input_shape, dtype, dlayout, in_mem_config, out_mem_config, permute_dims, data_seed, device):
    run_permute_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, permute_dims, data_seed, device)
