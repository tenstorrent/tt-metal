# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
import torch
import ttnn

from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_allclose
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import std_hw as tt_std_hw


def run_std_hw_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    torch.manual_seed(data_seed)

    x = torch.Tensor(size=input_shape).uniform_(-10, 10)
    x_ref = x.detach().clone()

    # compute ref value
    ref_value = pytorch_ops.std_hw(x_ref)

    tt_result = tt_std_hw(
        x=x,
        device=device,
        dtype=[dtype],
        layout=[dlayout],
        input_mem_config=[in_mem_config],
        output_mem_config=out_mem_config,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_allclose(ref_value, tt_result, atol=0.1)
    logger.debug(pcc_value)

    assert success


test_sweep_args = [
    (
        (4, 7, 32, 96),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        17155532,
    ),
    (
        (4, 7, 32, 96),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        16305027,
    ),
    (
        (1, 5, 36, 168),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        17477346,
    ),
    (
        (1, 5, 36, 168),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        11645166,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_std_hw_test(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_std_hw_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
