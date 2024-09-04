# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn

from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import eltwise_rpow as tt_eltwise_rpow


def run_eltwise_rpow_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, factor, data_seed, device):
    torch.manual_seed(data_seed)

    if in_mem_config == "SYSTEM_MEMORY":
        in_mem_config = None

    x = torch.Tensor(size=input_shape).uniform_(-100, 100)
    x_ref = x.detach().clone()

    # get ref result
    ref_value = pytorch_ops.eltwise_rpow(x_ref, factor=factor)

    tt_result = tt_eltwise_rpow(
        x=x,
        device=device,
        dtype=[dtype],
        layout=[dlayout],
        input_mem_config=[in_mem_config],
        output_mem_config=out_mem_config,
        factor=factor,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)

    assert success


test_sweep_args = [
    (
        (7, 14, 32, 160),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)),
        (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)),
        9.65810237498298,
        10177486,
    ),
    (
        (7, 14, 32, 160),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)),
        (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)),
        4.910684195971845,
        15991940,
    ),
    (
        (7, 14, 32, 160),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        "SYSTEM_MEMORY",
        (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)),
        9.190519884672804,
        12014143,
    ),
    (
        (11, 22, 448, 128),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)),
        (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)),
        5.473809829627938,
        10679014,
    ),
    (
        (11, 22, 448, 128),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)),
        (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)),
        9.320002705349427,
        10798651,
    ),
    (
        (11, 22, 448, 128),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        "SYSTEM_MEMORY",
        (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)),
        0.19607612839476013,
        1190117,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, factor, data_seed",
    (test_sweep_args),
)
def test_eltwise_rpow(input_shape, dtype, dlayout, in_mem_config, out_mem_config, factor, data_seed, device):
    run_eltwise_rpow_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, factor, data_seed, device)
