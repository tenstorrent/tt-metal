# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.pytorch_ops import ne as pt_ne
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import eltwise_ne as tt_ne


def run_eltwise_ne_tests(
    input_shape,
    in0_dtype,
    in1_dtype,
    in0_dlayout,
    in1_dlayout,
    in0_in_mem_config,
    in1_in_mem_config,
    out_mem_config,
    data_seed,
    device,
):
    random.seed(0)
    torch.manual_seed(data_seed)

    input0_mem_config = in0_in_mem_config
    if in0_in_mem_config == "SYSTEM_MEMORY":
        input0_mem_config = None

    input1_mem_config = in1_in_mem_config
    if in1_in_mem_config == "SYSTEM_MEMORY":
        input1_mem_config = None

    x = torch.Tensor(size=input_shape).uniform_(-100, 100).to(torch.bfloat16)
    y = torch.Tensor(size=input_shape).uniform_(-100, 100).to(torch.bfloat16)

    x_ref = x.detach().clone()
    y_ref = y.detach().clone()

    # get referent value
    ref_value = pt_ne(x_ref, y_ref).to(torch.bfloat16)

    # calculate tt output
    if in0_in_mem_config == "SYSTEM_MEMORY":
        in0_in_mem_config = None

    if in1_in_mem_config == "SYSTEM_MEMORY":
        in1_in_mem_config = None

    logger.info("Running eltwise_ne test")
    tt_result = tt_ne(
        x=x,
        y=y,
        device=device,
        dtype=[in0_dtype, in1_dtype],
        layout=[in0_dlayout, in1_dlayout],
        input_mem_config=[in0_in_mem_config, in1_in_mem_config],
        output_mem_config=out_mem_config,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)

    assert success


test_sweep_args = [
    # 2023-09-21
    # TILE, TILE
    (
        (7, 14, 32, 160),
        ttnn.bfloat16,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.TILE_LAYOUT,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        17155532,
    ),
    (
        (7, 14, 32, 160),
        ttnn.bfloat16,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.TILE_LAYOUT,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        16305027,
    ),
    (
        (7, 14, 32, 160),
        ttnn.bfloat16,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.TILE_LAYOUT,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        "SYSTEM_MEMORY",
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        13587334,
    ),
    (
        (7, 14, 32, 160),
        ttnn.bfloat16,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.TILE_LAYOUT,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        10177486,
    ),
    (
        (7, 14, 32, 160),
        ttnn.bfloat16,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.TILE_LAYOUT,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        15991940,
    ),
    (
        (7, 14, 32, 160),
        ttnn.bfloat16,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.TILE_LAYOUT,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        "SYSTEM_MEMORY",
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        12014143,
    ),
    (
        (7, 14, 32, 160),
        ttnn.bfloat16,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.TILE_LAYOUT,
        "SYSTEM_MEMORY",
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        19575052,
    ),
    (
        (7, 14, 32, 160),
        ttnn.bfloat16,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.TILE_LAYOUT,
        "SYSTEM_MEMORY",
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        7329721,
    ),
    (
        (7, 14, 32, 160),
        ttnn.bfloat16,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.TILE_LAYOUT,
        "SYSTEM_MEMORY",
        "SYSTEM_MEMORY",
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        16934160,
    ),
    # ROW_MAJOR, ROW_MAJOR
    (
        (4, 22, 303, 424),
        ttnn.bfloat16,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        14073508,
    ),
    (
        (4, 22, 303, 424),
        ttnn.bfloat16,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        19451336,
    ),
    (
        (4, 22, 303, 424),
        ttnn.bfloat16,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        "SYSTEM_MEMORY",
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        9234542,
    ),
    (
        (4, 22, 303, 424),
        ttnn.bfloat16,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        15118389,
    ),
    (
        (4, 22, 303, 424),
        ttnn.bfloat16,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        16530771,
    ),
    (
        (4, 22, 303, 424),
        ttnn.bfloat16,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        "SYSTEM_MEMORY",
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        11991265,
    ),
    (
        (4, 22, 303, 424),
        ttnn.bfloat16,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
        "SYSTEM_MEMORY",
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        2763978,
    ),
    (
        (4, 22, 303, 424),
        ttnn.bfloat16,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
        "SYSTEM_MEMORY",
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        10882535,
    ),
    (
        (4, 22, 303, 424),
        ttnn.bfloat16,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
        "SYSTEM_MEMORY",
        "SYSTEM_MEMORY",
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        3870495,
    ),
]


@pytest.mark.parametrize(
    "input_shape, in0_dtype, in1_dtype, in0_dlayout, in1_dlayout, in0_in_mem_config, in1_in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_eltwise_ne_test(
    input_shape,
    in0_dtype,
    in1_dtype,
    in0_dlayout,
    in1_dlayout,
    in0_in_mem_config,
    in1_in_mem_config,
    out_mem_config,
    data_seed,
    device,
):
    run_eltwise_ne_tests(
        input_shape,
        in0_dtype,
        in1_dtype,
        in0_dlayout,
        in1_dlayout,
        in0_in_mem_config,
        in1_in_mem_config,
        out_mem_config,
        data_seed,
        device,
    )
