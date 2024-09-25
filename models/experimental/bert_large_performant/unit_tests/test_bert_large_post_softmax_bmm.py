# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from loguru import logger


import numpy as np

import ttnn
from models.utility_functions import (
    comp_pcc,
)
from models.demos.metal_BERT_large_11.tt import custom_matmuls
import torch


def run_bert_large_post_softmax_bmm_test(device, dtype, in0_mem_config, in1_mem_config, out_mem_config):
    compute_grid_size = device.compute_with_storage_grid_size()
    if compute_grid_size.x < 12:
        pytest.skip(f"Grid size {compute_grid_size} is not supported")

    torch.manual_seed(1234)

    a_shape = [
        9,
        16,
        384,
        384,
    ]  # No-op reshape to [9, 16, 384, 384] in post_softmax_bmm
    b_shape = [9, 16, 384, 64]
    out_shape = [9, 16, 384, 64]

    A = torch.randn(a_shape)
    B = torch.randn(b_shape) - 0.95

    a_t = (
        ttnn.Tensor(
            A.flatten().tolist(),
            a_shape,
            dtype,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        .to(ttnn.TILE_LAYOUT)
        .to(device, in0_mem_config)
    )
    b_t = (
        ttnn.Tensor(
            B.flatten().tolist(),
            b_shape,
            dtype,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        .to(ttnn.TILE_LAYOUT)
        .to(device, in1_mem_config)
    )

    t2 = custom_matmuls.bert_large_post_softmax_bmm(a_t, b_t, out_mem_config)

    # Check memory of inputs and outputs
    assert a_t.memory_config().buffer_type == in0_mem_config.buffer_type
    assert b_t.memory_config().buffer_type == in1_mem_config.buffer_type
    assert t2.memory_config().buffer_type == out_mem_config.buffer_type
    logger.debug(f"in0 is on: {a_t.memory_config().buffer_type}")
    logger.debug(f"in1 is on: {b_t.memory_config().buffer_type}")
    logger.debug(f"out is on: {t2.memory_config().buffer_type}")

    assert t2.shape.with_tile_padding() == out_shape
    tt_host_rm = t2.cpu().to(ttnn.ROW_MAJOR_LAYOUT)
    pyt_got_back_rm = tt_host_rm.to_torch()

    ref_bmm = torch.matmul(A.reshape([9, 16, 384, 384]), B)
    passing_pcc, output_pcc = comp_pcc(ref_bmm, pyt_got_back_rm, 0.99)
    logger.debug(f"Passing={passing_pcc}")
    logger.debug(f"Output pcc={output_pcc}")

    assert passing_pcc


import pytest


@pytest.mark.parametrize(
    "out_mem_config",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "in1_mem_config",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["in1_DRAM", "in1_L1"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["in0_DRAM", "in0_L1"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat8_b, ttnn.bfloat16),
    ids=["BFLOAT8_B", "BFLOAT16"],
)
def test_bert_large_post_softmax_bmm_test(device, dtype, in0_mem_config, in1_mem_config, out_mem_config, request):
    run_bert_large_post_softmax_bmm_test(device, dtype, in0_mem_config, in1_mem_config, out_mem_config)


def test_bert_large_post_softmax_bmm_with_program_cache(device, use_program_cache):
    dtype = ttnn.bfloat8_b
    mem_config = ttnn.DRAM_MEMORY_CONFIG
    for _ in range(2):
        run_bert_large_post_softmax_bmm_test(device, dtype, mem_config, mem_config, mem_config)
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.Tensor(py_dummy_tensor, dtype).to(ttnn.TILE_LAYOUT).to(device, mem_config)

    mem_config = ttnn.L1_MEMORY_CONFIG
    for _ in range(2):
        run_bert_large_post_softmax_bmm_test(device, dtype, mem_config, mem_config, mem_config)
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.Tensor(py_dummy_tensor, dtype).to(ttnn.TILE_LAYOUT).to(device, mem_config)

    assert device.num_program_cache_entries() == 2
