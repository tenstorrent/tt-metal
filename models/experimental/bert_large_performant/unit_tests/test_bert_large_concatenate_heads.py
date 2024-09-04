# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger


import ttnn
from models.utility_functions import (
    comp_pcc,
)
import torch


def run_bert_large_concatenate_heads_test(device, batch, dtype, in0_mem_config, out_mem_config):
    compute_grid_size = device.compute_with_storage_grid_size()
    if compute_grid_size.x * compute_grid_size.y < (12 * 9):
        pytest.skip(f"Grid size {compute_grid_size} is not supported")

    torch.manual_seed(1234)

    a_shape = [batch, 16, 384, 64]

    A = torch.randn(a_shape)

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

    out = ttnn.experimental.concatenate_heads(a_t, ttnn.CoreCoord(12, 9), memory_config=out_mem_config)

    # Check memory of inputs and outputs
    assert a_t.memory_config().buffer_type == in0_mem_config.buffer_type
    assert out.memory_config().buffer_type == out_mem_config.buffer_type

    logger.debug(f"in0: {a_t.memory_config().buffer_type} and {a_t.get_dtype()}")
    logger.debug(f"out: {out.memory_config().buffer_type} and {out.get_dtype()}")

    assert out.get_legacy_shape() == [batch, 1, 384, 1024]
    tt_host_rm_out = out.cpu().to(ttnn.ROW_MAJOR_LAYOUT)
    pyt_got_back_rm_out = tt_host_rm_out.to_torch()

    ref_out = torch.transpose(A, -3, -2).reshape([batch, 1, 384, 1024])
    passing_pcc, output_pcc = comp_pcc(pyt_got_back_rm_out, ref_out, 0.99)
    logger.debug(f"passing={passing_pcc}")
    logger.debug(f"output pcc={output_pcc}")
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
@pytest.mark.parametrize(
    "batch",
    (9, 8, 7),
    ids=[
        "batch_9",
        "batch_8",
        "batch_7",
    ],
)
def test_bert_large_concatenate_heads_test(device, batch, dtype, in0_mem_config, out_mem_config, request):
    run_bert_large_concatenate_heads_test(device, batch, dtype, in0_mem_config, out_mem_config)


def test_bert_large_concatenate_heads_with_program_cache(device, use_program_cache):
    dtype = ttnn.bfloat8_b
    mem_config = ttnn.DRAM_MEMORY_CONFIG
    for _ in range(2):
        run_bert_large_concatenate_heads_test(device, 9, dtype, mem_config, mem_config)
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.Tensor(py_dummy_tensor, dtype).to(ttnn.TILE_LAYOUT).to(device, mem_config)

    mem_config = ttnn.L1_MEMORY_CONFIG
    for _ in range(2):
        run_bert_large_concatenate_heads_test(device, 9, dtype, mem_config, mem_config)
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.Tensor(py_dummy_tensor, dtype).to(ttnn.TILE_LAYOUT).to(device, mem_config)

    assert device.num_program_cache_entries() == 2
