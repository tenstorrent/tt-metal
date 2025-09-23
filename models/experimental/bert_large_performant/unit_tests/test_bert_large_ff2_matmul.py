# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from loguru import logger


import ttnn
from models.utility_functions import (
    comp_pcc,
)
from models.demos.metal_BERT_large_11.tt import custom_matmuls
import torch
import pytest


def run_bert_large_ff2_matmul_test(device, dtype, in0_mem_config, in1_mem_config, bias_mem_config, out_mem_config):
    compute_grid_size = device.compute_with_storage_grid_size()
    if compute_grid_size.x < 12:
        pytest.skip(f"Grid size {compute_grid_size} is not supported")

    torch.manual_seed(1234)

    a_shape = [9, 1, 384, 4096]
    b_shape = [1, 1, 4096, 1024]
    bias_shape = [1, 1, 1, 1024]
    bias_pad_shape = [1, 1, 32, 1024]

    A = torch.randn(a_shape)
    B = torch.randn(b_shape) - 0.95
    bias = torch.randint(-20, 20, bias_shape, dtype=torch.float)
    bias_padded = torch.nn.functional.pad(bias, (0, 0, 0, 32 - bias.size(2)))

    a_t = ttnn.Tensor(
        A.flatten().tolist(),
        a_shape,
        dtype,
        ttnn.TILE_LAYOUT,
    ).to(device, in0_mem_config)
    b_t = ttnn.Tensor(
        B.flatten().tolist(),
        b_shape,
        dtype,
        ttnn.TILE_LAYOUT,
    ).to(device, in1_mem_config)

    if bias_mem_config is not None:
        bias_t = ttnn.Tensor(
            bias_padded.flatten().tolist(),
            bias_pad_shape,
            dtype,
            ttnn.TILE_LAYOUT,
        ).to(device, bias_mem_config)
    else:
        bias_t = None

    t2 = custom_matmuls.bert_large_ff2_matmul(a_t, b_t, bias_t, out_mem_config)

    # Check memory of inputs and outputs
    assert a_t.memory_config().buffer_type == in0_mem_config.buffer_type
    assert b_t.memory_config().buffer_type == in1_mem_config.buffer_type
    if bias_mem_config is not None:
        assert bias_t.memory_config().buffer_type == bias_mem_config.buffer_type
    assert t2.memory_config().buffer_type == out_mem_config.buffer_type
    logger.debug(f"in0 is on: {a_t.memory_config().buffer_type}")
    logger.debug(f"in1 is on: {b_t.memory_config().buffer_type}")
    if bias_mem_config is not None:
        logger.debug(f"bias is on: {bias_t.memory_config().buffer_type}")
    logger.debug(f"out is on: {t2.memory_config().buffer_type}")

    assert t2.padded_shape == [9, 1, 384, 1024]
    pyt_got_back_rm = ttnn.to_torch(t2)

    ref_bmm = torch.matmul(A, B)
    if bias_mem_config is not None:
        ref_bmm = ref_bmm + bias
    passing_pcc, output_pcc = comp_pcc(ref_bmm, pyt_got_back_rm, 0.99)
    logger.debug(f"Passing={passing_pcc}")
    logger.debug(f"Output pcc={output_pcc}")

    assert passing_pcc


@pytest.mark.parametrize(
    "out_mem_config",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "bias_mem_config",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
        None,
    ),
    ids=["bias_DRAM", "bias_L1", "bias_None"],
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
def test_bert_large_ff2_matmul_test(
    device,
    dtype,
    in0_mem_config,
    in1_mem_config,
    bias_mem_config,
    out_mem_config,
    request,
):
    run_bert_large_ff2_matmul_test(device, dtype, in0_mem_config, in1_mem_config, bias_mem_config, out_mem_config)


def test_bert_large_ff2_matmul_with_program_cache(device):
    dtype = ttnn.bfloat8_b
    mem_config = ttnn.DRAM_MEMORY_CONFIG
    for _ in range(2):
        run_bert_large_ff2_matmul_test(
            device,
            dtype,
            mem_config,
            mem_config,
            mem_config,
            mem_config,
        )
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.Tensor(py_dummy_tensor, dtype, device, ttnn.TILE_LAYOUT, mem_config)

    mem_config = ttnn.L1_MEMORY_CONFIG
    for _ in range(2):
        run_bert_large_ff2_matmul_test(
            device,
            dtype,
            mem_config,
            mem_config,
            mem_config,
            mem_config,
        )
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.Tensor(py_dummy_tensor, dtype, device, ttnn.TILE_LAYOUT, mem_config)

    assert device.num_program_cache_entries() == 2
