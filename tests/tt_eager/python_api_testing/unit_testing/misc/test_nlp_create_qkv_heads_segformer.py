# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

from models.utility_functions import tt2torch_tensor, comp_pcc
from models.utility_functions import is_grayskull
import torch
import ttnn

"""
Segformer shapes + functionality
"""


def run_nlp_create_qkv_heads_segformer_test(batch, seq_len, hidden_dim, dtype, in0_mem_config, out_mem_config, device):
    torch.manual_seed(1234)

    in0_shape = [batch, 1, seq_len, hidden_dim]

    A = torch.randn(in0_shape)

    in0_t = ttnn.Tensor(A, dtype).to(ttnn.TILE_LAYOUT).to(device, in0_mem_config)

    q = ttnn.experimental.nlp_create_qkv_heads_segformer(in0_t, memory_config=out_mem_config)[0]

    # Check memory of inputs and outputs
    assert in0_t.memory_config().buffer_type == in0_mem_config.buffer_type
    assert q.memory_config().buffer_type == out_mem_config.buffer_type
    logger.debug(f"in0: {in0_t.memory_config().buffer_type} and {in0_t.get_dtype()}")
    logger.debug(f"q: {q.memory_config().buffer_type} and {q.get_dtype()}")

    head_dim = 32
    heads_num = hidden_dim // head_dim
    assert list(q.padded_shape) == [batch, heads_num, seq_len, head_dim]

    pyt_got_back_rm_q = tt2torch_tensor(q)

    ref_q = A
    # Additional shuffling for Q,K,V heads
    ref_q = torch.reshape(ref_q, [batch, seq_len, heads_num, head_dim]).transpose(-3, -2)

    if dtype == ttnn.bfloat8_b:
        pcc = 0.99
    else:
        pcc = 1.0

    passing_pcc_q, output_pcc_q = comp_pcc(pyt_got_back_rm_q, ref_q, pcc)
    logger.debug(f"Q passing={passing_pcc_q}")
    logger.debug(f"Q output pcc={output_pcc_q}")
    assert passing_pcc_q


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
    "batch, seq_len, hidden_dim",
    ((1, 4096, 64), (1, 1024, 160), (1, 256, 256)),
    ids=[
        "batch1_seq4k",
        "batch1_seq1k",
        "batch1_seq256",
    ],
)
def test_nlp_create_qkv_heads_segformer_test(
    batch, seq_len, hidden_dim, dtype, in0_mem_config, out_mem_config, request, device
):
    if is_grayskull() and dtype == ttnn.float32:
        pytest.skip("Skipping float32 tests on Grayskull")
    run_nlp_create_qkv_heads_segformer_test(batch, seq_len, hidden_dim, dtype, in0_mem_config, out_mem_config, device)


def test_nlp_create_qkv_heads_segformer_with_program_cache(device):
    dtype = ttnn.bfloat8_b
    mem_config = ttnn.DRAM_MEMORY_CONFIG
    for _ in range(2):
        run_nlp_create_qkv_heads_segformer_test(1, 32, 32, dtype, mem_config, mem_config, device)
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.Tensor(py_dummy_tensor, dtype).to(ttnn.TILE_LAYOUT).to(device, mem_config)

    mem_config = ttnn.L1_MEMORY_CONFIG
    for _ in range(2):
        run_nlp_create_qkv_heads_segformer_test(1, 32, 32, dtype, mem_config, mem_config, device)
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.Tensor(py_dummy_tensor, dtype).to(ttnn.TILE_LAYOUT).to(device, mem_config)

    assert device.num_program_cache_entries() == 2
