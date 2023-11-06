# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

import tt_lib as ttl
from models.utility_functions import tt2torch_tensor, comp_pcc
import torch


def run_nlp_create_qkv_heads_test(batch, seq_len, dtype, in0_mem_config, out_mem_config, device):
    torch.manual_seed(1234)

    in0_shape = [batch, 1, seq_len, 4672]

    A = torch.randn(in0_shape)

    in0_t = ttl.tensor.Tensor(A, dtype).to(ttl.tensor.Layout.TILE).to(device, in0_mem_config)

    q, k, v = ttl.tensor.nlp_create_qkv_heads(in0_t, out_mem_config)

    # Check memory of inputs and outputs
    assert in0_t.memory_config().buffer_type == in0_mem_config.buffer_type
    assert q.memory_config().buffer_type == out_mem_config.buffer_type
    assert k.memory_config().buffer_type == out_mem_config.buffer_type
    assert v.memory_config().buffer_type == out_mem_config.buffer_type
    logger.debug(f"in0: {in0_t.memory_config().buffer_type} and {in0_t.dtype()}")
    logger.debug(f"q: {q.memory_config().buffer_type} and {q.dtype()}")
    logger.debug(f"k: {k.memory_config().buffer_type} and {k.dtype()}")
    logger.debug(f"v: {v.memory_config().buffer_type} and {v.dtype()}")

    assert q.shape() == [batch, 71, seq_len, 64]
    assert k.shape() == [batch, 1, seq_len, 64]
    assert v.shape() == [batch, 1, seq_len, 64]

    pyt_got_back_rm_q = tt2torch_tensor(q)
    pyt_got_back_rm_k = tt2torch_tensor(k)
    pyt_got_back_rm_v = tt2torch_tensor(v)

    (ref_q, ref_k, ref_v) = torch.split(A, [4544, 64, 64], dim=-1)
    # Additional shuffling for Q head
    ref_q = torch.reshape(ref_q, [batch, seq_len, 71, 64]).transpose(-3, -2)

    if dtype == ttl.tensor.DataType.BFLOAT8_B:
        pcc = 0.99
    else:
        pcc = 1.0

    passing_pcc_q, output_pcc_q = comp_pcc(pyt_got_back_rm_q, ref_q, pcc)
    logger.info(f"Q passing={passing_pcc_q}")
    logger.info(f"Q output pcc={output_pcc_q}")
    assert passing_pcc_q
    passing_pcc_k, output_pcc_k = comp_pcc(pyt_got_back_rm_k, ref_k, pcc)
    logger.info(f"K passing={passing_pcc_k}")
    logger.info(f"K output pcc={output_pcc_k}")
    assert passing_pcc_k
    passing_pcc_v, output_pcc_v = comp_pcc(pyt_got_back_rm_v, ref_v, pcc)
    logger.info(f"V passing={passing_pcc_v}")
    logger.info(f"V output pcc={output_pcc_v}")
    assert passing_pcc_v


@pytest.mark.parametrize(
    "out_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=["in0_DRAM", "in0_L1"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
    ids=["BFLOAT8_B", "BFLOAT16"],
)
@pytest.mark.parametrize(
    "batch, seq_len",
    ((1, 32), (1, 64), (1, 128)),
    ids=[
        "batch1_seq32",
        "batch1_seq64",
        "batch1_seq128",
    ],
)
def test_nlp_create_qkv_heads_test(batch, seq_len, dtype, in0_mem_config, out_mem_config, request, device):
    ttl.profiler.set_profiler_location(f"nlp_create_qkv_heads_tm_{request.node.callspec.id}")
    run_nlp_create_qkv_heads_test(batch, seq_len, dtype, in0_mem_config, out_mem_config, device)


def test_nlp_create_qkv_heads_with_program_cache(use_program_cache, device):
    dtype = ttl.tensor.DataType.BFLOAT8_B
    dram_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
    for _ in range(2):
        run_nlp_create_qkv_heads_test(1, 32, dtype, dram_mem_config, dram_mem_config, device)

    dram_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)
    for _ in range(2):
        run_nlp_create_qkv_heads_test(1, 32, dtype, dram_mem_config, dram_mem_config, device)

    assert ttl.program_cache.num_entries() == 2
