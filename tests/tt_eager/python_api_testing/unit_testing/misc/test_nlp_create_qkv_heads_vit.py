# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

from models.common.utility_functions import tt2torch_tensor, comp_pcc
import torch
import ttnn

"""
ViT shapes + functionality
"""


def run_nlp_create_qkv_heads_vit_test(batch, seq_len, dtype, in0_mem_config, out_mem_config, device):
    torch.manual_seed(1234)

    in0_shape = [batch, 1, seq_len, 2304]

    A = torch.randn(in0_shape)

    in0_t = ttnn.Tensor(A, dtype).to(ttnn.TILE_LAYOUT).to(device, in0_mem_config)

    q, k, v = ttnn.experimental.nlp_create_qkv_heads_vit(in0_t, memory_config=out_mem_config)

    # Check memory of inputs and outputs
    assert in0_t.memory_config().buffer_type == in0_mem_config.buffer_type
    assert q.memory_config().buffer_type == out_mem_config.buffer_type
    assert k.memory_config().buffer_type == out_mem_config.buffer_type
    assert v.memory_config().buffer_type == out_mem_config.buffer_type
    logger.debug(f"in0: {in0_t.memory_config().buffer_type} and {in0_t.get_dtype()}")
    logger.debug(f"q: {q.memory_config().buffer_type} and {q.get_dtype()}")
    logger.debug(f"k: {k.memory_config().buffer_type} and {k.get_dtype()}")
    logger.debug(f"v: {v.memory_config().buffer_type} and {v.get_dtype()}")

    assert list(q.padded_shape) == [batch, 12, seq_len, 64]
    assert list(k.padded_shape) == [batch, 12, seq_len, 64]
    assert list(v.padded_shape) == [batch, 12, seq_len, 64]

    pyt_got_back_rm_q = tt2torch_tensor(q)
    pyt_got_back_rm_k = tt2torch_tensor(k)
    pyt_got_back_rm_v = tt2torch_tensor(v)

    (ref_q, ref_k, ref_v) = torch.split(A, [768, 768, 768], dim=-1)
    # Additional shuffling for Q,K,V heads
    ref_q = torch.reshape(ref_q, [batch, seq_len, 12, 64]).transpose(-3, -2)
    ref_k = torch.reshape(ref_k, [batch, seq_len, 12, 64]).transpose(-3, -2)
    ref_v = torch.reshape(ref_v, [batch, seq_len, 12, 64]).transpose(-3, -2)

    if dtype == ttnn.bfloat8_b:
        pcc = 0.99
    else:
        pcc = 1.0

    passing_pcc_q, output_pcc_q = comp_pcc(pyt_got_back_rm_q, ref_q, pcc)
    logger.debug(f"Q passing={passing_pcc_q}")
    logger.debug(f"Q output pcc={output_pcc_q}")
    assert passing_pcc_q
    passing_pcc_k, output_pcc_k = comp_pcc(pyt_got_back_rm_k, ref_k, pcc)
    logger.debug(f"K passing={passing_pcc_k}")
    logger.debug(f"K output pcc={output_pcc_k}")
    assert passing_pcc_k
    passing_pcc_v, output_pcc_v = comp_pcc(pyt_got_back_rm_v, ref_v, pcc)
    logger.debug(f"V passing={passing_pcc_v}")
    logger.debug(f"V output pcc={output_pcc_v}")
    assert passing_pcc_v


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
    "batch, seq_len",
    ((1, 224), (1, 4096)),
    ids=[
        "batch1_seq224",
        "batch1_seq4k",
    ],
)
def test_nlp_create_qkv_heads_vit_test(batch, seq_len, dtype, in0_mem_config, out_mem_config, request, device):
    run_nlp_create_qkv_heads_vit_test(batch, seq_len, dtype, in0_mem_config, out_mem_config, device)


def test_nlp_create_qkv_heads_vit_with_program_cache(device):
    dtype = ttnn.bfloat8_b
    mem_config = ttnn.DRAM_MEMORY_CONFIG
    for _ in range(2):
        run_nlp_create_qkv_heads_vit_test(1, 32, dtype, mem_config, mem_config, device)
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.Tensor(py_dummy_tensor, dtype).to(ttnn.TILE_LAYOUT).to(device, mem_config)

    mem_config = ttnn.L1_MEMORY_CONFIG
    for _ in range(2):
        run_nlp_create_qkv_heads_vit_test(1, 32, dtype, mem_config, mem_config, device)
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.Tensor(py_dummy_tensor, dtype).to(ttnn.TILE_LAYOUT).to(device, mem_config)

    assert device.num_program_cache_entries() == 2


@pytest.mark.parametrize("small_output", [0, 1, 2], ids=["q", "k", "v"])
def test_nlp_create_qkv_heads_vit_rejects_wrong_size_output_tensors(small_output, device, expect_error):
    batch, seq_len, num_heads, head_dim = 1, 1024, 12, 64
    to_dev = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    x = to_dev(torch.randn(batch, 1, seq_len, 2304))
    outputs = [
        to_dev(torch.zeros(batch, num_heads // 2 if i == small_output else num_heads, seq_len, head_dim))
        for i in range(3)
    ]
    neighbour = to_dev(torch.full((1, 1, 128, 128), 768.0))
    with expect_error(RuntimeError, "Preallocated output"):
        ttnn.experimental.nlp_create_qkv_heads_vit(x, output_tensors=outputs)
    assert torch.equal(ttnn.to_torch(neighbour), torch.full((1, 1, 128, 128), 768.0, dtype=torch.bfloat16))
