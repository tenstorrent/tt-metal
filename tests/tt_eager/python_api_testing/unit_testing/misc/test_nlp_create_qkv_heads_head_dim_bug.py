# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Test for nlp_create_qkv_heads bug when head_dim < TILE_WIDTH (32)

BUG: Integer division bug in nlp_create_qkv_heads_program_factory.cpp causes
incorrect output shapes when head_dim < 32.

Root cause: q_num_tiles = num_heads * (head_dim / TILE_WIDTH)
When head_dim=16: q_num_tiles = num_heads * 0 = 0 (reads zero tiles!)

Expected: q_num_tiles = (num_heads * head_dim + TILE_WIDTH - 1) / TILE_WIDTH

This test demonstrates the bug with head_dim=16 and verifies the fix.
"""

import pytest
from loguru import logger
import torch
import ttnn
from models.common.utility_functions import tt2torch_tensor, comp_pcc


def run_nlp_create_qkv_heads_small_head_dim_test(
    batch,
    seq_len,
    head_dim,
    num_heads,
    dtype,
    in_mem_config,
    out_mem_config,
    device,
):
    """
    Test nlp_create_qkv_heads with head_dim < 32 (TILE_WIDTH)

    This reproduces a bug where integer division causes incorrect tile calculations.
    """
    torch.manual_seed(1234)

    embedding_dim = num_heads * head_dim
    qkv_width = embedding_dim * 3  # Q + K + V concatenated

    logger.info(f"Testing head_dim={head_dim} < TILE_WIDTH(32) bug fix")
    logger.info(f"batch={batch}, seq_len={seq_len}, num_heads={num_heads}, embedding_dim={embedding_dim}")

    # Create fused QKV tensor: [batch, 1, seq_len, embedding_dim * 3]
    in0_shape = [batch, 1, seq_len, qkv_width]
    A = torch.randn(in0_shape)
    in0_t = ttnn.Tensor(A, dtype).to(ttnn.TILE_LAYOUT).to(device, in_mem_config)

    # Call nlp_create_qkv_heads - this triggers the bug when head_dim < 32
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        in0_t,
        None,  # No separate KV tensor
        num_heads=num_heads,
        num_kv_heads=num_heads,
        transpose_k_heads=False,
        memory_config=out_mem_config,
    )

    # Check memory configurations
    assert in0_t.memory_config().buffer_type == in_mem_config.buffer_type
    assert q.memory_config().buffer_type == out_mem_config.buffer_type
    assert k.memory_config().buffer_type == out_mem_config.buffer_type
    assert v.memory_config().buffer_type == out_mem_config.buffer_type

    # CRITICAL: Check output shapes - this is where the bug manifests
    # Expected: [batch, num_heads, seq_len, head_dim]
    expected_shape = [batch, num_heads, seq_len, head_dim]

    logger.info(f"Expected shape: {expected_shape}")
    logger.info(f"Q output shape: {list(q.padded_shape)}")
    logger.info(f"K output shape: {list(k.padded_shape)}")
    logger.info(f"V output shape: {list(v.padded_shape)}")

    # Verify shapes are correct
    assert list(q.padded_shape) == expected_shape, \
        f"Q shape mismatch! Expected {expected_shape}, got {list(q.padded_shape)}"
    assert list(k.padded_shape) == expected_shape, \
        f"K shape mismatch! Expected {expected_shape}, got {list(k.padded_shape)}"
    assert list(v.padded_shape) == expected_shape, \
        f"V shape mismatch! Expected {expected_shape}, got {list(v.padded_shape)}"

    # Verify numerical correctness
    pyt_got_back_rm_q = tt2torch_tensor(q)
    pyt_got_back_rm_k = tt2torch_tensor(k)
    pyt_got_back_rm_v = tt2torch_tensor(v)

    # Split reference QKV
    (ref_q, ref_k, ref_v) = torch.split(
        A, [embedding_dim, embedding_dim, embedding_dim], dim=-1
    )

    # Reshape to match expected output: [batch, num_heads, seq_len, head_dim]
    ref_q = torch.reshape(ref_q, [batch, seq_len, num_heads, head_dim]).transpose(-3, -2)
    ref_k = torch.reshape(ref_k, [batch, seq_len, num_heads, head_dim]).transpose(-3, -2)
    ref_v = torch.reshape(ref_v, [batch, seq_len, num_heads, head_dim]).transpose(-3, -2)

    if dtype == ttnn.bfloat8_b:
        pcc = 0.99
    else:
        pcc = 1.0

    passing_pcc_q, output_pcc_q = comp_pcc(pyt_got_back_rm_q, ref_q, pcc)
    logger.info(f"Q PCC: {output_pcc_q} (passing={passing_pcc_q})")
    assert passing_pcc_q, f"Q tensor PCC check failed: {output_pcc_q}"

    passing_pcc_k, output_pcc_k = comp_pcc(pyt_got_back_rm_k, ref_k, pcc)
    logger.info(f"K PCC: {output_pcc_k} (passing={passing_pcc_k})")
    assert passing_pcc_k, f"K tensor PCC check failed: {output_pcc_k}"

    passing_pcc_v, output_pcc_v = comp_pcc(pyt_got_back_rm_v, ref_v, pcc)
    logger.info(f"V PCC: {output_pcc_v} (passing={passing_pcc_v})")
    assert passing_pcc_v, f"V tensor PCC check failed: {output_pcc_v}"

    logger.info("Test passed - bug is fixed")


def run_nlp_create_qkv_heads_with_sdpa_test(
    batch,
    seq_len,
    head_dim,
    num_heads,
    device,
):
    """
    Integration test: nlp_create_qkv_heads → SDPA (Scaled Dot-Product Attention)

    This verifies that outputs from nlp_create_qkv_heads with head_dim < 32
    work correctly with downstream operations like SDPA. Tests for potential
    memory layout or tiling issues that could cause crashes.
    """
    torch.manual_seed(1234)

    embedding_dim = num_heads * head_dim
    qkv_width = embedding_dim * 3  # Q + K + V concatenated

    logger.info(f"Integration test: head_dim={head_dim} with SDPA")
    logger.info(f"batch={batch}, seq_len={seq_len}, num_heads={num_heads}")

    # Create fused QKV tensor: [batch, 1, seq_len, embedding_dim * 3]
    in0_shape = [batch, 1, seq_len, qkv_width]
    A = torch.randn(in0_shape)
    in0_t = ttnn.Tensor(A, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device, ttnn.DRAM_MEMORY_CONFIG)

    # Call nlp_create_qkv_heads
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        in0_t,
        None,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        transpose_k_heads=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # CRITICAL: Run SDPA on the outputs
    # This will crash if memory layout/tiling is incorrect
    scale = 1.0 / (head_dim ** 0.5)

    try:
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            scale=scale,
        )
    except Exception as e:
        logger.error(f"SDPA failed with head_dim={head_dim}: {e}")
        raise AssertionError(f"SDPA operation failed - likely memory layout issue: {e}")

    # Verify output shape
    # Expected: [batch, num_heads, seq_len, head_dim]
    expected_shape = [batch, num_heads, seq_len, head_dim]
    assert list(attn_output.padded_shape) == expected_shape, \
        f"SDPA output shape mismatch! Expected {expected_shape}, got {list(attn_output.padded_shape)}"

    # Verify numerical correctness against PyTorch reference
    attn_output_torch = tt2torch_tensor(attn_output)

    # Compute reference using PyTorch
    (ref_q, ref_k, ref_v) = torch.split(
        A, [embedding_dim, embedding_dim, embedding_dim], dim=-1
    )
    ref_q = torch.reshape(ref_q, [batch, seq_len, num_heads, head_dim]).transpose(-3, -2)
    ref_k = torch.reshape(ref_k, [batch, seq_len, num_heads, head_dim]).transpose(-3, -2)
    ref_v = torch.reshape(ref_v, [batch, seq_len, num_heads, head_dim]).transpose(-3, -2)

    # PyTorch SDPA (use manual computation for compatibility)
    attn_weights = torch.matmul(ref_q, ref_k.transpose(-2, -1)) * scale

    # Apply causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

    attn_weights = torch.softmax(attn_weights, dim=-1)
    ref_attn_output = torch.matmul(attn_weights, ref_v)

    # Compare with reference
    passing_pcc, output_pcc = comp_pcc(attn_output_torch, ref_attn_output, 0.99)
    logger.info(f"SDPA output PCC: {output_pcc} (passing={passing_pcc})")
    assert passing_pcc, f"SDPA output PCC check failed: {output_pcc}"

    logger.info("Integration test passed - nlp_create_qkv_heads outputs work with SDPA")


@pytest.mark.parametrize(
    "batch, seq_len, head_dim, num_heads",
    (
        (1, 32, 8, 8),     # head_dim=8: Smallest case
        (1, 32, 16, 4),    # head_dim=16: Main bug case
        (1, 32, 32, 4),    # head_dim=32: Boundary case
    ),
    ids=[
        "sdpa_head8",
        "sdpa_head16",
        "sdpa_head32_boundary",
    ],
)
def test_nlp_create_qkv_heads_with_sdpa_integration(
    batch,
    seq_len,
    head_dim,
    num_heads,
    device,
):
    """
    Integration test: Verify nlp_create_qkv_heads outputs work with SDPA

    Tests that outputs from nlp_create_qkv_heads (especially with head_dim < 32)
    can be consumed by downstream operations like scaled_dot_product_attention
    without crashes or errors. This catches potential memory layout issues.
    """
    run_nlp_create_qkv_heads_with_sdpa_test(
        batch,
        seq_len,
        head_dim,
        num_heads,
        device,
    )


@pytest.mark.parametrize(
    "out_mem_config",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "in_mem_config",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["in_DRAM", "in_L1"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16, ttnn.bfloat8_b),
    ids=["BFLOAT16", "BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "batch, seq_len, head_dim, num_heads",
    (
        # Critical test cases: head_dim < 32 (TILE_WIDTH)
        (1, 32, 16, 4),    # head_dim=16: Triggers bug
        (1, 64, 16, 4),    # head_dim=16: Different seq_len
        (1, 32, 8, 8),     # head_dim=8: Even smaller
        (2, 32, 16, 2),    # head_dim=16: Different batch/heads
        # Edge case: head_dim=32 (should work regardless)
        (1, 32, 32, 4),    # head_dim=32: Control case
    ),
    ids=[
        "batch1_seq32_head16_nheads4",
        "batch1_seq64_head16_nheads4",
        "batch1_seq32_head8_nheads8",
        "batch2_seq32_head16_nheads2",
        "batch1_seq32_head32_nheads4_control",
    ],
)
def test_nlp_create_qkv_heads_small_head_dim(
    batch,
    seq_len,
    head_dim,
    num_heads,
    dtype,
    in_mem_config,
    out_mem_config,
    device,
):
    """
    Test nlp_create_qkv_heads with head_dim < TILE_WIDTH (32)

    This test specifically targets the bug where integer division
    q_num_tiles = num_heads * (head_dim / TILE_WIDTH) returns 0
    when head_dim < 32.
    """
    run_nlp_create_qkv_heads_small_head_dim_test(
        batch,
        seq_len,
        head_dim,
        num_heads,
        dtype,
        in_mem_config,
        out_mem_config,
        device,
    )
