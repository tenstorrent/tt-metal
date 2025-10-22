# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Test for nlp_create_qkv_heads_decode bug when head_dim < TILE_WIDTH (32)

BUG: Integer division bug in nlp_create_qkv_heads_decode_program_factory.cpp causes
incorrect output shapes when head_dim < 32.

Root cause: head_tiles = head_dim / TILE_WIDTH
When head_dim=16: head_tiles = 0, head_size = 0 (causes memory calculation errors!)

Expected: head_tiles = (head_dim + TILE_WIDTH - 1) / TILE_WIDTH

This test demonstrates the bug with head_dim=16 and verifies the fix for the decode variant.
"""

import pytest
from loguru import logger
import torch
import ttnn
from models.common.utility_functions import tt2torch_tensor, comp_pcc


def run_nlp_create_qkv_heads_decode_small_head_dim_test(
    batch,
    seq_len,
    head_dim,
    num_q_heads,
    num_kv_heads,
    dtype,
    device,
):
    """
    Test nlp_create_qkv_heads_decode with head_dim < 32 (TILE_WIDTH)

    This reproduces a bug where integer division causes incorrect tile calculations
    in the decode operation (used for inference with batch_size=1, seq_len=1).
    """
    torch.manual_seed(1234)

    # Decode operation: batch=1, seq_len=1 (single token generation)
    # Input shape: [batch, 1, 1, (num_q_heads + 2*num_kv_heads) * head_dim]
    qkv_width = (num_q_heads + 2 * num_kv_heads) * head_dim

    logger.info(f"Testing decode variant with head_dim={head_dim} < TILE_WIDTH(32)")
    logger.info(
        f"batch={batch}, seq_len={seq_len}, num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}"
    )

    # Create fused QKV tensor for decode: [batch, 1, seq_len, qkv_width]
    # For decode, seq_len is typically 1 (single token)
    in0_shape = [batch, 1, seq_len, qkv_width]
    A = torch.randn(in0_shape)

    # Create sharded input tensor (decode typically uses sharding)
    # For decode, we need to create output tensors with proper sharding
    q_shape = ttnn.Shape([batch, num_q_heads, seq_len, head_dim])
    kv_shape = ttnn.Shape([batch, num_kv_heads, seq_len, head_dim])

    # Create shard specs for decode operation
    # Use simple height sharding for batch dimension
    compute_grid_size = device.compute_with_storage_grid_size()
    num_cores = min(batch, compute_grid_size.x * compute_grid_size.y)

    shard_shape = [batch // num_cores, num_q_heads * seq_len * head_dim]
    q_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))}),
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
        False,
    )

    kv_shard_shape = [batch // num_cores, num_kv_heads * seq_len * head_dim]
    kv_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))}),
        kv_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
        False,
    )

    q_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, q_shard_spec
    )
    kv_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, kv_shard_spec
    )

    # Create output tensors
    q_output = ttnn.allocate_tensor_on_device(q_shape, dtype, ttnn.TILE_LAYOUT, device, q_mem_config)
    k_output = ttnn.allocate_tensor_on_device(kv_shape, dtype, ttnn.TILE_LAYOUT, device, kv_mem_config)
    v_output = ttnn.allocate_tensor_on_device(kv_shape, dtype, ttnn.TILE_LAYOUT, device, kv_mem_config)

    # Create input tensor (interleaved for this test)
    in0_t = ttnn.Tensor(A, dtype).to(ttnn.TILE_LAYOUT).to(device, ttnn.DRAM_MEMORY_CONFIG)

    # Call nlp_create_qkv_heads_decode - this triggers the bug when head_dim < 32
    # Use interleaved input variant for simplicity in testing
    q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
        in0_t,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        output_tensors=[q_output, k_output, v_output],
    )

    # CRITICAL: Check output shapes - this is where the bug manifests
    expected_q_shape = [batch, num_q_heads, seq_len, head_dim]
    expected_kv_shape = [batch, num_kv_heads, seq_len, head_dim]

    logger.info(f"Expected Q shape: {expected_q_shape}")
    logger.info(f"Q output shape: {list(q.padded_shape)}")
    logger.info(f"Expected K shape: {expected_kv_shape}")
    logger.info(f"K output shape: {list(k.padded_shape)}")
    logger.info(f"Expected V shape: {expected_kv_shape}")
    logger.info(f"V output shape: {list(v.padded_shape)}")

    # Verify shapes are correct
    assert list(q.padded_shape) == expected_q_shape, f"Q shape mismatch! Expected {expected_q_shape}, got {list(q.padded_shape)}"
    assert list(k.padded_shape) == expected_kv_shape, f"K shape mismatch! Expected {expected_kv_shape}, got {list(k.padded_shape)}"
    assert list(v.padded_shape) == expected_kv_shape, f"V shape mismatch! Expected {expected_kv_shape}, got {list(v.padded_shape)}"

    # Verify numerical correctness
    pyt_got_back_rm_q = tt2torch_tensor(q)
    pyt_got_back_rm_k = tt2torch_tensor(k)
    pyt_got_back_rm_v = tt2torch_tensor(v)

    # Split reference QKV
    # For decode: Q has num_q_heads * head_dim, K and V each have num_kv_heads * head_dim
    q_width = num_q_heads * head_dim
    kv_width = num_kv_heads * head_dim
    (ref_q, ref_k, ref_v) = torch.split(A, [q_width, kv_width, kv_width], dim=-1)

    # Reshape to match expected output: [batch, num_heads, seq_len, head_dim]
    ref_q = torch.reshape(ref_q, [batch, seq_len, num_q_heads, head_dim]).transpose(-3, -2)
    ref_k = torch.reshape(ref_k, [batch, seq_len, num_kv_heads, head_dim]).transpose(-3, -2)
    ref_v = torch.reshape(ref_v, [batch, seq_len, num_kv_heads, head_dim]).transpose(-3, -2)

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

    logger.info("Decode test passed - bug is fixed")


@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16, ttnn.bfloat8_b),
    ids=["BFLOAT16", "BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "batch, seq_len, head_dim, num_q_heads, num_kv_heads",
    (
        # Critical test cases: head_dim < 32 (TILE_WIDTH)
        # Decode typically uses batch=32 (parallel decode), seq_len=1
        (32, 1, 16, 8, 1),  # head_dim=16: Triggers bug, GQA (8:1 ratio)
        (32, 1, 16, 4, 4),  # head_dim=16: Triggers bug, MHA
        (32, 1, 8, 8, 2),  # head_dim=8: Even smaller, GQA
        (32, 1, 8, 8, 8),  # head_dim=8: Even smaller, MHA
        # Edge case: head_dim=32 (boundary - should work regardless)
        (32, 1, 32, 8, 1),  # head_dim=32: Control case, GQA
        (32, 1, 32, 4, 4),  # head_dim=32: Control case, MHA
    ),
    ids=[
        "batch32_seq1_head16_q8_kv1_gqa",
        "batch32_seq1_head16_q4_kv4_mha",
        "batch32_seq1_head8_q8_kv2_gqa",
        "batch32_seq1_head8_q8_kv8_mha",
        "batch32_seq1_head32_q8_kv1_gqa_control",
        "batch32_seq1_head32_q4_kv4_mha_control",
    ],
)
def test_nlp_create_qkv_heads_decode_small_head_dim(
    batch,
    seq_len,
    head_dim,
    num_q_heads,
    num_kv_heads,
    dtype,
    device,
):
    """
    Test nlp_create_qkv_heads_decode with head_dim < TILE_WIDTH (32)

    This test specifically targets the bug where integer division
    head_tiles = head_dim / TILE_WIDTH returns 0 when head_dim < 32,
    causing head_size = 0 and incorrect memory calculations.
    """
    run_nlp_create_qkv_heads_decode_small_head_dim_test(
        batch,
        seq_len,
        head_dim,
        num_q_heads,
        num_kv_heads,
        dtype,
        device,
    )
