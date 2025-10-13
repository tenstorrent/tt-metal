# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Regression tests for nlp_create_qkv_heads to ensure the bug fix doesn't break existing functionality.

These tests explicitly cover combinations of features with existing "working" head_dim values
(64, 96, 128) to ensure our ceiling division fix doesn't introduce regressions.

TEST STRATEGY:
- Run these tests BEFORE applying the bug fix (should PASS)
- Run these tests AFTER applying the bug fix (should still PASS)
- Any failures indicate a regression introduced by the fix
"""

import pytest
from loguru import logger
import torch
import ttnn
from models.common.utility_functions import tt2torch_tensor, comp_pcc, is_grayskull


def run_nlp_create_qkv_heads_regression_test(
    batch,
    seq_len,
    head_dim,
    num_q_heads,
    num_kv_heads,
    transpose_k_heads,
    read_from_input_tensor_kv,
    dtype,
    in_mem_config,
    out_mem_config,
    device,
):
    """
    Comprehensive regression test for nlp_create_qkv_heads.

    Tests various feature combinations with "known good" head_dim values.
    """
    torch.manual_seed(1234)

    logger.info(
        f"Regression test: head_dim={head_dim}, transpose_k={transpose_k_heads}, "
        f"separate_kv={read_from_input_tensor_kv}, {num_q_heads}Q:{num_kv_heads}KV"
    )

    if read_from_input_tensor_kv:
        in0_shape = [batch, 1, seq_len, num_q_heads * head_dim]
        in1_shape = [batch, 1, seq_len, 2 * num_kv_heads * head_dim]
        A = torch.randn(in0_shape)
        B = torch.randn(in1_shape)
        in0_t = ttnn.Tensor(A, dtype).to(ttnn.TILE_LAYOUT).to(device, in_mem_config)
        in1_t = ttnn.Tensor(B, dtype).to(ttnn.TILE_LAYOUT).to(device, in_mem_config)
    else:
        in0_shape = [batch, 1, seq_len, (num_q_heads + 2 * num_kv_heads) * head_dim]
        A = torch.randn(in0_shape)
        in0_t = ttnn.Tensor(A, dtype).to(ttnn.TILE_LAYOUT).to(device, in_mem_config)

    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        in0_t,
        in1_t if read_from_input_tensor_kv else None,
        num_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        transpose_k_heads=transpose_k_heads,
        memory_config=out_mem_config,
    )

    # Check shapes
    expected_q_shape = [batch, num_q_heads, seq_len, head_dim]
    if transpose_k_heads:
        expected_k_shape = [batch, num_kv_heads, head_dim, seq_len]
    else:
        expected_k_shape = [batch, num_kv_heads, seq_len, head_dim]
    expected_v_shape = [batch, num_kv_heads, seq_len, head_dim]

    assert list(q.padded_shape) == expected_q_shape, \
        f"Q shape mismatch! Expected {expected_q_shape}, got {list(q.padded_shape)}"
    assert list(k.padded_shape) == expected_k_shape, \
        f"K shape mismatch! Expected {expected_k_shape}, got {list(k.padded_shape)}"
    assert list(v.padded_shape) == expected_v_shape, \
        f"V shape mismatch! Expected {expected_v_shape}, got {list(v.padded_shape)}"

    # Verify numerical correctness
    pyt_got_back_rm_q = tt2torch_tensor(q)
    pyt_got_back_rm_k = tt2torch_tensor(k)
    pyt_got_back_rm_v = tt2torch_tensor(v)

    if read_from_input_tensor_kv:
        ref_q = A
        (ref_k, ref_v) = torch.split(B, [num_kv_heads * head_dim, num_kv_heads * head_dim], dim=-1)
    else:
        (ref_q, ref_k, ref_v) = torch.split(
            A, [num_q_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim], dim=-1
        )

    ref_q = torch.reshape(ref_q, [batch, seq_len, num_q_heads, head_dim]).transpose(-3, -2)
    ref_k = torch.reshape(ref_k, [batch, seq_len, num_kv_heads, head_dim]).transpose(-3, -2)
    ref_v = torch.reshape(ref_v, [batch, seq_len, num_kv_heads, head_dim]).transpose(-3, -2)

    if transpose_k_heads:
        ref_k = ref_k.transpose(-2, -1)

    if dtype == ttnn.bfloat8_b:
        pcc = 0.99
    else:
        pcc = 1.0

    passing_pcc_q, output_pcc_q = comp_pcc(pyt_got_back_rm_q, ref_q, pcc)
    passing_pcc_k, output_pcc_k = comp_pcc(pyt_got_back_rm_k, ref_k, pcc)
    passing_pcc_v, output_pcc_v = comp_pcc(pyt_got_back_rm_v, ref_v, pcc)

    assert passing_pcc_q, f"Q PCC check failed: {output_pcc_q}"
    assert passing_pcc_k, f"K PCC check failed: {output_pcc_k}"
    assert passing_pcc_v, f"V PCC check failed: {output_pcc_v}"

    logger.info(f"  Q PCC: {output_pcc_q:.6f}, K PCC: {output_pcc_k:.6f}, V PCC: {output_pcc_v:.6f}")


# ============================================================================
# REGRESSION TEST 1: Transpose with all existing head_dims
# ============================================================================
@pytest.mark.parametrize(
    "out_mem_config",
    (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "in_mem_config",
    (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    ids=["in_DRAM", "in_L1"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16, ttnn.bfloat8_b),
    ids=["BFLOAT16", "BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "head_dim, transpose_k",
    (
        (64, True),   # Most common - MUST NOT BREAK
        (64, False),  # Control
        (96, True),   # From existing tests - MUST NOT BREAK
        (96, False),  # Control
        (128, True),  # Llama - MUST NOT BREAK
        (128, False), # Control
    ),
    ids=[
        "head64_transpose",
        "head64_no_transpose",
        "head96_transpose",
        "head96_no_transpose",
        "head128_transpose",
        "head128_no_transpose",
    ],
)
def test_nlp_create_qkv_heads_regression_transpose(
    head_dim,
    transpose_k,
    dtype,
    in_mem_config,
    out_mem_config,
    device,
):
    """
    Regression test: Verify transpose works correctly with existing head_dims.

    CRITICAL: These MUST pass both before and after the bug fix.
    """
    run_nlp_create_qkv_heads_regression_test(
        batch=1,
        seq_len=128,
        head_dim=head_dim,
        num_q_heads=8,
        num_kv_heads=8,
        transpose_k_heads=transpose_k,
        read_from_input_tensor_kv=False,
        dtype=dtype,
        in_mem_config=in_mem_config,
        out_mem_config=out_mem_config,
        device=device,
    )


# ============================================================================
# REGRESSION TEST 2: Separate KV tensor with all existing head_dims
# ============================================================================
@pytest.mark.parametrize(
    "out_mem_config",
    (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "in_mem_config",
    (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    ids=["in_DRAM", "in_L1"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16, ttnn.bfloat8_b),
    ids=["BFLOAT16", "BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "head_dim",
    (64, 96, 128),
    ids=["head64", "head96", "head128"],
)
def test_nlp_create_qkv_heads_regression_separate_kv(
    head_dim,
    dtype,
    in_mem_config,
    out_mem_config,
    device,
):
    """
    Regression test: Verify separate KV tensor works with existing head_dims.

    CRITICAL: These MUST pass both before and after the bug fix.
    """
    run_nlp_create_qkv_heads_regression_test(
        batch=1,
        seq_len=128,
        head_dim=head_dim,
        num_q_heads=8,
        num_kv_heads=8,
        transpose_k_heads=False,
        read_from_input_tensor_kv=True,
        dtype=dtype,
        in_mem_config=in_mem_config,
        out_mem_config=out_mem_config,
        device=device,
    )


# ============================================================================
# REGRESSION TEST 3: GQA with all existing head_dims
# ============================================================================
@pytest.mark.parametrize(
    "out_mem_config",
    (ttnn.DRAM_MEMORY_CONFIG,),
    ids=["out_DRAM"],
)
@pytest.mark.parametrize(
    "in_mem_config",
    (ttnn.DRAM_MEMORY_CONFIG,),
    ids=["in_DRAM"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16,),
    ids=["BFLOAT16"],
)
@pytest.mark.parametrize(
    "head_dim, num_q_heads, num_kv_heads",
    (
        (64, 16, 2),   # GQA 8:1 - common ratio
        (64, 32, 4),   # GQA 8:1 - Llama style with head_dim=64
        (96, 8, 2),    # GQA 4:1 - with head_dim=96
        (128, 32, 4),  # GQA 8:1 - Llama 2/3 actual config
        (128, 32, 8),  # GQA 4:1 - Llama variant
    ),
    ids=[
        "head64_gqa_16q_2kv",
        "head64_gqa_32q_4kv",
        "head96_gqa_8q_2kv",
        "head128_gqa_32q_4kv_llama",
        "head128_gqa_32q_8kv",
    ],
)
def test_nlp_create_qkv_heads_regression_gqa(
    head_dim,
    num_q_heads,
    num_kv_heads,
    dtype,
    in_mem_config,
    out_mem_config,
    device,
):
    """
    Regression test: Verify GQA works correctly with existing head_dims.

    CRITICAL: These MUST pass both before and after the bug fix.
    Tests various Q:KV ratios with known good head_dims.
    """
    run_nlp_create_qkv_heads_regression_test(
        batch=1,
        seq_len=128,
        head_dim=head_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        transpose_k_heads=False,
        read_from_input_tensor_kv=False,
        dtype=dtype,
        in_mem_config=in_mem_config,
        out_mem_config=out_mem_config,
        device=device,
    )


# ============================================================================
# REGRESSION TEST 4: Feature combinations (stress test)
# ============================================================================
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16,),
    ids=["BFLOAT16"],
)
@pytest.mark.parametrize(
    "head_dim, num_q_heads, num_kv_heads, transpose_k, separate_kv",
    (
        # head_dim=64 combinations
        (64, 16, 2, True, False),   # GQA + transpose
        (64, 16, 2, False, True),   # GQA + separate_kv
        (64, 8, 8, True, True),     # transpose + separate_kv
        (64, 16, 2, True, True),    # ALL features

        # head_dim=96 combinations
        (96, 8, 2, True, False),    # GQA + transpose
        (96, 8, 2, False, True),    # GQA + separate_kv

        # head_dim=128 combinations
        (128, 32, 4, True, False),  # Llama GQA + transpose
        (128, 32, 4, False, True),  # Llama GQA + separate_kv
        (128, 16, 16, True, True),  # All features
    ),
    ids=[
        "head64_gqa_transpose",
        "head64_gqa_sepkv",
        "head64_transpose_sepkv",
        "head64_all_features",
        "head96_gqa_transpose",
        "head96_gqa_sepkv",
        "head128_llama_gqa_transpose",
        "head128_llama_gqa_sepkv",
        "head128_all_features",
    ],
)
def test_nlp_create_qkv_heads_regression_feature_combinations(
    head_dim,
    num_q_heads,
    num_kv_heads,
    transpose_k,
    separate_kv,
    dtype,
    device,
):
    """
    Regression test: Verify feature combinations work correctly.

    CRITICAL STRESS TEST: These test multiple features together.
    MUST pass both before and after the bug fix.
    """
    run_nlp_create_qkv_heads_regression_test(
        batch=1,
        seq_len=128,
        head_dim=head_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        transpose_k_heads=transpose_k,
        read_from_input_tensor_kv=separate_kv,
        dtype=dtype,
        in_mem_config=ttnn.DRAM_MEMORY_CONFIG,
        out_mem_config=ttnn.DRAM_MEMORY_CONFIG,
        device=device,
    )


# ============================================================================
# REGRESSION TEST 5: Boundary case head_dim=32
# ============================================================================
@pytest.mark.parametrize(
    "out_mem_config",
    (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "in_mem_config",
    (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    ids=["in_DRAM", "in_L1"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16, ttnn.bfloat8_b),
    ids=["BFLOAT16", "BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "transpose_k, separate_kv",
    (
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ),
    ids=["basic", "transpose", "separate_kv", "all_features"],
)
def test_nlp_create_qkv_heads_regression_head_dim_32_boundary(
    transpose_k,
    separate_kv,
    dtype,
    in_mem_config,
    out_mem_config,
    device,
):
    """
    Regression test: head_dim=32 (exactly TILE_WIDTH).

    BOUNDARY CASE: head_dim = 32 should work BOTH before and after fix.
    This is the boundary between buggy (<32) and working (>=32) regions.
    """
    run_nlp_create_qkv_heads_regression_test(
        batch=1,
        seq_len=128,
        head_dim=32,
        num_q_heads=8,
        num_kv_heads=8,
        transpose_k_heads=transpose_k,
        read_from_input_tensor_kv=separate_kv,
        dtype=dtype,
        in_mem_config=in_mem_config,
        out_mem_config=out_mem_config,
        device=device,
    )


# ============================================================================
# REGRESSION TEST 6: Large dimensions (sanity check)
# ============================================================================
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16,),
    ids=["BFLOAT16"],
)
@pytest.mark.parametrize(
    "batch, seq_len, head_dim",
    (
        (8, 256, 64),    # Larger batch
        (1, 2048, 64),   # Long sequence
        (1, 1024, 128),  # Llama-style long
        (2, 512, 96),    # Mixed large
    ),
    ids=["large_batch", "long_seq_64", "long_seq_128", "mixed_large"],
)
def test_nlp_create_qkv_heads_regression_large_dimensions(
    batch,
    seq_len,
    head_dim,
    dtype,
    device,
):
    """
    Regression test: Large dimension combinations.

    SANITY CHECK: Ensure fix doesn't break with larger tensors.
    """
    run_nlp_create_qkv_heads_regression_test(
        batch=batch,
        seq_len=seq_len,
        head_dim=head_dim,
        num_q_heads=16,
        num_kv_heads=16,
        transpose_k_heads=False,
        read_from_input_tensor_kv=False,
        dtype=dtype,
        in_mem_config=ttnn.DRAM_MEMORY_CONFIG,
        out_mem_config=ttnn.DRAM_MEMORY_CONFIG,
        device=device,
    )
