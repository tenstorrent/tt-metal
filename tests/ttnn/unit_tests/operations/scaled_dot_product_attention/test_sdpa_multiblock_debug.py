# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic debug test for SDPA multi-block hang.

DO NOT DELETE — documents the debugging process for Refinement 1.

Tests specific multi-block scenarios with hand-calculable inputs to isolate
where the compute kernel hangs during multi-block iteration.
"""

import pytest
import torch
import ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def test_multi_kv_block_no_hang(device):
    """Multi-KV-block: S=128, D=64 → S_kv_t=4 KV blocks, D_t=2.

    This is the minimal shape that hangs: single Q block (S_q_t=4 with B_q=1
    means 4 Q blocks), 4 KV blocks per Q block, D_t=2.
    """
    torch.manual_seed(42)
    B, H, S_q, D = 1, 1, 128, 64
    S_kv = S_q
    dtype = torch.bfloat16

    q = torch.randn(B, H, S_q, D, dtype=dtype)
    k = torch.randn(B, H, S_kv, D, dtype=dtype)
    v = torch.randn(B, H, S_kv, D, dtype=dtype)

    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    q_t = ttnn.from_torch(
        q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    k_t = ttnn.from_torch(
        k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    v_t = ttnn.from_torch(
        v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output = scaled_dot_product_attention(q_t, k_t, v_t)
    output_torch = ttnn.to_torch(output)

    from tests.ttnn.utils_for_testing import assert_with_pcc

    assert_with_pcc(ref, output_torch, pcc=0.995)


def test_multi_q_block_no_hang(device):
    """Multi-Q-block: S_q=128, D=32 → S_q_t=4 Q blocks, D_t=1.

    Minimal multi-Q-block test.
    """
    torch.manual_seed(42)
    B, H, S_q, D = 1, 1, 128, 32
    S_kv = S_q
    dtype = torch.bfloat16

    q = torch.randn(B, H, S_q, D, dtype=dtype)
    k = torch.randn(B, H, S_kv, D, dtype=dtype)
    v = torch.randn(B, H, S_kv, D, dtype=dtype)

    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    q_t = ttnn.from_torch(
        q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    k_t = ttnn.from_torch(
        k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    v_t = ttnn.from_torch(
        v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output = scaled_dot_product_attention(q_t, k_t, v_t)
    output_torch = ttnn.to_torch(output)

    from tests.ttnn.utils_for_testing import assert_with_pcc

    assert_with_pcc(ref, output_torch, pcc=0.995)


def test_multi_head_d32_no_hang(device):
    """Multi-head with D=32: B=1, H=4, S=128, D=32.

    Multiple heads means multiple cores are active.
    """
    torch.manual_seed(42)
    B, H, S_q, D = 1, 4, 128, 32
    S_kv = S_q
    dtype = torch.bfloat16

    q = torch.randn(B, H, S_q, D, dtype=dtype)
    k = torch.randn(B, H, S_kv, D, dtype=dtype)
    v = torch.randn(B, H, S_kv, D, dtype=dtype)

    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    q_t = ttnn.from_torch(
        q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    k_t = ttnn.from_torch(
        k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    v_t = ttnn.from_torch(
        v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output = scaled_dot_product_attention(q_t, k_t, v_t)
    output_torch = ttnn.to_torch(output)

    from tests.ttnn.utils_for_testing import assert_with_pcc

    assert_with_pcc(ref, output_torch, pcc=0.995)
