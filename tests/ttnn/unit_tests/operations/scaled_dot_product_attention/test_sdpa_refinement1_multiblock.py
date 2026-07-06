# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1 — Multi-block kernel fix: refinement-specific tests.

Tests that exercise multi-block shapes (S > 32, D > 32) directly, verifying
no hangs and correct numerical output. These shapes were hanging before
Refinement 1 due to a CB write-pointer alignment issue in the PV matmul.
"""

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


@pytest.mark.parametrize(
    "B, H, S_q, D, S_kv",
    [
        # Multi-KV-block: S > 32
        pytest.param(1, 1, 64, 64, 64, id="multi_kv_64x64"),
        pytest.param(1, 1, 128, 64, 128, id="multi_kv_128x64"),
        pytest.param(1, 1, 256, 64, 256, id="multi_kv_256x64"),
        pytest.param(1, 1, 512, 64, 512, id="multi_kv_512x64"),
        pytest.param(1, 1, 1024, 64, 1024, id="multi_kv_1024x64"),
        pytest.param(1, 1, 2048, 64, 2048, id="multi_kv_2048x64"),
        # Multi-Q-block: D > 32
        pytest.param(1, 1, 128, 128, 128, id="multi_d_128x128"),
        pytest.param(1, 1, 128, 256, 128, id="multi_d_128x256"),
        # Multi-head + multi-block
        pytest.param(1, 4, 128, 64, 128, id="multi_head_4_128x64"),
        pytest.param(1, 8, 256, 64, 256, id="multi_head_8_256x64"),
        pytest.param(1, 12, 128, 64, 128, id="multi_head_12_128x64"),
        pytest.param(1, 32, 128, 128, 128, id="multi_head_32_128x128"),
        # Multi-batch + multi-block
        pytest.param(2, 4, 128, 64, 128, id="multi_batch_2x4_128x64"),
        pytest.param(4, 8, 128, 64, 128, id="multi_batch_4x8_128x64"),
        pytest.param(4, 8, 256, 64, 256, id="multi_batch_4x8_256x64"),
        # Cross-attention (S_q != S_kv)
        pytest.param(1, 4, 64, 64, 128, id="cross_attn_64_to_128"),
        pytest.param(1, 4, 128, 64, 64, id="cross_attn_128_to_64"),
        pytest.param(1, 12, 128, 64, 512, id="cross_attn_128_to_512"),
        # Long context
        pytest.param(1, 1, 4096, 64, 4096, id="long_context_4096"),
        pytest.param(1, 1, 2048, 64, 2048, id="long_context_2048"),
    ],
)
def test_multiblock_no_mask(device, B, H, S_q, D, S_kv):
    """Multi-block SDPA without mask — verifies no hang and correct output."""
    torch.manual_seed(42)
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

    assert_with_pcc(ref, output_torch, pcc=0.995)


@pytest.mark.parametrize(
    "B, H, S_q, D, S_kv",
    [
        pytest.param(1, 1, 128, 64, 128, id="explicit_scale_128x64"),
        pytest.param(1, 4, 128, 64, 128, id="explicit_scale_multihead"),
        pytest.param(1, 1, 256, 64, 256, id="explicit_scale_256x64"),
        pytest.param(2, 4, 128, 64, 128, id="explicit_scale_multibatch"),
    ],
)
def test_multiblock_explicit_scale(device, B, H, S_q, D, S_kv):
    """Multi-block SDPA with explicit scale — verifies scale is applied correctly."""
    torch.manual_seed(42)
    dtype = torch.bfloat16
    scale = 0.125

    q = torch.randn(B, H, S_q, D, dtype=dtype)
    k = torch.randn(B, H, S_kv, D, dtype=dtype)
    v = torch.randn(B, H, S_kv, D, dtype=dtype)

    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)

    q_t = ttnn.from_torch(
        q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    k_t = ttnn.from_torch(
        k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    v_t = ttnn.from_torch(
        v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output = scaled_dot_product_attention(q_t, k_t, v_t, scale=scale)
    output_torch = ttnn.to_torch(output)

    assert_with_pcc(ref, output_torch, pcc=0.995)
