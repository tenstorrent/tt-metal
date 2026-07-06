# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Extended tests for scaled_dot_product_attention — focused shape/parameter
coverage. Keeps the test matrix small: only shapes that work with the current
single-block kernel (S=32, D=32), plus mask and scale parameter variations.

Multi-block shapes (S > 32 or D > 32) are covered by the acceptance test suite
and the golden suite; they currently hang due to a known CB sync issue
(see verification_report.md, Refinement 1).
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


@pytest.mark.parametrize(
    "B, H, S, D, S_kv",
    [
        pytest.param(1, 1, 32, 32, 32, id="self_1x1x32x32"),
        pytest.param(1, 4, 32, 32, 32, id="self_1x4x32x32"),
        pytest.param(2, 4, 32, 32, 32, id="self_2x4x32x32"),
        pytest.param(1, 1, 32, 32, 32, id="cross_same_shape"),  # same shape, self
    ],
)
def test_sdpa_extended_shapes(device, B, H, S, D, S_kv):
    """Test SDPA with various batch/head configs on single-tile shapes."""
    torch.manual_seed(42)
    dtype = torch.bfloat16

    q = torch.randn(B, H, S, D, dtype=dtype)
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


def test_sdpa_extended_explicit_scale(device):
    """Test SDPA with explicit scale on single-tile shape."""
    torch.manual_seed(42)
    B, H, S, D = 1, 1, 32, 32
    scale = 0.125

    q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, dtype=torch.bfloat16)

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


def test_sdpa_extended_custom_mask(device):
    """Test SDPA with a custom additive mask on single-tile shape."""
    torch.manual_seed(42)
    B, H, S, D = 1, 1, 32, 32

    q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, dtype=torch.bfloat16)

    # Triangular mask (causal pattern via additive mask)
    mask = torch.zeros(B, 1, S, S, dtype=torch.bfloat16)
    for i in range(S):
        for j in range(S):
            if j > i:
                mask[:, :, i, j] = float("-inf")

    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    q_t = ttnn.from_torch(
        q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    k_t = ttnn.from_torch(
        k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    v_t = ttnn.from_torch(
        v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    mask_t = ttnn.from_torch(
        mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output = scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=mask_t)
    output_torch = ttnn.to_torch(output)
    assert_with_pcc(ref, output_torch, pcc=0.995)
