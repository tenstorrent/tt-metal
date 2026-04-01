# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone repro for SDPA decode causal mask padding leak.

Bug: paged_scaled_dot_product_attention_decode (and the non-paged version)
does not properly mask positions beyond cur_pos in cache blocks. Non-zero
data at positions cur_pos+1 through block_size-1 leaks through the causal
mask and affects the attention output.

This causes KV cache bleed in batched vLLM inference — users see content
from other users' responses mixed into theirs.

The bug occurs:
  - On both Wormhole and Blackhole
  - For ALL cur_pos values (tested 4, 8, 14, 15, 16, 24, 30)
  - For both paged and non-paged SDPA decode
  - With block_size=32 (= TILE_HEIGHT), so has_block_padding=false
  - The block padding mask fix from #30362 / cc70746c does NOT apply

Related: https://github.com/tenstorrent/tt-xla/issues/3899

Run: python tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_decode_causal_mask_leak.py
"""

import torch
import ttnn


def test_causal_mask_padding_leak():
    device = ttnn.open_device(device_id=0)

    num_users = 2
    num_kv_heads = 8
    head_dim = 64
    block_size = 32  # == TILE_HEIGHT, so has_block_padding=false
    num_blocks = num_users
    real_seq_len = 14  # cur_pos for first decode step

    torch.manual_seed(42)
    q = torch.randn(1, num_users, num_kv_heads, head_dim, dtype=torch.bfloat16)
    page_table = torch.arange(num_users, dtype=torch.int32).unsqueeze(1)
    cur_pos = torch.tensor([real_seq_len] * num_users, dtype=torch.int32)

    # Build cache with real data at [0, real_seq_len) and non-zero padding at
    # [real_seq_len, block_size). The padding simulates what vLLM's
    # min_context_len produces when it pads shorter prompts to block_size.
    k_cache = torch.zeros(num_blocks, num_kv_heads, block_size, head_dim, dtype=torch.bfloat16)
    v_cache = torch.zeros(num_blocks, num_kv_heads, block_size, head_dim, dtype=torch.bfloat16)
    for u in range(num_users):
        torch.manual_seed(1000 + u)
        k_cache[u, :, :real_seq_len, :] = torch.randn(
            num_kv_heads, real_seq_len, head_dim
        ).bfloat16()
        v_cache[u, :, :real_seq_len, :] = torch.randn(
            num_kv_heads, real_seq_len, head_dim
        ).bfloat16()
        # Non-zero padding — same value at all padding positions per user
        pad = torch.randn(num_kv_heads, 1, head_dim).bfloat16() * 2.0
        k_cache[u, :, real_seq_len:, :] = pad.expand(-1, block_size - real_seq_len, -1)
        v_cache[u, :, real_seq_len:, :] = pad.expand(-1, block_size - real_seq_len, -1)

    # Clean cache: same real data, zeros at padding positions
    k_clean = k_cache.clone()
    v_clean = v_cache.clone()
    for u in range(num_users):
        k_clean[u, :, real_seq_len:, :] = 0
        v_clean[u, :, real_seq_len:, :] = 0

    # Move to device
    q_tt = ttnn.from_torch(q, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    pt_tt = ttnn.from_torch(page_table, device=device, dtype=ttnn.int32)
    cp_tt = ttnn.from_torch(cur_pos, device=device, dtype=ttnn.int32)

    # Run with dirty cache (non-zero padding)
    out_dirty = ttnn.to_torch(
        ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_tt,
            ttnn.from_torch(k_cache, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
            ttnn.from_torch(v_cache, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
            pt_tt,
            cur_pos_tensor=cp_tt,
            is_causal=True,
        )
    )

    # Run with clean cache (zeros at padding)
    out_clean = ttnn.to_torch(
        ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_tt,
            ttnn.from_torch(k_clean, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
            ttnn.from_torch(v_clean, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
            pt_tt,
            cur_pos_tensor=cp_tt,
            is_causal=True,
        )
    )

    diff = (out_dirty.float() - out_clean.float()).abs()
    max_diff = diff.max().item()

    print(f"Paged SDPA decode causal mask leak test")
    print(f"  block_size={block_size}, cur_pos={real_seq_len}, num_users={num_users}")
    print(f"  max_diff between dirty and clean padding: {max_diff:.4f}")
    print(f"  Expected: ~0 (causal mask should exclude positions {real_seq_len + 1}-{block_size - 1})")
    print(f"  Result: {'FAIL — padding leaks through causal mask' if max_diff > 0.1 else 'PASS'}")

    # Also test non-paged version
    k_np = k_cache[:num_users]
    v_np = v_cache[:num_users]
    k_np_clean = k_clean[:num_users]
    v_np_clean = v_clean[:num_users]

    out_np_dirty = ttnn.to_torch(
        ttnn.transformer.scaled_dot_product_attention_decode(
            q_tt,
            ttnn.from_torch(k_np, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
            ttnn.from_torch(v_np, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
            cur_pos_tensor=cp_tt,
            is_causal=True,
        )
    )
    out_np_clean = ttnn.to_torch(
        ttnn.transformer.scaled_dot_product_attention_decode(
            q_tt,
            ttnn.from_torch(k_np_clean, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
            ttnn.from_torch(v_np_clean, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
            cur_pos_tensor=cp_tt,
            is_causal=True,
        )
    )

    np_diff = (out_np_dirty.float() - out_np_clean.float()).abs().max().item()
    print(f"\nNon-paged SDPA decode:")
    print(f"  max_diff: {np_diff:.4f}")
    print(f"  Result: {'FAIL — same bug in non-paged version' if np_diff > 0.1 else 'PASS'}")

    ttnn.close_device(device)

    assert max_diff < 0.1, f"Paged SDPA decode: padding leaks through causal mask (max_diff={max_diff:.4f})"
    assert np_diff < 0.1, f"Non-paged SDPA decode: padding leaks through causal mask (max_diff={np_diff:.4f})"


if __name__ == "__main__":
    test_causal_mask_padding_leak()
