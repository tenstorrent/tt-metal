# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the gpt-oss vLLM-style hybrid kv_cache_groups wiring.

This file does NOT stand up the full gpt-oss Model — that's covered by the
existing model-level accuracy tests. Instead it exercises the bounded-cache
plumbing in isolation:

  - ``build_hybrid_page_tables`` returns the SlidingWindowSpec-shaped
    page_table (valid prefix sized to sliding_window/block_size, zero-padded
    tail to max_seq_len/block_size) for sliding layers, and the full
    FullAttentionSpec page_table for non-sliding layers.
  - The page_table that the helper produces, fed to the three paged ops with
    ``cache_position_modulo=sliding_window``, correctly addresses a decode
    walk that crosses the sliding-window boundary multiple times.
The kernel-level "without the kwarg, positions past the boundary clobber block
0" bug is already locked in by
``tests/ttnn/unit_tests/operations/sdpa/test_bounded_sliding_kv_cache.py``;
this file focuses on the gpt-oss-specific page_table builder and end-to-end
correctness with the bounded physical pool.

A larger end-to-end check (real weights, full model forward) is intentionally
out of scope for this file; it would be slow and the failure modes it covers
are already covered by the model accuracy tests once they enable the flag.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.demos.gpt_oss.tt.attention.kv_cache_hybrid import build_hybrid_page_tables
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

# ── helpers ────────────────────────────────────────────────────────────────


def _sharded_kv_input(device, x_padded):
    """Mimic the decode-time height-sharded KV tensor that paged_update_cache wants."""
    num_users = x_padded.shape[1]
    xt = ttnn.Tensor(x_padded, ttnn.bfloat16).to(ttnn.TILE_LAYOUT)
    shard_grid = ttnn.num_cores_to_corerangeset(num_users, device.compute_with_storage_grid_size(), True)
    shard_spec = ttnn.ShardSpec(
        shard_grid,
        [xt.volume() // xt.padded_shape[-1] // num_users, xt.padded_shape[-1]],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    return xt.to(device, mem_cfg)


def _torch_sliding_ref(k_hist, v_hist, q, cur_pos, sliding_window, scale):
    """Single-user sliding-window attention reference."""
    lo = max(0, cur_pos - sliding_window + 1)
    k_win = k_hist[lo : cur_pos + 1].float()
    v_win = v_hist[lo : cur_pos + 1].float()
    q_b = q[0, 0, :, :].float()
    repeat = q_b.shape[0] // k_win.shape[1]
    k_rep = k_win.repeat_interleave(repeat, dim=1)
    v_rep = v_win.repeat_interleave(repeat, dim=1)
    scores = torch.einsum("hd,shd->hs", q_b, k_rep) * scale
    weights = torch.softmax(scores, dim=-1)
    out = torch.einsum("hs,shd->hd", weights, v_rep)
    return out.unsqueeze(0).unsqueeze(0)


# ── helper-shape tests (pure CPU) ──────────────────────────────────────────


def test_build_hybrid_page_tables_shapes_and_padding():
    """Sliding layers: first sliding_blocks entries valid (per-user contiguous),
    rest zero-padded. Full layers: all max_blocks entries valid."""
    num_layers = 4
    sliding_layers_mask = [True, False, True, False]
    num_users = 3
    block_size = 32
    sliding_window = 128
    max_seq_len = 1024
    sliding_blocks = sliding_window // block_size
    max_blocks = max_seq_len // block_size

    page_tables = build_hybrid_page_tables(
        num_layers,
        sliding_layers_mask,
        num_users=num_users,
        block_size=block_size,
        max_seq_len=max_seq_len,
        sliding_window=sliding_window,
    )

    assert len(page_tables) == num_layers
    for layer_idx, pt in enumerate(page_tables):
        assert pt.shape == (num_users, max_blocks), f"layer {layer_idx} wrong shape"
        assert pt.dtype == torch.int32

        if sliding_layers_mask[layer_idx]:
            # Per-user contiguous block IDs for the sliding window, zero-padded tail.
            for u in range(num_users):
                expected_valid = torch.arange(u * sliding_blocks, (u + 1) * sliding_blocks, dtype=torch.int32)
                assert torch.equal(pt[u, :sliding_blocks], expected_valid)
                assert torch.all(pt[u, sliding_blocks:] == 0)
        else:
            for u in range(num_users):
                expected = torch.arange(u * max_blocks, (u + 1) * max_blocks, dtype=torch.int32)
                assert torch.equal(pt[u], expected)


def test_build_hybrid_page_tables_rejects_non_multiple_sliding_window():
    with pytest.raises(ValueError, match="must be a multiple of block_size"):
        build_hybrid_page_tables(
            num_layers=1,
            sliding_layers_mask=[True],
            num_users=1,
            block_size=32,
            max_seq_len=256,
            sliding_window=100,  # not a multiple of 32
        )


def test_build_hybrid_page_tables_no_sliding_window_falls_back_to_full():
    """sliding_window=None makes every layer use the full max_blocks allocation
    regardless of the mask — there's no SlidingWindowSpec to apply."""
    pts = build_hybrid_page_tables(
        num_layers=2,
        sliding_layers_mask=[True, False],
        num_users=1,
        block_size=32,
        max_seq_len=128,
        sliding_window=None,
    )
    for pt in pts:
        assert torch.equal(pt[0], torch.arange(4, dtype=torch.int32))


# ── on-device end-to-end: bounded sliding decode round-trip ─────────────────


def _run_bounded_decode_walk(device, sliding_window, use_modulo, decode_steps):
    """Drive paged_update_cache + paged_sdpa_decode through a bounded sliding-window
    page_table built by build_hybrid_page_tables (sliding spec for layer 0)."""
    torch.manual_seed(0)

    block_size = 32
    sliding_blocks = sliding_window // block_size
    max_seq_len = 4 * sliding_window  # zero-pad ratio 4× to match vLLM behaviour
    max_blocks = max_seq_len // block_size

    num_kv_heads = 1
    num_q_heads = 1
    head_dim = 128
    num_users = 1
    PADDED_HEADS = 32
    scale = 1.0 / (head_dim**0.5)

    # Bounded physical cache: sliding_blocks * num_users blocks total.
    cache_blocks = sliding_blocks * num_users
    k_init = torch.zeros(cache_blocks, num_kv_heads, block_size, head_dim).bfloat16().float()
    v_init = torch.zeros(cache_blocks, num_kv_heads, block_size, head_dim).bfloat16().float()
    k_cache_tt = ttnn.Tensor(k_init, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    v_cache_tt = ttnn.Tensor(v_init, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    # Use the gpt-oss helper to build the per-layer page table — exactly the
    # shape vLLM's SlidingWindowSpec emits.
    pts = build_hybrid_page_tables(
        num_layers=1,
        sliding_layers_mask=[True],
        num_users=num_users,
        block_size=block_size,
        max_seq_len=max_seq_len,
        sliding_window=sliding_window,
    )
    page_table = pts[0]
    assert page_table.shape == (num_users, max_blocks)
    page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)

    k_hist = torch.zeros(decode_steps, num_kv_heads, head_dim).bfloat16().float()
    v_hist = torch.zeros(decode_steps, num_kv_heads, head_dim).bfloat16().float()

    modulo_kwargs = {"cache_position_modulo": sliding_window} if use_modulo else {}
    results = []
    for pos in range(decode_steps):
        K_new = torch.randn(1, 1, num_kv_heads, head_dim).bfloat16().float()
        V_new = torch.randn(1, 1, num_kv_heads, head_dim).bfloat16().float()
        Q = torch.randn(1, 1, num_q_heads, head_dim).bfloat16().float()
        k_hist[pos, :, :] = K_new[0, 0]
        v_hist[pos, :, :] = V_new[0, 0]

        K_padded = torch.nn.functional.pad(K_new, (0, 0, 0, PADDED_HEADS - num_kv_heads))
        V_padded = torch.nn.functional.pad(V_new, (0, 0, 0, PADDED_HEADS - num_kv_heads))
        Kt = _sharded_kv_input(device, K_padded)
        Vt = _sharded_kv_input(device, V_padded)
        pos_tt = ttnn.Tensor(torch.tensor([pos], dtype=torch.int32), ttnn.int32).to(device)

        ttnn.experimental.paged_update_cache(
            k_cache_tt, Kt, update_idxs_tensor=pos_tt, page_table=page_table_tt, **modulo_kwargs
        )
        ttnn.experimental.paged_update_cache(
            v_cache_tt, Vt, update_idxs_tensor=pos_tt, page_table=page_table_tt, **modulo_kwargs
        )

        Q_padded = torch.nn.functional.pad(Q, (0, 0, 0, PADDED_HEADS - num_q_heads))
        Qt = ttnn.Tensor(Q_padded, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

        out_tt = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            Qt,
            k_cache_tt,
            v_cache_tt,
            cur_pos_tensor=pos_tt,
            page_table_tensor=page_table_tt,
            scale=scale,
            sliding_window_size=sliding_window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **modulo_kwargs,
        )
        out_torch_padded = ttnn.to_torch(out_tt)
        out_tt_first = out_torch_padded[..., :num_q_heads, :]

        ref = _torch_sliding_ref(k_hist, v_hist, Q, pos, sliding_window, scale)
        passing, msg = comp_pcc(ref, out_tt_first, pcc=0.99)
        results.append((pos, passing, msg))
    return results


@pytest.mark.timeout(180)
def test_bounded_sliding_decode_with_hybrid_page_table(device):
    """End-to-end: hybrid page_table built by build_hybrid_page_tables, fed to the
    paged ops with cache_position_modulo set, correctly tracks the sliding window
    across multiple wrap cycles in a bounded physical cache."""
    sliding_window = 128
    decode_steps = sliding_window * 2 + 16  # well past the boundary
    results = _run_bounded_decode_walk(device, sliding_window, use_modulo=True, decode_steps=decode_steps)

    failing = [(p, m) for (p, ok, m) in results if not ok]
    if failing:
        msg = "\n".join(f"  pos={p}: {m}" for p, m in failing[:10])
        pytest.fail(f"{len(failing)}/{decode_steps} steps failed PCC≥0.99 with hybrid page_table:\n{msg}")


def test_bounded_pool_rejects_without_modulo(device):
    """The relaxed paged_update_cache validation only allows page_table.shape[1] >
    cache.shape[0] when cache_position_modulo is set. Confirm the strict legacy
    check still fires (and surfaces a clear error) when a caller forgets to pass
    the kwarg — keeps the bounded layout from silently clobbering block 0."""
    torch.manual_seed(7)
    block_size = 32
    sliding_window = 128
    sliding_blocks = sliding_window // block_size
    max_seq_len = 4 * sliding_window
    max_blocks = max_seq_len // block_size
    num_users = 1
    num_kv_heads = 1
    head_dim = 128
    PADDED_HEADS = 32

    cache_blocks = sliding_blocks * num_users  # bounded
    cache_torch = torch.zeros(cache_blocks, num_kv_heads, block_size, head_dim).bfloat16().float()
    cache_tt = ttnn.Tensor(cache_torch, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    pts = build_hybrid_page_tables(
        num_layers=1,
        sliding_layers_mask=[True],
        num_users=num_users,
        block_size=block_size,
        max_seq_len=max_seq_len,
        sliding_window=sliding_window,
    )
    page_table_tt = ttnn.Tensor(pts[0], ttnn.int32).to(device)
    cache_idxs_tt = ttnn.Tensor(torch.zeros(num_users, dtype=torch.int32), ttnn.int32).to(device)

    x = torch.randn([1, num_users, num_kv_heads, head_dim]).bfloat16().float()
    x_padded = torch.nn.functional.pad(x, (0, 0, 0, PADDED_HEADS - num_kv_heads))
    xt = _sharded_kv_input(device, x_padded)

    # page_table.shape[1] (max_blocks=16) > cache.shape[0] (sliding_blocks=4) → strict check fires
    assert max_blocks > cache_blocks
    with pytest.raises(RuntimeError, match="max_num_blocks_per_seq must be less than max_num_blocks"):
        ttnn.experimental.paged_update_cache(
            cache_tt,
            xt,
            update_idxs_tensor=cache_idxs_tt,
            page_table=page_table_tt,
        )
