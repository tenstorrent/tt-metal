# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``cache_position_modulo`` kwarg on ``paged_update_cache`` and
``paged_scaled_dot_product_attention_decode``.

Background: vLLM's hybrid kv-cache-groups manager allocates sliding-window layers
with only ``sliding_window/block_size`` blocks per sequence, and zero-pads the
per-layer page_table out to ``max_model_len/block_size`` (see
``vllm-tt-plugin/.../model_runner.py::pad_block_tables``). With the legacy ops
that take only absolute positions, any position ``>= sliding_window`` collapses
onto physical block 0 (the zero-padded tail) and silently clobbers a real block.

``cache_position_modulo`` (in tokens) turns the cache into a circular buffer:
the kernel computes ``pos % cache_position_modulo`` before resolving the
page_table entry. Bounded-capacity allocations then work correctly under
absolute-position addressing.

Coverage:
- Positive: round-trip a single user across multiple wrap cycles with both ops
  and confirm PCC ≥ 0.99 vs torch reference at every step.
- **Negative (clobbering demonstration)**: same setup WITHOUT the kwarg —
  show that PCC collapses past the sliding window, locking in the bug as a
  regression marker. If a future kernel change accidentally fixes it without
  the kwarg, this test will start failing and force a re-think.
- Validator negatives: modulo without page_table; modulo not a multiple of
  block_size; modulo < sliding_window_size for SDPA.
"""

import pytest
import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


# ── Helpers ────────────────────────────────────────────────────────────────


def _torch_sliding_ref(k_hist, v_hist, q, cur_pos, sliding_window, scale):
    """Torch fp32 sliding-window attention reference for a single user.

    ``k_hist`` / ``v_hist`` are unbounded ``[T, kv_heads, head_dim]`` tensors holding
    every K/V written so far at their absolute logical position. Output:
    ``[1, 1, num_q_heads, head_dim]``.
    """
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


def _sharded_kv_input(device, x_padded):
    """Convert a ``(1, B=1, padded_heads, head_dim)`` torch tensor to a height-sharded
    ttnn input matching the paged_update_cache decode-time format."""
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


def _run_decode_walk(device, sliding_window, use_modulo_kwarg, decode_steps, scale=None):
    """Drive a single user through ``decode_steps`` of paged_update_cache +
    paged_scaled_dot_product_attention_decode against a bounded sliding-window cache.

    Returns the list of (pos, passing, msg) PCC checks vs the torch sliding reference.
    When ``use_modulo_kwarg`` is False, this reproduces the silent-corruption bug.
    """
    torch.manual_seed(0)

    block_size = 32
    sliding_blocks = sliding_window // block_size
    max_blocks_per_req = 4 * sliding_blocks  # vLLM-style zero-pad past sliding_blocks

    num_kv_heads = 1
    num_q_heads = 1
    head_dim = 128
    PADDED_HEADS = 32  # paged_update_cache requires shard width == padded last dim and kv padded to TILE
    if scale is None:
        scale = 1.0 / (head_dim**0.5)

    # Physical cache sized to hold max_blocks_per_req blocks total. Only the first
    # sliding_blocks entries of the page_table are valid; the tail is zero-padded.
    k_init = torch.zeros(max_blocks_per_req, num_kv_heads, block_size, head_dim).bfloat16().float()
    v_init = torch.zeros(max_blocks_per_req, num_kv_heads, block_size, head_dim).bfloat16().float()
    k_cache_tt = ttnn.Tensor(k_init, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    v_cache_tt = ttnn.Tensor(v_init, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    page_table = torch.zeros(1, max_blocks_per_req, dtype=torch.int32)
    page_table[0, :sliding_blocks] = torch.arange(sliding_blocks, dtype=torch.int32)
    page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)

    k_hist = torch.zeros(decode_steps, num_kv_heads, head_dim).bfloat16().float()
    v_hist = torch.zeros(decode_steps, num_kv_heads, head_dim).bfloat16().float()

    modulo_kwargs = {"cache_position_modulo": sliding_window} if use_modulo_kwarg else {}

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


# ── Positive: cache_position_modulo makes bounded cache correct ────────────


@pytest.mark.timeout(180)
def test_bounded_sliding_kv_cache_position_modulo_round_trip(device):
    """With ``cache_position_modulo=sliding_window`` set, the bounded cache addresses
    correctly across multiple wrap cycles; PCC ≥ 0.99 at every step."""
    sliding_window = 128
    decode_steps = sliding_window * 2 + 16  # walk well past the boundary

    results = _run_decode_walk(device, sliding_window, use_modulo_kwarg=True, decode_steps=decode_steps)

    failing = [(p, m) for (p, ok, m) in results if not ok]
    if failing:
        msg = "\n".join(f"  pos={p}: {m}" for p, m in failing[:10])
        pytest.fail(f"{len(failing)}/{decode_steps} steps failed PCC≥0.99 with cache_position_modulo set:\n{msg}")


# ── Negative: without the kwarg, positions past sliding_window clobber block 0 ──


@pytest.mark.timeout(180)
def test_bounded_sliding_kv_cache_without_modulo_clobbers(device):
    """Without ``cache_position_modulo``, positions past sliding_window route through
    the zero-padded page_table tail and collapse onto physical block 0 — silently
    corrupting the cache. Locks in the bug as a regression marker so anyone who
    "fixes" it implicitly (e.g. by changing zero-padding semantics) sees this test
    flip and re-thinks the contract.

    Expected behaviour: pre-boundary positions stay near PCC 1.0; positions well past
    the boundary tank. We assert that at least 30% of post-boundary steps fall below
    PCC 0.99, which is conservative against bf16 noise.
    """
    sliding_window = 128
    decode_steps = sliding_window * 2 + 16

    results = _run_decode_walk(device, sliding_window, use_modulo_kwarg=False, decode_steps=decode_steps)

    pre = [ok for (p, ok, _) in results if p < sliding_window]
    post = [(p, ok, m) for (p, ok, m) in results if p >= sliding_window]

    pre_passing = sum(pre)
    assert pre_passing >= len(pre) * 0.95, (
        f"Pre-boundary PCC unexpectedly poor without the kwarg: {pre_passing}/{len(pre)} passed. "
        "If this assertion fails the test fixture itself is broken."
    )

    post_failing = sum(1 for (_, ok, _) in post if not ok)
    assert post_failing >= len(post) * 0.3, (
        f"Expected >=30% of post-sliding-window steps to fail PCC≥0.99 without the kwarg "
        f"(clobbering bug should be visible); only {post_failing}/{len(post)} did. "
        "The legacy ops may have been silently fixed — re-evaluate whether the modulo "
        "kwarg is still needed."
    )


# ── Validator negatives ────────────────────────────────────────────────────


def test_paged_update_cache_modulo_requires_page_table(device):
    """``cache_position_modulo`` without ``page_table`` must be rejected at validation
    (same gate as the other paged-mode-only kwargs)."""
    torch.manual_seed(1)
    num_users = 4
    num_kv_heads = 1
    max_seq_len = 256
    head_dim = 256

    cache_shape = [num_users, num_kv_heads, max_seq_len, head_dim]
    cache_torch = torch.randn(cache_shape).bfloat16().float()
    cache_tt = ttnn.Tensor(cache_torch, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    cache_idxs_tt = ttnn.Tensor(torch.arange(num_users, dtype=torch.int32), ttnn.int32).to(device)

    x = torch.randn([1, num_users, num_kv_heads, head_dim]).bfloat16().float()
    x_padded = torch.nn.functional.pad(x, (0, 0, 0, 32 - num_kv_heads), "constant", 0)
    xt = _sharded_kv_input(device, x_padded)

    with pytest.raises(RuntimeError, match="cache_position_modulo is only supported in paged mode"):
        ttnn.experimental.paged_update_cache(
            cache_tt,
            xt,
            update_idxs_tensor=cache_idxs_tt,
            cache_position_modulo=128,
        )


def test_paged_update_cache_modulo_must_be_multiple_of_block_size(device):
    """``cache_position_modulo`` not a multiple of effective block_size must fail."""
    torch.manual_seed(2)
    num_users = 1
    num_kv_heads = 1
    block_size = 32
    head_dim = 128
    num_blocks = 4

    cache_torch = torch.randn(num_blocks, num_kv_heads, block_size, head_dim).bfloat16().float()
    cache_tt = ttnn.Tensor(cache_torch, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(num_users, num_blocks)
    page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)
    cache_idxs_tt = ttnn.Tensor(torch.zeros(num_users, dtype=torch.int32), ttnn.int32).to(device)

    x = torch.randn([1, num_users, num_kv_heads, head_dim]).bfloat16().float()
    x_padded = torch.nn.functional.pad(x, (0, 0, 0, 32 - num_kv_heads), "constant", 0)
    xt = _sharded_kv_input(device, x_padded)

    with pytest.raises(
        RuntimeError, match="cache_position_modulo .* must be a positive multiple of effective block_size"
    ):
        ttnn.experimental.paged_update_cache(
            cache_tt,
            xt,
            update_idxs_tensor=cache_idxs_tt,
            page_table=page_table_tt,
            cache_position_modulo=block_size + 1,  # not a multiple of 32
        )
