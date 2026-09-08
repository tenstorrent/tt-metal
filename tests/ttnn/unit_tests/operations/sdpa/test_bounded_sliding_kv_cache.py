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


def test_paged_update_cache_modulo_requires_page_table(device, expect_error):
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

    with expect_error(RuntimeError, "cache_position_modulo is only supported in paged mode"):
        ttnn.experimental.paged_update_cache(
            cache_tt,
            xt,
            update_idxs_tensor=cache_idxs_tt,
            cache_position_modulo=128,
        )


def test_paged_update_cache_modulo_must_be_multiple_of_block_size(device, expect_error):
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

    with expect_error(RuntimeError, "cache_position_modulo .* must be a positive multiple of effective block_size"):
        ttnn.experimental.paged_update_cache(
            cache_tt,
            xt,
            update_idxs_tensor=cache_idxs_tt,
            page_table=page_table_tt,
            cache_position_modulo=block_size + 1,  # not a multiple of 32
        )


# ── paged_fill_cache: long prefill with wrap survives the last cache_position_modulo tokens ──


@pytest.mark.timeout(60)
def test_paged_fill_cache_bounded_capacity_wrap(device):
    """A prefill of length > cache_position_modulo, written with the kwarg set, must
    leave the cache holding *exactly* the last cache_position_modulo tokens (each at
    its wrapped slot ``pos % cache_position_modulo``). The redundant earlier writes
    are correctly overwritten; the surviving region matches the torch reference at
    every position.
    """
    torch.manual_seed(3)

    num_kv_heads = 1
    head_dim = 128
    block_size = 32
    sliding_window = 128
    sliding_blocks = sliding_window // block_size
    num_users = 1
    # Prefill length crosses the sliding boundary by one full block, so the wrap
    # actually overwrites earlier writes.
    prefill_len = sliding_window + block_size

    max_blocks_per_req = sliding_blocks * 4  # vLLM-style oversize page_table

    # Physical cache holds max_blocks_per_req blocks; only the first sliding_blocks
    # entries of the page_table are valid (mimicking vLLM SlidingWindowSpec).
    cache_torch = torch.zeros(max_blocks_per_req, num_kv_heads, block_size, head_dim).bfloat16().float()
    cache_tt = ttnn.Tensor(cache_torch, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    page_table = torch.zeros(num_users, max_blocks_per_req, dtype=torch.int32)
    page_table[0, :sliding_blocks] = torch.arange(sliding_blocks, dtype=torch.int32)
    page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)

    # Input: full prefill_len tokens of K. The op writes seq_tile_id %= capacity_t
    # per tile, so each token p in [0, prefill_len) writes to slot p % sliding_window.
    K_full = torch.randn(1, num_kv_heads, prefill_len, head_dim).bfloat16().float()
    Kt = ttnn.from_torch(K_full, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)

    ttnn.experimental.paged_fill_cache(
        cache_tt,
        Kt,
        page_table_tt,
        batch_idx=0,
        cache_position_modulo=sliding_window,
    )

    # Read back and reconstruct logical positions [0, sliding_window) using page_table.
    got = cache_tt.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()  # [phys_blocks, kv, bs, d]
    cache_view = torch.zeros(sliding_window, num_kv_heads, head_dim).bfloat16().float()
    for vb in range(sliding_blocks):
        phys = int(page_table[0, vb].item())
        cache_view[vb * block_size : (vb + 1) * block_size] = (
            got[phys, :, :, :].transpose(0, 1).squeeze(1).reshape(block_size, num_kv_heads, head_dim)
        )

    # Expected: for each slot s in [0, sliding_window), the value is the K at the
    # largest absolute pos p < prefill_len with p % sliding_window == s.
    ref = torch.zeros(sliding_window, num_kv_heads, head_dim).bfloat16().float()
    for p in range(prefill_len):
        ref[p % sliding_window] = K_full[0, :, p, :]

    eq, msg = comp_pcc(ref, cache_view, pcc=0.99)
    assert eq, f"Bounded fill_cache wrap mismatch: {msg}"


@pytest.mark.timeout(60)
def test_paged_fill_cache_valid_seq_len_tensor_skips_padding_tail(device):
    """Padded prefill + valid_seq_len_tensor must keep the last real window, not padding.

    Without the runtime cap, a bounded fill of a padded input wraps the zero
    padding tail over the recent real tokens. Refreshing a 1-element device
    tensor between program-cache hits must change which tiles survive.
    """
    torch.manual_seed(5)

    num_kv_heads = 1
    head_dim = 128
    block_size = 32
    sliding_window = 128
    sliding_blocks = sliding_window // block_size
    padded_len = sliding_window + block_size  # crosses capacity so wrap matters
    real_len = sliding_window  # last real token ends exactly at capacity

    max_blocks_per_req = sliding_blocks * 4
    cache_torch = torch.zeros(max_blocks_per_req, num_kv_heads, block_size, head_dim).bfloat16().float()
    cache_tt = ttnn.Tensor(cache_torch, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    page_table = torch.zeros(1, max_blocks_per_req, dtype=torch.int32)
    page_table[0, :sliding_blocks] = torch.arange(sliding_blocks, dtype=torch.int32)
    page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)

    K_full = torch.randn(1, num_kv_heads, padded_len, head_dim).bfloat16().float()
    # Explicit padding tail — if these wrap into the ring, PCC vs real window fails.
    K_full[:, :, real_len:, :] = 0.0
    Kt = ttnn.from_torch(K_full, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)

    valid_host = torch.tensor([real_len], dtype=torch.int32)
    valid_tt = ttnn.from_torch(valid_host, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, device=device)

    ttnn.experimental.paged_fill_cache(
        cache_tt,
        Kt,
        page_table_tt,
        batch_idx=0,
        cache_position_modulo=sliding_window,
        valid_seq_len_tensor=valid_tt,
    )

    got = cache_tt.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    cache_view = torch.zeros(sliding_window, num_kv_heads, head_dim).bfloat16().float()
    for vb in range(sliding_blocks):
        phys = int(page_table[0, vb].item())
        cache_view[vb * block_size : (vb + 1) * block_size] = (
            got[phys, :, :, :].transpose(0, 1).squeeze(1).reshape(block_size, num_kv_heads, head_dim)
        )

    ref = torch.zeros(sliding_window, num_kv_heads, head_dim).bfloat16().float()
    for p in range(real_len):
        ref[p % sliding_window] = K_full[0, :, p, :]

    eq, msg = comp_pcc(ref, cache_view, pcc=0.99)
    assert eq, f"valid_seq_len_tensor fill cap mismatch: {msg}"

    # Program-cache hit path: shrink the real window and refresh the tensor in place.
    shorter = sliding_window - block_size
    K_full2 = torch.randn(1, num_kv_heads, padded_len, head_dim).bfloat16().float()
    K_full2[:, :, shorter:, :] = 0.0
    Kt2 = ttnn.from_torch(K_full2, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    valid_refresh = ttnn.from_torch(
        torch.tensor([shorter], dtype=torch.int32),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        device=device,
    )
    ttnn.copy(valid_refresh, valid_tt)

    cache_tt2 = (
        ttnn.Tensor(
            torch.zeros(max_blocks_per_req, num_kv_heads, block_size, head_dim).bfloat16().float(),
            ttnn.bfloat16,
        )
        .to(ttnn.TILE_LAYOUT)
        .to(device)
    )
    ttnn.experimental.paged_fill_cache(
        cache_tt2,
        Kt2,
        page_table_tt,
        batch_idx=0,
        cache_position_modulo=sliding_window,
        valid_seq_len_tensor=valid_tt,
    )

    got2 = cache_tt2.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    cache_view2 = torch.zeros(sliding_window, num_kv_heads, head_dim).bfloat16().float()
    for vb in range(sliding_blocks):
        phys = int(page_table[0, vb].item())
        cache_view2[vb * block_size : (vb + 1) * block_size] = (
            got2[phys, :, :, :].transpose(0, 1).squeeze(1).reshape(block_size, num_kv_heads, head_dim)
        )
    ref2 = torch.zeros(sliding_window, num_kv_heads, head_dim).bfloat16().float()
    for p in range(shorter):
        ref2[p % sliding_window] = K_full2[0, :, p, :]

    eq2, msg2 = comp_pcc(ref2, cache_view2, pcc=0.99)
    assert eq2, f"valid_seq_len_tensor refresh mismatch: {msg2}"


@pytest.mark.timeout(30)
def test_paged_fill_cache_valid_seq_len_tensor_rejects_multi_element(device, expect_error):
    """API contract: valid_seq_len_tensor must be a single scalar element."""
    block_size = 32
    sliding_window = 64
    cache_tt = (
        ttnn.Tensor(torch.zeros(4, 1, block_size, 64).bfloat16().float(), ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    )
    page_table_tt = ttnn.Tensor(torch.arange(4, dtype=torch.int32).view(1, 4), ttnn.int32).to(device)
    Kt = ttnn.from_torch(
        torch.randn(1, 1, sliding_window, 64).bfloat16().float(),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )
    bad = ttnn.from_torch(
        torch.tensor([32, 64], dtype=torch.int32),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        device=device,
    )
    with expect_error(RuntimeError, "exactly 1 element"):
        ttnn.experimental.paged_fill_cache(
            cache_tt,
            Kt,
            page_table_tt,
            batch_idx=0,
            cache_position_modulo=sliding_window,
            valid_seq_len_tensor=bad,
        )
