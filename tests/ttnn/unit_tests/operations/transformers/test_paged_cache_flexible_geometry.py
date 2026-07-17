# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``block_size`` override on ``paged_update_cache`` and
``paged_fill_cache``.

These ops accept an optional ``block_size`` kwarg so callers (e.g. vLLM's hybrid
kv-cache-groups manager) can write into one physical cache with different
``(block_size, head_dim)`` views per layer, as long as
``num_kv_heads * block_size * head_dim`` is preserved across views.

Coverage: legacy (no override), override in both directions, shared buffer with
interleaved views, and negative cases (byte-count mismatch, override without page_table).
"""

import pytest
import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal


# ── Test harness helpers ───────────────────────────────────────────────────


def _sharded_input(device, x_padded):
    """Convert ``(1, B, padded_heads, head_dim)`` to a height-sharded ttnn tensor,
    matching the layout in ``test_paged_update_cache.py::run_test_paged_update_cache_decode``."""
    num_users = x_padded.shape[1]
    xt = ttnn.Tensor(x_padded, ttnn.bfloat16).to(ttnn.TILE_LAYOUT)
    compute_grid_size = device.compute_with_storage_grid_size()
    shard_grid = ttnn.num_cores_to_corerangeset(num_users, compute_grid_size, True)
    input_shard_spec = ttnn.ShardSpec(
        shard_grid,
        [
            xt.volume() // xt.padded_shape[-1] // num_users,
            xt.padded_shape[-1],
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)
    return xt.to(device, input_mem_config)


def _make_paged_cache(num_blocks, num_kv_heads, block_size, head_dim, device):
    """Allocate a ttnn paged cache tensor filled with deterministic random
    data, plus its torch reference."""
    cache_shape = [num_blocks, num_kv_heads, block_size, head_dim]
    cache_torch = torch.randn(cache_shape).bfloat16().float()
    cache_tt = ttnn.Tensor(cache_torch, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    return cache_tt, cache_torch


def _read_paged_cache(cache_tt):
    """Read a ttnn paged cache back to torch in its declared shape."""
    return cache_tt.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()


def _permute_tile_grid(t, view_block_size, view_head_dim):
    """Reinterpret ``t``'s last two dims under a different tile-grid.

    The kernel writes by linear tile position within a block; ttnn's untilize is bound
    to the cache's *declared* shape. So when alloc-shape differs from call-view, a plain
    ``torch.view`` reshape of the untilized tensor would mis-place data — the tile
    boundaries are in different places. This permutation re-groups tiles by their linear
    index into the view's tile-grid, keeping each 32×32 tile intact.
    """
    N, KV, alloc_block_size, alloc_head_dim = t.shape
    TILE = 32
    assert alloc_block_size % TILE == 0 and alloc_head_dim % TILE == 0, "alloc dims must be tile-aligned"
    assert view_block_size % TILE == 0 and view_head_dim % TILE == 0, "view dims must be tile-aligned"
    alloc_BR_t = alloc_block_size // TILE
    alloc_Wt = alloc_head_dim // TILE
    view_BR_t = view_block_size // TILE
    view_Wt = view_head_dim // TILE
    total_tiles = alloc_BR_t * alloc_Wt
    assert total_tiles == view_BR_t * view_Wt, "per-block tile count must match"

    # Split into (tile-grid, intra-tile), flatten tile-grid, re-split under view, repack.
    t = t.view(N, KV, alloc_BR_t, TILE, alloc_Wt, TILE)
    t = t.permute(0, 1, 2, 4, 3, 5).contiguous()
    t = t.view(N, KV, total_tiles, TILE, TILE)
    t = t.view(N, KV, view_BR_t, view_Wt, TILE, TILE)
    t = t.permute(0, 1, 2, 4, 3, 5).contiguous()
    return t.view(N, KV, view_block_size, view_head_dim)


def _permute_view_general(t_alloc, view_kv, view_block_size, view_head_dim):
    """Like ``_permute_tile_grid`` but also reinterprets the kv-heads dimension.

    Per-block tiles are stored linearly in ``(kv_head, bs_tile, hd_tile)`` order; with
    that linearization the kv-head axis is just the outermost factor of the per-block
    tile count, so changing it is the same kind of regroup as changing block_size or
    head_dim. ``_permute_tile_grid`` is the special case where alloc_kv == view_kv.

    Used to verify HMA cross-group sharing where sliding and full layers see one
    physical buffer through views with different ``num_kv_heads`` (e.g. Gemma4-26B-A4B
    sliding kv=8 / full kv=2). The per-block element-count invariant
    ``alloc_kv * alloc_bs * alloc_hd == view_kv * view_bs * view_hd`` must hold.
    """
    N, alloc_kv, alloc_block_size, alloc_head_dim = t_alloc.shape
    TILE = 32
    assert alloc_block_size % TILE == 0 and alloc_head_dim % TILE == 0, "alloc dims must be tile-aligned"
    assert view_block_size % TILE == 0 and view_head_dim % TILE == 0, "view dims must be tile-aligned"
    alloc_BR_t = alloc_block_size // TILE
    alloc_Wt = alloc_head_dim // TILE
    view_BR_t = view_block_size // TILE
    view_Wt = view_head_dim // TILE
    alloc_total_tiles = alloc_kv * alloc_BR_t * alloc_Wt
    view_total_tiles = view_kv * view_BR_t * view_Wt
    assert alloc_total_tiles == view_total_tiles, (
        f"per-block tile count mismatch: alloc {alloc_kv}*{alloc_BR_t}*{alloc_Wt}={alloc_total_tiles} "
        f"vs view {view_kv}*{view_BR_t}*{view_Wt}={view_total_tiles}"
    )

    # Alloc-view (N, KV, BR_t, TILE, Wt, TILE) → tile-grid major (N, KV, BR_t, Wt, TILE, TILE),
    # then flatten (KV, BR_t, Wt) into a single per-block tile axis.
    t = t_alloc.view(N, alloc_kv, alloc_BR_t, TILE, alloc_Wt, TILE)
    t = t.permute(0, 1, 2, 4, 3, 5).contiguous()
    t = t.reshape(N, alloc_total_tiles, TILE, TILE)
    # Repack under view layout: (N, view_KV, view_BR_t, view_Wt, TILE, TILE) tile-grid major,
    # then back to (N, view_KV, view_block_size, view_head_dim).
    t = t.reshape(N, view_kv, view_BR_t, view_Wt, TILE, TILE)
    t = t.permute(0, 1, 2, 4, 3, 5).contiguous()
    return t.reshape(N, view_kv, view_block_size, view_head_dim)


# ── paged_update_cache tests ───────────────────────────────────────────────


@pytest.mark.parametrize("block_size, head_dim", [(64, 256), (128, 256), (64, 512)])
def test_paged_update_cache_legacy_no_override(block_size, head_dim, device):
    """No ``block_size`` kwarg → behavior identical to the pre-change op. Backward-compat guard."""
    torch.manual_seed(0)
    num_users = 4
    num_kv_heads = 1
    max_seq_len = 256
    max_num_blocks_per_seq = max_seq_len // block_size
    max_num_blocks = num_users * max_num_blocks_per_seq

    cache_tt, cache_ref = _make_paged_cache(max_num_blocks, num_kv_heads, block_size, head_dim, device)

    page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(num_users, max_num_blocks_per_seq)
    page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)

    # One decode token per user.
    cache_idx = block_size + 5  # straddles a block boundary
    cache_idxs = [cache_idx + i * 3 for i in range(num_users)]
    cache_idxs_tt = ttnn.Tensor(torch.tensor(cache_idxs), ttnn.int32).to(device)

    # Input shape: [1, B, num_kv_heads_padded_to_32, head_dim]
    x = torch.randn([1, num_users, num_kv_heads, head_dim]).bfloat16().float()
    x_padded = torch.nn.functional.pad(x, (0, 0, 0, 32 - num_kv_heads), "constant", 0)
    xt = _sharded_input(device, x_padded)

    # Reference: walk page_table to find each user's physical block, then write at
    # ``cache_idxs[u] % block_size`` within it.
    for u in range(num_users):
        pos = cache_idxs[u]
        physical_block = page_table[u, pos // block_size].item()
        cache_ref[physical_block, 0:num_kv_heads, pos % block_size : pos % block_size + 1, :] = x[:, u, :, :]

    ttnn.experimental.paged_update_cache(
        cache_tt,
        xt,
        update_idxs_tensor=cache_idxs_tt,
        page_table=page_table_tt,
    )

    got = _read_paged_cache(cache_tt)
    eq, msg = comp_equal(cache_ref, got)
    assert eq, f"legacy path diverged from reference: {msg}"


@pytest.mark.parametrize("num_kv_heads", [1, 8])
def test_paged_update_cache_override_view_full_into_sliding_buffer(num_kv_heads, device):
    """Buffer allocated as sliding (block=128, head=256); writes use the full-attention
    view (block=64, head=512) via the override. Canonical Gemma4-E2B scenario after
    vLLM's page-size unifier matches sliding's block_size to full's.

    Parametrized on ``num_kv_heads`` to exercise the multi-head per-block stride.
    """
    torch.manual_seed(1)

    num_users = 4
    # Alloc view: sliding. Call view: full.
    alloc_block_size = 128
    alloc_head_dim = 256
    view_block_size = 64
    view_head_dim = 512
    assert num_kv_heads * alloc_block_size * alloc_head_dim == num_kv_heads * view_block_size * view_head_dim

    max_seq_len_view = 512
    max_num_blocks_per_seq = max_seq_len_view // view_block_size
    max_num_blocks = num_users * max_num_blocks_per_seq

    assert max_seq_len_view % alloc_block_size == 0
    cache_tt, cache_ref_alloc = _make_paged_cache(
        max_num_blocks, num_kv_heads, alloc_block_size, alloc_head_dim, device
    )

    page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(num_users, max_num_blocks_per_seq)
    page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)

    # cache_idxs are in view coordinates.
    cache_idxs = [10 + u * 17 for u in range(num_users)]
    cache_idxs_tt = ttnn.Tensor(torch.tensor(cache_idxs), ttnn.int32).to(device)

    x = torch.randn([1, num_users, num_kv_heads, view_head_dim]).bfloat16().float()
    x_padded = torch.nn.functional.pad(x, (0, 0, 0, 32 - num_kv_heads), "constant", 0)
    xt = _sharded_input(device, x_padded)

    ttnn.experimental.paged_update_cache(
        cache_tt,
        xt,
        update_idxs_tensor=cache_idxs_tt,
        page_table=page_table_tt,
        block_size=view_block_size,
    )

    got_alloc = _read_paged_cache(cache_tt)
    assert got_alloc.shape == (max_num_blocks, num_kv_heads, alloc_block_size, alloc_head_dim)
    got_view = _permute_tile_grid(got_alloc, view_block_size, view_head_dim)

    cache_ref_view = _permute_tile_grid(cache_ref_alloc, view_block_size, view_head_dim).clone()
    for u in range(num_users):
        pos = cache_idxs[u]
        physical_block = page_table[u, pos // view_block_size].item()
        # unsqueeze(1) broadcasts [KV, head_dim] into the [KV, 1, head_dim] slot.
        cache_ref_view[physical_block, 0:num_kv_heads, pos % view_block_size : pos % view_block_size + 1, :] = x[
            0, u
        ].unsqueeze(1)

    eq, msg = comp_equal(cache_ref_view, got_view)
    assert eq, f"override view mismatch: {msg}"


def test_paged_update_cache_override_view_sliding_into_full_buffer(device):
    """Symmetric: buffer allocated for full (block=64, head=512); call
    writes with the sliding view (block=128, head=256)."""
    torch.manual_seed(2)

    num_users = 4
    num_kv_heads = 1
    alloc_block_size = 64
    alloc_head_dim = 512
    view_block_size = 128
    view_head_dim = 256
    assert num_kv_heads * alloc_block_size * alloc_head_dim == num_kv_heads * view_block_size * view_head_dim

    max_seq_len_view = 512
    max_num_blocks_per_seq = max_seq_len_view // view_block_size
    max_num_blocks = num_users * max_num_blocks_per_seq

    cache_tt, cache_ref_alloc = _make_paged_cache(
        max_num_blocks, num_kv_heads, alloc_block_size, alloc_head_dim, device
    )

    page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(num_users, max_num_blocks_per_seq)
    page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)

    cache_idxs = [40 + u * 33 for u in range(num_users)]
    cache_idxs_tt = ttnn.Tensor(torch.tensor(cache_idxs), ttnn.int32).to(device)

    x = torch.randn([1, num_users, num_kv_heads, view_head_dim]).bfloat16().float()
    x_padded = torch.nn.functional.pad(x, (0, 0, 0, 32 - num_kv_heads), "constant", 0)
    xt = _sharded_input(device, x_padded)

    ttnn.experimental.paged_update_cache(
        cache_tt,
        xt,
        update_idxs_tensor=cache_idxs_tt,
        page_table=page_table_tt,
        block_size=view_block_size,
    )

    got_alloc = _read_paged_cache(cache_tt)
    got_view = _permute_tile_grid(got_alloc, view_block_size, view_head_dim)
    cache_ref_view = _permute_tile_grid(cache_ref_alloc, view_block_size, view_head_dim).clone()
    for u in range(num_users):
        pos = cache_idxs[u]
        physical_block = page_table[u, pos // view_block_size].item()
        cache_ref_view[physical_block, 0:num_kv_heads, pos % view_block_size : pos % view_block_size + 1, :] = x[
            :, u, :, :
        ]

    eq, msg = comp_equal(cache_ref_view, got_view)
    assert eq, f"symmetric override view mismatch: {msg}"


def test_paged_update_cache_shared_buffer_both_views_disjoint_blocks(device):
    """One physical buffer, two consecutive writes from different views.

    Writes block IDs ``0..num_users-1`` with the sliding view, then block
    IDs ``num_users..2*num_users-1`` with the full view, and verifies
    both writes land correctly without disturbing each other. This is the
    interleaved access pattern that vLLM's hybrid kv-cache-groups manager
    produces when sliding and full layers share one ``KVCacheTensor``.
    """
    torch.manual_seed(3)

    num_users = 4
    num_kv_heads = 1
    sliding_block_size = 128
    sliding_head_dim = 256
    full_block_size = 64
    full_head_dim = 512
    assert num_kv_heads * sliding_block_size * sliding_head_dim == num_kv_heads * full_block_size * full_head_dim

    # Each view uses its own block-count slice. The cache is allocated as
    # sliding-shape and has 2 * num_users blocks total — first half for
    # sliding writes, second half for full writes.
    max_num_blocks = 2 * num_users
    sliding_blocks_per_user = 1
    full_blocks_per_user = 1

    # Allocate as sliding view. Per-block bytes are identical between
    # views, so the byte capacity supports either view's block layout.
    cache_tt, cache_ref_alloc = _make_paged_cache(
        max_num_blocks, num_kv_heads, sliding_block_size, sliding_head_dim, device
    )

    # Sliding write: block IDs 0..num_users-1, one block per user.
    sliding_pt = torch.arange(0, num_users, dtype=torch.int32).reshape(num_users, sliding_blocks_per_user)
    sliding_pt_tt = ttnn.Tensor(sliding_pt, ttnn.int32).to(device)
    sliding_idxs = [10 + u for u in range(num_users)]
    sliding_idxs_tt = ttnn.Tensor(torch.tensor(sliding_idxs), ttnn.int32).to(device)
    x_sliding = torch.randn([1, num_users, num_kv_heads, sliding_head_dim]).bfloat16().float()
    xt_sliding = _sharded_input(device, torch.nn.functional.pad(x_sliding, (0, 0, 0, 32 - num_kv_heads), "constant", 0))
    ttnn.experimental.paged_update_cache(
        cache_tt,
        xt_sliding,
        update_idxs_tensor=sliding_idxs_tt,
        page_table=sliding_pt_tt,
        # No override: call view matches alloc view.
    )

    # Full write: block IDs num_users..2*num_users-1.
    full_pt = torch.arange(num_users, 2 * num_users, dtype=torch.int32).reshape(num_users, full_blocks_per_user)
    full_pt_tt = ttnn.Tensor(full_pt, ttnn.int32).to(device)
    full_idxs = [20 + u for u in range(num_users)]
    full_idxs_tt = ttnn.Tensor(torch.tensor(full_idxs), ttnn.int32).to(device)
    x_full = torch.randn([1, num_users, num_kv_heads, full_head_dim]).bfloat16().float()
    xt_full = _sharded_input(device, torch.nn.functional.pad(x_full, (0, 0, 0, 32 - num_kv_heads), "constant", 0))
    ttnn.experimental.paged_update_cache(
        cache_tt,
        xt_full,
        update_idxs_tensor=full_idxs_tt,
        page_table=full_pt_tt,
        block_size=full_block_size,
    )

    # Reference: write sliding in alloc coords, permute to full coords, then write full.
    cache_ref_sliding = cache_ref_alloc.clone()
    for u in range(num_users):
        pos = sliding_idxs[u]
        physical_block = sliding_pt[u, 0].item()
        cache_ref_sliding[
            physical_block, 0:num_kv_heads, pos % sliding_block_size : pos % sliding_block_size + 1, :
        ] = x_sliding[:, u, :, :]

    cache_ref_full = _permute_tile_grid(cache_ref_sliding, full_block_size, full_head_dim).clone()
    for u in range(num_users):
        pos = full_idxs[u]
        physical_block = full_pt[u, 0].item()
        cache_ref_full[physical_block, 0:num_kv_heads, pos % full_block_size : pos % full_block_size + 1, :] = x_full[
            :, u, :, :
        ]

    got_alloc = _read_paged_cache(cache_tt)
    got_view = _permute_tile_grid(got_alloc, full_block_size, full_head_dim)
    eq, msg = comp_equal(cache_ref_full, got_view)
    assert eq, f"shared-buffer interleaved writes diverged: {msg}"


def test_paged_update_cache_asymmetric_num_heads_per_block_match(device):
    """Cache allocated with one ``num_kv_heads``, decode updates with a different
    ``num_kv_heads`` — the canonical Gemma4-26B-A4B / 31B HMA cross-group case on
    the decode path. Buffer holds sliding's spec (kv=8, bs=64, hd=256), the call
    writes via full's view (kv=2, bs=128, hd=512) using both the
    ``block_size`` and ``num_kv_heads`` kwargs. Round-trips through
    ``_permute_view_general`` to verify writes land where the full view will
    later read them. Guards the relaxation in the validator and the program
    factory: per-block element-count equality is the real invariant, not strict
    ``num_heads`` equality.
    """
    torch.manual_seed(6)
    num_users = 2
    # Sliding spec (alloc): kv=8, bs=64, hd=256 → 131072 elems/block.
    cache_kv = 8
    alloc_block_size = 64
    alloc_head_dim = 256
    # Full spec (call view): kv=2, bs=128, hd=512 → also 131072 elems/block.
    view_kv = 2
    view_block_size = 128
    view_head_dim = 512
    assert cache_kv * alloc_block_size * alloc_head_dim == view_kv * view_block_size * view_head_dim

    max_seq_len_view = 256
    max_num_blocks_per_seq = max_seq_len_view // view_block_size
    max_num_blocks = num_users * max_num_blocks_per_seq

    cache_tt, cache_ref_alloc = _make_paged_cache(max_num_blocks, cache_kv, alloc_block_size, alloc_head_dim, device)

    page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(num_users, max_num_blocks_per_seq)
    page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)

    # cache_idxs are in view coordinates (seq-len under full's block_size).
    cache_idxs = [10 + u * 17 for u in range(num_users)]
    cache_idxs_tt = ttnn.Tensor(torch.tensor(cache_idxs), ttnn.int32).to(device)

    # Input under full's view: (1, B, view_kv, view_hd). Pad kv axis to 32 for the
    # shard's tile alignment; the kernel reads view_kv rows per shard (via the
    # num_kv_heads override), the rest are padding.
    x = torch.randn([1, num_users, view_kv, view_head_dim]).bfloat16().float()
    x_padded = torch.nn.functional.pad(x, (0, 0, 0, 32 - view_kv), "constant", 0)
    xt = _sharded_input(device, x_padded)

    ttnn.experimental.paged_update_cache(
        cache_tt,
        xt,
        update_idxs_tensor=cache_idxs_tt,
        page_table=page_table_tt,
        block_size=view_block_size,
        num_kv_heads=view_kv,
    )

    got_alloc = _read_paged_cache(cache_tt)
    assert got_alloc.shape == (max_num_blocks, cache_kv, alloc_block_size, alloc_head_dim)
    got_view = _permute_view_general(got_alloc, view_kv, view_block_size, view_head_dim)

    cache_ref_view = _permute_view_general(cache_ref_alloc, view_kv, view_block_size, view_head_dim).clone()
    for u in range(num_users):
        pos = cache_idxs[u]
        physical_block = page_table[u, pos // view_block_size].item()
        cache_ref_view[physical_block, 0:view_kv, pos % view_block_size : pos % view_block_size + 1, :] = x[
            0, u
        ].unsqueeze(1)

    eq, msg = comp_equal(cache_ref_view, got_view)
    assert eq, f"asymmetric num_heads update_cache round trip mismatch: {msg}"


def test_paged_update_cache_negative_asymmetric_num_heads_byte_count_mismatch(device):
    """Different ``num_kv_heads`` between cache and ``num_kv_heads`` kwarg *without*
    the per-block element count being preserved must still be rejected. Guards
    against the relaxation accidentally allowing arbitrary mismatched-byte
    writes on the decode path.
    """
    torch.manual_seed(7)
    num_users = 2
    # Cache: 8 * 64 * 256 = 131072 elems/block.
    cache_kv = 8
    alloc_block_size = 64
    alloc_head_dim = 256
    # Override view: 4 * 128 * 512 = 262144 elems/block — twice the cache, mismatched.
    view_kv = 4
    view_block_size = 128
    view_head_dim = 512
    assert cache_kv * alloc_block_size * alloc_head_dim != view_kv * view_block_size * view_head_dim

    max_num_blocks = num_users

    cache_tt, _ = _make_paged_cache(max_num_blocks, cache_kv, alloc_block_size, alloc_head_dim, device)

    page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(num_users, 1)
    page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)

    cache_idxs_tt = ttnn.Tensor(torch.zeros(num_users, dtype=torch.int32), ttnn.int32).to(device)

    x = torch.randn([1, num_users, view_kv, view_head_dim]).bfloat16().float()
    x_padded = torch.nn.functional.pad(x, (0, 0, 0, 32 - view_kv), "constant", 0)
    xt = _sharded_input(device, x_padded)

    with pytest.raises(RuntimeError, match="geometry mismatch"):
        ttnn.experimental.paged_update_cache(
            cache_tt,
            xt,
            update_idxs_tensor=cache_idxs_tt,
            page_table=page_table_tt,
            block_size=view_block_size,
            num_kv_heads=view_kv,
        )


def test_paged_update_cache_negative_num_kv_heads_override_without_page_table(device):
    """``num_kv_heads`` override is paged-mode only; validator must reject it
    without a page_table (analogous to the ``block_size`` gate)."""
    torch.manual_seed(8)
    num_users = 4
    num_kv_heads = 1
    max_seq_len = 256
    head_dim = 256

    cache_shape = [num_users, num_kv_heads, max_seq_len, head_dim]
    cache_torch = torch.randn(cache_shape).bfloat16().float()
    cache_tt = ttnn.Tensor(cache_torch, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    cache_idxs = [0 + i for i in range(num_users)]
    cache_idxs_tt = ttnn.Tensor(torch.tensor(cache_idxs), ttnn.int32).to(device)

    x = torch.randn([1, num_users, num_kv_heads, head_dim]).bfloat16().float()
    x_padded = torch.nn.functional.pad(x, (0, 0, 0, 32 - num_kv_heads), "constant", 0)
    xt = _sharded_input(device, x_padded)

    with pytest.raises(RuntimeError, match="num_kv_heads_override is only supported in paged mode"):
        ttnn.experimental.paged_update_cache(
            cache_tt,
            xt,
            update_idxs_tensor=cache_idxs_tt,
            # No page_table.
            num_kv_heads=num_kv_heads,
        )


def test_paged_update_cache_negative_byte_count_mismatch(device):
    """Override that breaks the per-block byte invariant must be rejected at validation."""
    torch.manual_seed(4)
    num_users = 4
    num_kv_heads = 1
    # Alloc: 32768 elem/block. View: 65536 (mismatched).
    alloc_block_size = 64
    alloc_head_dim = 512
    view_block_size = 256
    view_head_dim = 256
    max_num_blocks = 4

    cache_tt, _ = _make_paged_cache(max_num_blocks, num_kv_heads, alloc_block_size, alloc_head_dim, device)

    page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(num_users, 1)
    page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)
    cache_idxs_tt = ttnn.Tensor(torch.zeros(num_users, dtype=torch.int32), ttnn.int32).to(device)

    x = torch.randn([1, num_users, num_kv_heads, view_head_dim]).bfloat16().float()
    x_padded = torch.nn.functional.pad(x, (0, 0, 0, 32 - num_kv_heads), "constant", 0)
    xt = _sharded_input(device, x_padded)

    with pytest.raises(RuntimeError, match="geometry mismatch"):
        ttnn.experimental.paged_update_cache(
            cache_tt,
            xt,
            update_idxs_tensor=cache_idxs_tt,
            page_table=page_table_tt,
            block_size=view_block_size,
        )


def test_paged_update_cache_negative_override_without_page_table(device):
    """``block_size`` is paged-mode only; validator must reject it without a page_table."""
    torch.manual_seed(5)
    num_users = 4
    num_kv_heads = 1
    max_seq_len = 256
    head_dim = 256

    cache_shape = [num_users, num_kv_heads, max_seq_len, head_dim]
    cache_torch = torch.randn(cache_shape).bfloat16().float()
    cache_tt = ttnn.Tensor(cache_torch, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    cache_idxs = [0 + i for i in range(num_users)]
    cache_idxs_tt = ttnn.Tensor(torch.tensor(cache_idxs), ttnn.int32).to(device)

    x = torch.randn([1, num_users, num_kv_heads, head_dim]).bfloat16().float()
    x_padded = torch.nn.functional.pad(x, (0, 0, 0, 32 - num_kv_heads), "constant", 0)
    xt = _sharded_input(device, x_padded)

    with pytest.raises(RuntimeError, match="block_size_override is only supported in paged mode"):
        ttnn.experimental.paged_update_cache(
            cache_tt,
            xt,
            update_idxs_tensor=cache_idxs_tt,
            # No page_table.
            block_size=64,
        )


def _sharded_input_with_num_cores(device, x_padded, num_cores):
    """Like ``_sharded_input`` but builds a height-sharded tensor whose grid has
    ``num_cores`` cores instead of ``num_users``. Used to force the bad case the
    program factory can't handle (issue #44923)."""
    xt = ttnn.Tensor(x_padded, ttnn.bfloat16).to(ttnn.TILE_LAYOUT)
    total_height = xt.volume() // xt.padded_shape[-1]
    assert (
        total_height % num_cores == 0
    ), f"unsharded test setup: total height {total_height} not divisible by num_cores {num_cores}"
    compute_grid_size = device.compute_with_storage_grid_size()
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, True)
    input_shard_spec = ttnn.ShardSpec(
        shard_grid,
        [total_height // num_cores, xt.padded_shape[-1]],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)
    return xt.to(device, input_mem_config)


@pytest.mark.parametrize(
    "num_users, bad_num_cores",
    [
        # Fewer cores than users (two users per core) — the kernel would treat the second user's
        # data as the first user's overflow, corrupting the cache.
        (4, 2),
    ],
)
def test_paged_update_cache_negative_input_shard_grid_num_cores_mismatch(num_users, bad_num_cores, device):
    """Validator must reject input shard grids whose num_cores != num_users; the program
    factory iterates one user per core and silently miscomputes otherwise (issue #44923)."""
    torch.manual_seed(9)
    num_kv_heads = 1
    max_seq_len = 256
    head_dim = 256

    cache_shape = [num_users, num_kv_heads, max_seq_len, head_dim]
    cache_torch = torch.randn(cache_shape).bfloat16().float()
    cache_tt = ttnn.Tensor(cache_torch, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    cache_idxs = [i for i in range(num_users)]
    cache_idxs_tt = ttnn.Tensor(torch.tensor(cache_idxs), ttnn.int32).to(device)

    x = torch.randn([1, num_users, num_kv_heads, head_dim]).bfloat16().float()
    x_padded = torch.nn.functional.pad(x, (0, 0, 0, 32 - num_kv_heads), "constant", 0)
    xt = _sharded_input_with_num_cores(device, x_padded, bad_num_cores)

    with pytest.raises(RuntimeError, match="num_cores"):
        ttnn.experimental.paged_update_cache(
            cache_tt,
            xt,
            update_idxs_tensor=cache_idxs_tt,
        )


# ── paged_fill_cache tests ────────────────────────────────────────────────


def _run_fill_cache_round_trip(
    device,
    num_kv_heads,
    alloc_block_size,
    alloc_head_dim,
    view_block_size,
    view_head_dim,
    num_users,
    input_seq_len,
    block_size_kwarg,
):
    """Shared fill-cache test body. Allocates a paged cache under the alloc view, fills
    user-by-user with input shaped under the call view, and checks the round-trip in
    view coordinates. ``block_size_kwarg=None`` means no override (alloc == call view).
    """
    max_num_blocks_per_seq = input_seq_len // view_block_size
    assert max_num_blocks_per_seq * view_block_size == input_seq_len
    max_num_blocks = num_users * max_num_blocks_per_seq

    cache_shape = [max_num_blocks, num_kv_heads, alloc_block_size, alloc_head_dim]
    cache_torch_alloc = torch.randn(cache_shape).bfloat16().float()
    cache_tt = ttnn.Tensor(cache_torch_alloc, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(num_users, max_num_blocks_per_seq)
    page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)

    cache_ref_view = _permute_tile_grid(cache_torch_alloc, view_block_size, view_head_dim).clone()
    for u in range(num_users):
        x = torch.randn([1, num_kv_heads, input_seq_len, view_head_dim]).bfloat16().float()
        xt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)

        kwargs = dict(batch_idx=u)
        if block_size_kwarg is not None:
            kwargs["block_size"] = block_size_kwarg
        ttnn.experimental.paged_fill_cache(cache_tt, xt, page_table_tt, **kwargs)

        for t in range(input_seq_len):
            vb = t // view_block_size
            slot = t % view_block_size
            physical_block = page_table[u, vb].item()
            cache_ref_view[physical_block, 0:num_kv_heads, slot : slot + 1, :] = x[0, :, t : t + 1, :]

    got_alloc = _read_paged_cache(cache_tt)
    got_view = _permute_tile_grid(got_alloc, view_block_size, view_head_dim)

    eq, msg = comp_equal(cache_ref_view, got_view)
    assert eq, f"fill cache round trip mismatch: {msg}"


def test_paged_fill_cache_legacy_no_override(device):
    """Legacy path: matched views, no override. Backward-compat guard."""
    torch.manual_seed(10)
    _run_fill_cache_round_trip(
        device,
        num_kv_heads=1,
        alloc_block_size=64,
        alloc_head_dim=256,
        view_block_size=64,
        view_head_dim=256,
        num_users=2,
        input_seq_len=128,
        block_size_kwarg=None,
    )


@pytest.mark.parametrize("num_kv_heads", [1, 8])
def test_paged_fill_cache_override_view_full_into_sliding_buffer(num_kv_heads, device):
    """Cache allocated as sliding ``(128, 256)``, filled via override with full
    ``(64, 512)``. Parametrized on ``num_kv_heads`` to exercise the multi-head per-block
    stride and the ``input_num_heads == cache_num_heads`` validator.
    """
    torch.manual_seed(11)
    _run_fill_cache_round_trip(
        device,
        num_kv_heads=num_kv_heads,
        alloc_block_size=128,
        alloc_head_dim=256,
        view_block_size=64,
        view_head_dim=512,
        num_users=2,
        input_seq_len=128,
        block_size_kwarg=64,
    )


def test_paged_fill_cache_override_view_sliding_into_full_buffer(device):
    """Symmetric: cache allocated as full ``(64, 512)``, filled via
    override with sliding ``(128, 256)``."""
    torch.manual_seed(12)
    _run_fill_cache_round_trip(
        device,
        num_kv_heads=1,
        alloc_block_size=64,
        alloc_head_dim=512,
        view_block_size=128,
        view_head_dim=256,
        num_users=2,
        input_seq_len=128,
        block_size_kwarg=128,
    )


def test_paged_fill_cache_negative_byte_count_mismatch(device):
    """Same as ``paged_update_cache`` byte-count mismatch but for fill."""
    torch.manual_seed(13)
    num_users = 2
    num_kv_heads = 1
    # Alloc: 32768 elem/block. View: 65536 (mismatched).
    alloc_block_size = 64
    alloc_head_dim = 512
    view_block_size = 256
    view_head_dim = 256
    input_seq_len = 128
    max_num_blocks = 4

    cache_torch = torch.randn([max_num_blocks, num_kv_heads, alloc_block_size, alloc_head_dim]).bfloat16().float()
    cache_tt = ttnn.Tensor(cache_torch, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(num_users, 2)
    page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)

    x = torch.randn([1, num_kv_heads, input_seq_len, view_head_dim]).bfloat16().float()
    xt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)

    with pytest.raises(RuntimeError, match="geometry mismatch"):
        ttnn.experimental.paged_fill_cache(cache_tt, xt, page_table_tt, batch_idx=0, block_size=view_block_size)


def test_paged_fill_cache_asymmetric_num_heads_per_block_match(device):
    """Cache allocated with one ``num_kv_heads``, input filled with a different
    ``num_kv_heads`` — the canonical Gemma4-26B-A4B / 31B HMA cross-group case
    where sliding (kv=8, hd=256) and full (kv=2, hd=512) layers share one
    physical buffer via vLLM's hybrid kv-cache-groups manager.

    Allocates the cache under sliding's spec, fills via the full view, then
    round-trips through ``_permute_view_general`` (which also reinterprets the
    kv-head dimension) to verify the kernel laid the writes out where the full
    view would later read them. Guards the relaxation in
    ``paged_fill_cache_device_operation.cpp``: the per-block element-count
    invariant ``input_kv * eff_bs * input_hd == cache_kv * cache_bs * cache_hd``
    is the real constraint, not the strict ``input_kv == cache_kv``.
    """
    torch.manual_seed(14)
    num_users = 2
    # Sliding spec (alloc): kv=8, bs=64, hd=256 → 131072 elems/block.
    cache_kv = 8
    alloc_block_size = 64
    alloc_head_dim = 256
    # Full spec (call view): kv=2, bs=128, hd=512 → also 131072 elems/block.
    view_kv = 2
    view_block_size = 128
    view_head_dim = 512
    input_seq_len = 128  # exactly one view block per user
    assert cache_kv * alloc_block_size * alloc_head_dim == view_kv * view_block_size * view_head_dim

    max_num_blocks_per_seq = input_seq_len // view_block_size
    max_num_blocks = num_users * max_num_blocks_per_seq

    cache_shape = [max_num_blocks, cache_kv, alloc_block_size, alloc_head_dim]
    cache_torch_alloc = torch.randn(cache_shape).bfloat16().float()
    cache_tt = ttnn.Tensor(cache_torch_alloc, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(num_users, max_num_blocks_per_seq)
    page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)

    # Reference cache in the view's coordinate system. Re-tile the alloc-layout
    # randoms into the (view_kv, view_bs, view_hd) layout so the slots the kernel
    # touches line up with where we expect them in the view.
    cache_ref_view = _permute_view_general(cache_torch_alloc, view_kv, view_block_size, view_head_dim).clone()
    for u in range(num_users):
        x = torch.randn([1, view_kv, input_seq_len, view_head_dim]).bfloat16().float()
        xt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
        ttnn.experimental.paged_fill_cache(cache_tt, xt, page_table_tt, batch_idx=u, block_size=view_block_size)
        for t in range(input_seq_len):
            vb = t // view_block_size
            slot = t % view_block_size
            physical_block = page_table[u, vb].item()
            cache_ref_view[physical_block, 0:view_kv, slot : slot + 1, :] = x[0, :, t : t + 1, :]

    got_alloc = _read_paged_cache(cache_tt)
    got_view = _permute_view_general(got_alloc, view_kv, view_block_size, view_head_dim)

    eq, msg = comp_equal(cache_ref_view, got_view)
    assert eq, f"asymmetric num_heads fill cache round trip mismatch: {msg}"


def test_paged_fill_cache_negative_asymmetric_num_heads_byte_count_mismatch(device):
    """Different ``num_kv_heads`` between cache and input *without* the per-block
    element count being preserved must still be rejected. Guards against the
    relaxation accidentally allowing arbitrary mismatched-byte writes.
    """
    torch.manual_seed(15)
    num_users = 2
    # Cache: 8 * 64 * 256 = 131072 elems/block.
    cache_kv = 8
    alloc_block_size = 64
    alloc_head_dim = 256
    # Input: 4 * 128 * 512 = 262144 elems/block — twice the cache, mismatched.
    view_kv = 4
    view_block_size = 128
    view_head_dim = 512
    input_seq_len = 128
    assert cache_kv * alloc_block_size * alloc_head_dim != view_kv * view_block_size * view_head_dim

    max_num_blocks_per_seq = input_seq_len // view_block_size
    max_num_blocks = num_users * max_num_blocks_per_seq

    cache_torch = torch.randn([max_num_blocks, cache_kv, alloc_block_size, alloc_head_dim]).bfloat16().float()
    cache_tt = ttnn.Tensor(cache_torch, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(num_users, max_num_blocks_per_seq)
    page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)

    x = torch.randn([1, view_kv, input_seq_len, view_head_dim]).bfloat16().float()
    xt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)

    with pytest.raises(RuntimeError, match="geometry mismatch"):
        ttnn.experimental.paged_fill_cache(cache_tt, xt, page_table_tt, batch_idx=0, block_size=view_block_size)


def test_paged_fill_cache_program_cache_scalar_batch_idx(device):
    """Program-cache hit with a DIFFERENT scalar ``batch_idx`` must still fill the
    correct physical block. ``batch_idx`` is excluded from the program hash and baked
    into a writer runtime arg, so a frozen value would re-route the 2nd fill to the
    1st user's block. Two fills reuse one cache entry; each re-allocates its input so
    buffer addresses differ across the hit too.
    """
    torch.manual_seed(20)
    num_users = 2
    num_kv_heads = 1
    block_size = 64
    head_dim = 256
    input_seq_len = block_size  # one block per user

    cache_tt, cache_torch = _make_paged_cache(num_users, num_kv_heads, block_size, head_dim, device)
    ref = cache_torch.clone()

    # page_table[u] = [u]: user u -> physical block u. A frozen batch_idx would send
    # both fills to block 0 and leave block 1 unchanged.
    page_table = torch.arange(num_users, dtype=torch.int32).reshape(num_users, 1)
    page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)

    entries_before = device.num_program_cache_entries()
    for u in range(num_users):
        x = torch.randn([1, num_kv_heads, input_seq_len, head_dim]).bfloat16().float()
        xt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
        ttnn.experimental.paged_fill_cache(cache_tt, xt, page_table_tt, batch_idx=u)
        ref[u, 0:num_kv_heads, :, :] = x[0, :, :, :]

    # Second fill reuses the first fill's entry (only batch_idx differs, which is unhashed).
    assert device.num_program_cache_entries() - entries_before == 1

    got = _read_paged_cache(cache_tt)
    eq, msg = comp_equal(ref, got)
    assert eq, f"program-cache fill mismatch (frozen batch_idx?): {msg}"
