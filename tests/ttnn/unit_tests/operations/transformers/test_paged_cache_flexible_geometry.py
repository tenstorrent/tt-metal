# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``block_size`` override on ``paged_update_cache`` and
``paged_fill_cache``.

These ops gained an optional ``block_size`` kwarg so that callers (e.g.
vLLM's hybrid kv-cache-groups manager) can write into a single physical
cache buffer with different ``(block_size, head_dim)`` views per layer
type, as long as ``num_kv_heads * block_size * head_dim`` (the per-block
element count) is preserved across views. The kernel now derives
``head_dim`` from the input tensor (CUDA-style) and ``block_size`` from
the override or the cache shape.

The tests cover:

* legacy behavior (no override, matching head_dim): byte-identical to
  the pre-change path.
* "flips view" cases in both directions: cache allocated as one shape,
  written through the override as the other shape; verified by
  reinterpreting the round-tripped buffer with ``_permute_tile_grid``
  rather than a torch ``view`` reshape.
* shared buffer with both views interleaved: one physical cache, writes
  from each view land at disjoint block IDs without trampling.
* negative cases: byte-count mismatch and override without page_table
  must both raise ``RuntimeError`` from validation.
"""

import pytest
import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal


# ── Test harness helpers ───────────────────────────────────────────────────


def _sharded_input(device, x_padded):
    """Convert a torch input ``(1, B, padded_heads, head_dim)`` to a height-
    sharded ttnn tensor on ``device``, one shard per batch user.

    Matches the layout that ``run_test_paged_update_cache_decode`` uses in
    ``tests/ttnn/nightly/unit_tests/operations/transformers/test_paged_update_cache.py``.
    """
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

    Why this is needed:
    The op's ``block_size`` override lets a single physical buffer be
    accessed with different ``(block_size, head_dim)`` views per call —
    the kernel addresses DRAM tiles by their *linear* position within a
    block, computed from the view's tile-grid. ttnn's tilize/untilize,
    on the other hand, is bound to the cache's *declared* shape and
    interprets the same DRAM bytes through the cache's tile-grid.

    Concretely: a write to view tile ``(br, hc)`` (linear position
    ``br * view_Wt + hc``) lands at the same DRAM bytes as the cache
    tile ``(BR_alloc, HC_alloc) == (linear // alloc_Wt, linear % alloc_Wt)``.
    Inside the tile, the kernel's 32x32-element layout is preserved.
    The element-level permutation that takes the cache-shape torch
    tensor (after ttnn's untilize) to the view-shape torch tensor
    therefore has to respect the linear-tile correspondence, not the
    naive element-row-major flat offset that ``torch.view`` would use.

    Implementation: split the last two dims into ``(tile_rows, 32,
    tile_cols, 32)``, group ``(tile_rows, tile_cols)`` into a linear
    tile index, re-split that linear index under the view's
    ``(view_tile_rows, view_tile_cols)`` shape, then put the inner
    32x32 dims back. No data copy beyond the standard non-contiguous
    permutes ``.contiguous()`` forces.
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

    # (N, KV, BS, HD) -> split into tile-grid + intra-tile.
    t = t.view(N, KV, alloc_BR_t, TILE, alloc_Wt, TILE)
    # Move tile-grid dims together: (N, KV, alloc_BR_t, alloc_Wt, TILE, TILE).
    t = t.permute(0, 1, 2, 4, 3, 5).contiguous()
    # Flatten tile-grid into linear tile index.
    t = t.view(N, KV, total_tiles, TILE, TILE)
    # Re-split under view's tile-grid.
    t = t.view(N, KV, view_BR_t, view_Wt, TILE, TILE)
    # Put intra-tile rows next to tile-row dim, intra-tile cols next to tile-col dim.
    t = t.permute(0, 1, 2, 4, 3, 5).contiguous()
    # Collapse back to (N, KV, view_BS, view_HD).
    return t.view(N, KV, view_block_size, view_head_dim)


# ── paged_update_cache tests ───────────────────────────────────────────────


@pytest.mark.parametrize("block_size, head_dim", [(64, 256), (128, 256), (64, 512)])
def test_paged_update_cache_legacy_no_override(block_size, head_dim, device):
    """No ``block_size`` kwarg → behavior identical to the pre-change op.

    This guards backward compatibility: every existing caller that doesn't
    pass ``block_size`` must keep working unchanged.
    """
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

    # Update reference: in cache_ref, each user maps virtual block →
    # physical via page_table; for user i, position cache_idxs[i] lands
    # in block page_table[i, cache_idxs[i] // block_size] at row
    # cache_idxs[i] % block_size.
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
    """Buffer allocated for sliding (block=128, head=256); call writes
    with the full-attention view (block=64, head=512) via the override.

    Verifies the kernel addresses the right block boundary, and after
    reading back + ``view``-reshaping to ``(N, 1, 64, 512)`` the written
    data appears at the expected (view-relative) position.

    This is the canonical Gemma4-E2B scenario: the page-size unifier in
    vLLM doubles sliding's block_size from 64 to 128 to match full's
    65,536-byte page; the shared DRAM buffer then needs to support both
    views.

    Parametrized on ``num_kv_heads`` so the multi-head per-block stride
    (``num_kv_heads * block_size * head_dim`` elements) gets exercised:
    single-head values can't catch a regression where the wrong
    dimension is used in the stride math.
    """
    torch.manual_seed(1)

    num_users = 4
    # Allocation view: sliding shape.
    alloc_block_size = 128
    alloc_head_dim = 256
    # Call view: full shape.
    view_block_size = 64
    view_head_dim = 512
    # Both views have ``num_kv_heads * 32768`` elements/block, same per-block bytes.
    assert num_kv_heads * alloc_block_size * alloc_head_dim == num_kv_heads * view_block_size * view_head_dim

    # Max logical sequence length under the *view*; the page table indexes
    # the view's block count.
    max_seq_len_view = 512
    max_num_blocks_per_seq = max_seq_len_view // view_block_size
    max_num_blocks = num_users * max_num_blocks_per_seq

    # Cache is allocated under the legacy/alloc view. The number of
    # cache-blocks must also match (per-block-byte-count is identical, so
    # the block count is identical too).
    assert max_seq_len_view % alloc_block_size == 0
    cache_tt, cache_ref_alloc = _make_paged_cache(
        max_num_blocks, num_kv_heads, alloc_block_size, alloc_head_dim, device
    )

    page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(num_users, max_num_blocks_per_seq)
    page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)

    # Decode update positions, expressed in the *view* coordinate system
    # (block_size=64 here).
    cache_idxs = [10 + u * 17 for u in range(num_users)]  # spread across blocks
    cache_idxs_tt = ttnn.Tensor(torch.tensor(cache_idxs), ttnn.int32).to(device)

    # Input with the view's head_dim.
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

    # Read cache back as allocated, then re-interpret under the view's
    # tile-grid via ``_permute_tile_grid`` (NOT torch.view — see that
    # helper's docstring).
    got_alloc = _read_paged_cache(cache_tt)
    assert got_alloc.shape == (max_num_blocks, num_kv_heads, alloc_block_size, alloc_head_dim)
    got_view = _permute_tile_grid(got_alloc, view_block_size, view_head_dim)

    # Build the reference in the view's coordinate system: take the
    # initial cache, translate to view coords via the same tile-grid
    # permutation, then apply the writes in view coords.
    cache_ref_view = _permute_tile_grid(cache_ref_alloc, view_block_size, view_head_dim).clone()
    for u in range(num_users):
        pos = cache_idxs[u]
        physical_block = page_table[u, pos // view_block_size].item()
        # ``x[0, u]`` is ``[KV, head_dim]``; target slot is ``[KV, 1, head_dim]``.
        # ``unsqueeze(1)`` makes the kv-head dim broadcast-compatible with the
        # 1-token slot — both single-head and multi-head land at the right slot.
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
        # No override here — call view matches the allocation view; this
        # also exercises that legacy callers on a shared buffer keep
        # working when other layers later use the override path.
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

    # Reference: apply sliding write directly in alloc/sliding coords
    # (alloc shape == sliding view), then transition into full view via
    # the tile-grid-aware permutation for the full write — same as the
    # kernel does on device.
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


def test_paged_update_cache_negative_byte_count_mismatch(device):
    """An override that breaks the per-block byte invariant must be
    rejected at validation time, not silently corrupt memory."""
    torch.manual_seed(4)
    num_users = 4
    num_kv_heads = 1
    # Allocate (1, 64, 512) — 32768 elem/block.
    alloc_block_size = 64
    alloc_head_dim = 512
    # Call with mismatched view: 256 * 256 = 65536 (double the bytes).
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
    """``block_size`` only makes sense in paged mode. The validator
    explicitly rejects it when no ``page_table`` is supplied to surface
    the misconfiguration instead of letting the kernel use it silently
    in non-paged mode."""
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
    """Shared fill-cache test body.

    Allocates a paged cache with ``(alloc_block_size, alloc_head_dim)``,
    fills it user-by-user with an input shaped under the view's
    ``(view_block_size, view_head_dim)``, and verifies the round trip
    matches the reference torch fill (viewed in the same coordinate
    system as the write).

    ``block_size_kwarg`` is what we pass to ``paged_fill_cache``; when
    None, we expect alloc-view == call-view (no override).
    """
    max_num_blocks_per_seq = input_seq_len // view_block_size
    assert max_num_blocks_per_seq * view_block_size == input_seq_len
    max_num_blocks = num_users * max_num_blocks_per_seq

    # Allocate as alloc view.
    cache_shape = [max_num_blocks, num_kv_heads, alloc_block_size, alloc_head_dim]
    cache_torch_alloc = torch.randn(cache_shape).bfloat16().float()
    cache_tt = ttnn.Tensor(cache_torch_alloc, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(num_users, max_num_blocks_per_seq)
    page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)

    # One paged_fill_cache call per user. Reference is built in the view
    # coordinate system, mapping alloc-shape bytes to view-shape via the
    # tile-grid-aware permutation (see ``_permute_tile_grid``).
    cache_ref_view = _permute_tile_grid(cache_torch_alloc, view_block_size, view_head_dim).clone()
    for u in range(num_users):
        x = torch.randn([1, num_kv_heads, input_seq_len, view_head_dim]).bfloat16().float()
        xt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)

        kwargs = dict(batch_idx=u)
        if block_size_kwarg is not None:
            kwargs["block_size"] = block_size_kwarg
        ttnn.experimental.paged_fill_cache(cache_tt, xt, page_table_tt, **kwargs)

        # Apply equivalent torch fill: distribute tokens 0..input_seq_len-1
        # to the user's virtual blocks per the page table.
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
    """Cache allocated as sliding ``(128, 256)``, filled via override
    with full ``(64, 512)``.

    Parametrized on ``num_kv_heads`` so the kernel's per-block stride
    (``num_kv_heads * block_size * head_dim``) is exercised on a
    realistic multi-head case as well. ``paged_fill_cache``'s program
    factory derives ``num_heads`` from ``input.padded_shape()[1]``, so
    the multi-head case also catches regressions in the
    ``input_num_heads == cache_num_heads`` validator.
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
    alloc_block_size = 64
    alloc_head_dim = 512
    # 256 * 256 = 65536 elem/block, doubled from alloc's 32768.
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
