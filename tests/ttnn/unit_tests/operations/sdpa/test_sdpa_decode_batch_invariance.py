# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Acceptance tests for tenstorrent/tt-metal#59300.

``paged_scaled_dot_product_attention_decode`` allocates its cores per *padded*
batch row, so the same row computed inside a 4-row graph and inside a 32-row
graph (the other rows idle, ``cur_pos = -1``) takes a different number of
cores and a different reduction order, and its output differs in the last
bits. A serving stack that decodes concurrent requests in different padded
batches therefore cannot reproduce a seeded completion, and a wide graph with
a few active rows leaves most of the grid idle.

Both tests are expected to fail until the kernel allocates cores by active
rows (or reduces in a split-invariant order); the strict ``xfail`` flips to a
failure once it is fixed, which is the reminder to drop the marker.
"""

import time

import pytest
import torch

import ttnn
from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import fa_rand, nearest_n, nearest_pow_2

ISSUE = "tenstorrent/tt-metal#59300: decode SDPA core split depends on the padded batch"


def _paged_cache(b, nkv, s, d, block_size):
    """K, V as a shuffled paged cache plus the page table that maps rows to pages."""
    torch.manual_seed(1234)
    blocks_per_seq = s // block_size
    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)

    def to_paged(cache):
        return (
            cache.reshape(b, nkv, blocks_per_seq, block_size, d)
            .transpose(1, 2)
            .reshape(b * blocks_per_seq, nkv, block_size, d)
        )

    permutation = torch.randperm(b * blocks_per_seq)
    page_table = torch.argsort(permutation).reshape(b, blocks_per_seq)
    return to_paged(K)[permutation], to_paged(V)[permutation], page_table


def _decode(device, tt_K, tt_V, Q, page_table, cur_pos, grid_size, k_chunk_size, nh, d):
    """One decode step for ``Q.shape[1]`` rows; rows with ``cur_pos == -1`` are padding."""
    b = Q.shape[1]
    padded_heads = nearest_pow_2(nearest_n(nh, n=32))
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=padded_heads,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    tt_Q = ttnn.as_tensor(Q, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_page_table = ttnn.Tensor(page_table, ttnn.int32).to(device)
    tt_cur_pos = ttnn.Tensor(torch.tensor(cur_pos, dtype=torch.int32), ttnn.int32).to(device)
    out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        tt_Q,
        tt_K,
        tt_V,
        tt_page_table,
        cur_pos_tensor=tt_cur_pos,
        scale=d**-0.5,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return ttnn.to_torch(out)[:, :, :nh, :]


@pytest.mark.xfail(strict=True, reason=ISSUE)
@pytest.mark.timeout(300)
@pytest.mark.parametrize("active_rows, wide_batch", [(4, 32), (3, 32), (8, 32)])
def test_active_rows_match_bitwise_across_padded_batches(device, active_rows, wide_batch):
    """The active rows must produce bitwise-identical outputs whether the op runs
    with ``active_rows`` rows or inside a ``wide_batch`` graph padded with idle rows."""
    nh, nkv, d, s, block_size, k_chunk = 8, 1, 128, 4096, 128, 128
    grid_size = (8, 8)
    grid = device.compute_with_storage_grid_size()
    if grid_size[0] > grid.x or grid_size[1] > grid.y:
        pytest.skip(f"needs an {grid_size} grid, device has {grid.x}x{grid.y}")

    paged_k, paged_v, page_table = _paged_cache(wide_batch, nkv, s, d, block_size)
    tt_K = ttnn.as_tensor(paged_k, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    tt_V = ttnn.as_tensor(paged_v, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

    torch.manual_seed(7)
    Q = fa_rand(1, wide_batch, nh, d)
    # Active rows sit deep in their context; the padded rows carry cur_pos -1 and whatever Q.
    cur_pos = [s - 1 - 37 * i for i in range(active_rows)] + [-1] * (wide_batch - active_rows)

    narrow = _decode(
        device, tt_K, tt_V, Q[:, :active_rows], page_table[:active_rows], cur_pos[:active_rows], grid_size, k_chunk, nh, d
    )
    wide = _decode(device, tt_K, tt_V, Q, page_table, cur_pos, grid_size, k_chunk, nh, d)

    narrow_rows = narrow[:, :active_rows]
    wide_rows = wide[:, :active_rows]
    max_abs = (narrow_rows.float() - wide_rows.float()).abs().max().item()
    assert torch.equal(narrow_rows, wide_rows), (
        f"{active_rows} active rows differ between the {active_rows}-row and the {wide_batch}-row graph "
        f"(max |diff| {max_abs:.3e}); the per-row core split changed the reduction order"
    )


@pytest.mark.xfail(strict=True, reason=ISSUE)
@pytest.mark.timeout(600)
def test_wide_graph_with_few_active_rows_uses_the_idle_cores(device):
    """Four deep active rows inside a 32-row graph should not take much longer than the
    same four rows in a 4-row graph: idle rows do no work, so their cores should help."""
    nh, nkv, d, s, block_size, k_chunk = 8, 1, 128, 8192, 128, 128
    grid_size = (8, 8)
    grid = device.compute_with_storage_grid_size()
    if grid_size[0] > grid.x or grid_size[1] > grid.y:
        pytest.skip(f"needs an {grid_size} grid, device has {grid.x}x{grid.y}")
    active_rows, wide_batch, iters = 4, 32, 20

    paged_k, paged_v, page_table = _paged_cache(wide_batch, nkv, s, d, block_size)
    tt_K = ttnn.as_tensor(paged_k, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    tt_V = ttnn.as_tensor(paged_v, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    Q = fa_rand(1, wide_batch, nh, d)
    cur_pos = [s - 1] * active_rows + [-1] * (wide_batch - active_rows)

    def timed(rows):
        q, pt, pos = Q[:, :rows], page_table[:rows], cur_pos[:rows]
        _decode(device, tt_K, tt_V, q, pt, pos, grid_size, k_chunk, nh, d)  # compile
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            _decode(device, tt_K, tt_V, q, pt, pos, grid_size, k_chunk, nh, d)
        ttnn.synchronize_device(device)
        return (time.perf_counter() - t0) / iters

    narrow_s, wide_s = timed(active_rows), timed(wide_batch)
    assert wide_s <= 1.5 * narrow_s, (
        f"4 active rows take {wide_s * 1e3:.2f} ms in the 32-row graph vs {narrow_s * 1e3:.2f} ms in the "
        f"4-row graph: the idle rows' cores are not used"
    )
