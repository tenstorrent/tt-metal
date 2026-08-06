# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side accounting and wrap invariants for phase-disjoint CB aliases."""

import ttnn

from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu_geometry as geo


def _graded_block():
    return geo.Blocking(
        11,
        8,
        7168,
        2048,
        5120 // geo.TILE,
        w_tile=ttnn.tile_size(ttnn.bfloat4_b),
        bfp8_tile=ttnn.tile_size(ttnn.bfloat8_b),
        bf16_tile=ttnn.tile_size(ttnn.bfloat16),
        x_stick=28 * geo.TILE * 2,
        l1_budget=geo.L1_CB_BUDGET - geo.L1_CB_RESERVE,
    )


def test_bfp8_phase_views_share_one_exact_allocation():
    blk = _graded_block()
    logical = blk.cb_layout(True)
    allocations = blk.cb_allocations(True)
    aliases = [(physical_bytes, views) for physical_bytes, views in allocations if len(views) > 1]

    assert len(aliases) == 1
    physical_bytes, views = aliases[0]
    assert tuple(view[0] for view in views) == geo.PHASE_CB_ALIAS
    assert tuple(view[1] for view in views) == (48, 6, 48)
    assert physical_bytes == 48 * ttnn.tile_size(ttnn.bfloat8_b)

    # A logical view's producer pushes in whole logical-capacity cycles. The shared physical
    # capacity must be divisible by each cycle so no reserve can straddle the allocation end.
    physical_pages = physical_bytes // views[0][2]
    assert all(physical_pages % view[1] == 0 for view in views)

    logical_bytes = sum(pages * page for _, pages, page, _ in logical)
    assert logical_bytes == 1_415_104
    assert blk.l1_bytes(True) == 1_356_352
    assert logical_bytes - blk.l1_bytes(True) == 58_752

    flattened = [view[0] for _, allocation_views in allocations for view in allocation_views]
    assert sorted(flattened) == sorted(view[0] for view in logical), "every logical CB index must occur exactly once"


def test_non_bfp8_output_keeps_phase_views_separate():
    blk = _graded_block()
    bf16_tile = ttnn.tile_size(ttnn.bfloat16)
    allocations = blk.cb_allocations(True, bf16_tile)

    assert all(len(views) == 1 for _, views in allocations)


def test_phase_alias_can_be_disabled_for_ablation():
    blk = _graded_block()
    allocations = blk.cb_allocations(True, enable_phase_alias=False)
    assert all(len(views) == 1 for _, views in allocations)
