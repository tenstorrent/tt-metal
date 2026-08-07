# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Allocation, wrap, and device-correctness invariants for phase-disjoint CB aliases."""

import pytest
import torch
import ttnn

from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu
from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu_geometry as geo
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_helpers import weight_memory_configs


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

    assert len(aliases) == 3
    assert any(tuple(view[0] for view in views) == geo.MAILBOX_CB_ALIAS for _, views in aliases)
    physical_bytes, views = next((size, views) for size, views in aliases if views[0][0] == geo.PHASE_CB_ALIAS[0])
    assert tuple(view[0] for view in views) == geo.PHASE_CB_ALIAS
    assert tuple(view[1] for view in views) == (48, 6, 48)
    assert physical_bytes == 48 * ttnn.tile_size(ttnn.bfloat8_b)

    # A logical view's producer pushes in whole logical-capacity cycles. The shared physical
    # capacity must be divisible by each cycle so no reserve can straddle the allocation end.
    physical_pages = physical_bytes // views[0][2]
    assert all(physical_pages % view[1] == 0 for view in views)

    logical_bytes = sum(pages * page for _, pages, page, _ in logical)
    assert logical_bytes == 1_460_288
    assert blk.l1_bytes(True) == 1_389_120
    assert logical_bytes - blk.l1_bytes(True) == 71_168

    flattened = [view[0] for _, allocation_views in allocations for view in allocation_views]
    assert sorted(flattened) == sorted(view[0] for view in logical), "every logical CB index must occur exactly once"


def test_non_bfp8_output_only_uses_bf16_scratch_alias():
    blk = _graded_block()
    bf16_tile = ttnn.tile_size(ttnn.bfloat16)
    allocations = blk.cb_allocations(True, bf16_tile)

    aliases = [(physical_bytes, views) for physical_bytes, views in allocations if len(views) > 1]
    assert len(aliases) == 2
    assert any(tuple(view[0] for view in views) == geo.MAILBOX_CB_ALIAS for _, views in aliases)
    physical_bytes, views = next((size, views) for size, views in aliases if views[0][0] == geo.PHASE_BF16_ALIAS[0])
    assert tuple(view[0] for view in views) == geo.PHASE_BF16_ALIAS
    assert physical_bytes == 12 * bf16_tile


def test_phase_alias_can_be_disabled_for_ablation():
    blk = _graded_block()
    allocations = blk.cb_allocations(True, enable_phase_alias=False)
    aliases = [tuple(view[0] for view in views) for _, views in allocations if len(views) > 1]
    assert aliases == [geo.MAILBOX_CB_ALIAS]


def test_unprofitable_bf16_alias_keeps_separate_allocations():
    """At hidden=3072, aliasing 18 and 24 pages would allocate their 72-page LCM."""
    blk = geo.Blocking(
        11,
        8,
        7168,
        3072,
        1024 // geo.TILE,
        w_tile=ttnn.tile_size(ttnn.bfloat4_b),
        bfp8_tile=ttnn.tile_size(ttnn.bfloat8_b),
        bf16_tile=ttnn.tile_size(ttnn.bfloat16),
        x_stick=28 * geo.TILE * 2,
        l1_budget=geo.L1_CB_BUDGET - geo.L1_CB_RESERVE,
    )
    allocations = blk.cb_allocations(True)

    assert not any(tuple(view[0] for view in views) == geo.PHASE_BF16_ALIAS for _, views in allocations)
    logical_bytes = sum(pages * page for _, pages, page, _ in blk.cb_layout(True))
    assert blk.l1_bytes(True) == logical_bytes - 2 * 64


def test_n3072_rm_pressure_fallback_keeps_mblock8_and_fits_l1():
    """The one-slot X fallback is legal only because RM prefetch lands in cb_x_in first."""
    budget = geo.L1_CB_BUDGET - geo.L1_CB_RESERVE
    blk = geo.Blocking(
        11,
        8,
        7168,
        3072,
        5120 // geo.TILE,
        w_tile=ttnn.tile_size(ttnn.bfloat4_b),
        bfp8_tile=ttnn.tile_size(ttnn.bfloat8_b),
        bf16_tile=ttnn.tile_size(ttnn.bfloat16),
        x_stick=28 * geo.TILE * 2,
        l1_budget=budget,
        x_is_rm=True,
    )

    assert geo.M_BLOCK == 8
    assert blk.depth_x == 1
    assert blk.depth_h == 2
    assert not blk.wd_resident
    assert blk.l1_bytes(True) == 1_418_112
    assert budget - blk.l1_bytes(True) == 43_264


def _pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a -= a.mean()
    b -= b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


@pytest.mark.parametrize(
    "input_dtype,input_layout",
    [(ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT), (ttnn.bfloat8_b, ttnn.TILE_LAYOUT)],
)
def test_phase_alias_tracks_second_m_block_physical_slot(device, input_dtype, input_layout):
    """N=1024 makes gather advance 24 pages inside a 48-page alias.

    The old writer used its local gather cursor as a remote address proxy, so M-block 1's gate
    payload landed in the wrong physical half. Results were usually finite (PCC ~0.38 for block 1),
    which is why finiteness and repeat-determinism tests were insufficient. BF16 output is the
    same computation without the BFP8 phase alias and serves as the device reference here.
    """
    emb, hidden, capacity, count = 7168, 1024, 1024, 512
    grid = (11, 8)
    num_global, num_local, local_id, global_id = 256, 8, 3, 137

    torch.manual_seed(173)
    x = torch.randn((1, 1, capacity, emb), dtype=torch.bfloat16)
    weights = [torch.randn(shape, dtype=torch.bfloat16) for shape in ((emb, hidden), (emb, hidden), (hidden, emb))]
    gu_mc, down_mc = weight_memory_configs(device, emb, hidden, core_grid=grid)
    tt_x = ttnn.from_torch(
        x, dtype=input_dtype, layout=input_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_w = [
        ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
        for w, mc in zip(weights, (gu_mc, gu_mc, down_mc))
    ]
    counts = torch.zeros(num_global, dtype=torch.int32)
    counts[global_id] = count
    idx = torch.tensor([(11 + 37 * i) % num_global for i in range(num_local)], dtype=torch.int32)
    idx[local_id] = global_id
    to_device_u32 = lambda tensor: ttnn.from_torch(  # noqa: E731
        tensor,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_counts, tt_idx = to_device_u32(counts), to_device_u32(idx)

    reference = moe_fused_swiglu(tt_x, *tt_w, tt_counts, tt_idx, local_id, core_grid=grid, dtype=ttnn.bfloat16)
    actual = moe_fused_swiglu(tt_x, *tt_w, tt_counts, tt_idx, local_id, core_grid=grid, dtype=ttnn.bfloat8_b)
    reference = ttnn.to_torch(reference)[0, 0, :count].float()
    actual = ttnn.to_torch(actual)[0, 0, :count].float()

    assert torch.isfinite(actual).all()
    for block in range(2):
        lo, hi = block * 256, (block + 1) * 256
        pcc = _pcc(reference[lo:hi], actual[lo:hi])
        assert pcc > 0.999, f"M-block {block} PCC {pcc} exposes a wrong aliased gather slot"
