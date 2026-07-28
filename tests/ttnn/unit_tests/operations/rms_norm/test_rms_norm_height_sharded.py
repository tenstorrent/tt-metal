# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 4 — HEIGHT_SHARDED placement (local shard, zero-copy).

The phase-0 row split made physical: the shard grid pins which core owns which
tile-rows, the shard height pins how many, each core still holds WHOLE rows and
so the reduce stays entirely local. Three things here are invisible to a PCC
gate and are therefore asserted directly:

1. **The shard really is consumed in place.** An accessor read of a core's OWN
   shard passes every numerical gate — each core still holds its full rows, so
   the accessor reads exactly the right bytes — while the placement was never
   actually implemented. Checked on the descriptor (CB aliased onto the tensor
   buffer), not on the test colour.
2. **No cross-core traffic is added.** HEIGHT sharding must NOT engage the
   Refinement 2/3 combine: `cw == 1`, no groups, no semaphores, no multicast.
   A combine that quietly engaged would still be correct, just slower — and
   nothing else in the suite would notice.
3. **Every element is still counted exactly once.** An all-ones absolute check,
   not a correlational one: a wrong row->core map or a dropped tile only
   rescales each row, and PCC is scale-invariant (this is the R1/R2 lesson).

The ROW_MAJOR case is the deliberate exception to (1): eval.sharding's RM
granule is (1 row, L1_align/elem_bytes columns), so an RM shard is a handful of
STICKS and a tile-row's 32 sticks live on up to 32 different cores. That read is
genuinely NON-local, so it goes through the TensorAccessor — which is what the
accessor is for, and the opposite of re-reading a core's own block.
"""

from __future__ import annotations

import pytest
import torch

import ttnn

from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import default_compute_kernel_config, rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

HEIGHT = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

_TORCH = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32, ttnn.bfloat8_b: torch.bfloat16}


def _ref(x, gamma=None, eps=1e-6):
    xf = x.to(torch.float32)
    out = xf / torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + eps)
    if gamma is not None:
        out = out * gamma.to(torch.float32).reshape(-1)
    return out


def _pcc(a, b):
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _shard(device, shape, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return auto_shard_config(list(shape), HEIGHT, layout=layout, dtype=dtype, device=device)


def _placement(device, tensor):
    grid = device.compute_with_storage_grid_size()
    ht_total, wt_global = pd._tile_geometry(tensor)
    return pd._select_placement(device, grid, tensor, ht_total, wt_global, True)


# ---------------------------------------------------------------------------
# 1. the placement is pinned by the shard, and the reduce stays LOCAL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 256, 512), (4, 8, 32, 256), (1, 1, 8192, 1024)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_height_shard_pins_the_placement_and_keeps_the_reduce_local(device, shape):
    """Knob-turn, not a scheme-change: the shard grid IS the core assignment and
    the shard height IS the per-core row count. Because every core holds whole
    rows, none of the cross-core W-split machinery may engage."""
    mc = _shard(device, shape)
    tt_x = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mc,
    )
    p = _placement(device, tt_x)

    shard_h, shard_w = (int(v) for v in mc.shard_spec.shape)
    shard_cores = ttnn.corerange_to_cores(mc.shard_spec.grid, None, True)

    assert p.num_cores == len(shard_cores), "core assignment is not the shard grid"
    assert [(int(w.core.x), int(w.core.y)) for w in p.works] == [(int(c.x), int(c.y)) for c in shard_cores]
    assert p.rows_core_max == shard_h // 32, "shard height must pin the per-core tile-row count"
    assert p.wt_core == shard_w // 32, "a HEIGHT shard owns the FULL W"

    # The reduction is local: no combine, so nothing cross-core is built.
    assert p.cw == 1 and not p.w_split, "HEIGHT sharding must not engage the W-split combine"
    assert p.groups == []
    # Every tile-row of the tensor is owned exactly once, in order.
    ht_total, _ = pd._tile_geometry(tt_x)
    covered = sorted(r for w in p.works for r in range(w.start_row, w.start_row + w.num_rows))
    assert covered == list(range(ht_total)), "tile-rows are not covered exactly once"


def test_height_shard_program_has_no_combine_wiring(device):
    """The combine's semaphores and multicasts are the cross-core cost. A local
    shard must pay none of it."""
    shape = (1, 1, 256, 512)
    mc = _shard(device, shape)
    tt_x = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mc,
    )
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
    desc = pd.create_program_descriptor(
        tt_x, tt_out, device=device, compute_kernel_config=default_compute_kernel_config()
    )
    assert list(desc.semaphores) == [], "HEIGHT sharding allocated combine semaphores"


# ---------------------------------------------------------------------------
# 2. zero-copy: the CB IS the shard (checked on the descriptor, not the colour)
# ---------------------------------------------------------------------------


def test_height_shard_is_consumed_in_place(device):
    """Both sides: the input CB is aliased onto the input shard and the output CB
    onto the output shard, so the reader issues no input read and the writer no
    output write. An accessor read of a core's own shard would pass every
    numerical gate in this file — hence the descriptor-level assertion."""
    shape = (1, 1, 256, 512)
    mc = _shard(device, shape)
    tt_x = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mc,
    )
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
    desc = pd.create_program_descriptor(
        tt_x, tt_out, device=device, compute_kernel_config=default_compute_kernel_config()
    )
    in_cb = next(cb for cb in desc.cbs if cb.format_descriptors[0].buffer_index == pd.CB_INPUT_TILES)
    out_cb = next(cb for cb in desc.cbs if cb.format_descriptors[0].buffer_index == pd.CB_OUTPUT_TILES)
    assert ttnn.get_cb_address(in_cb) == tt_x.buffer_address(), "input CB is not aliased onto the shard"
    assert ttnn.get_cb_address(out_cb) == tt_out.buffer_address(), "output CB is not aliased onto the shard"


def test_row_major_height_shard_is_a_non_local_read(device):
    """The RM exception. An RM shard is a few STICKS (granule = 1 row), so a
    tile-row's 32 sticks live on up to 32 different cores: the shard is NOT this
    core's block and the read is genuinely non-local, which is exactly the case
    TensorAccessor exists for. Assert we did NOT alias the CB onto it."""
    shape = (1, 1, 64, 128)
    mc = _shard(device, shape, layout=ttnn.ROW_MAJOR_LAYOUT)
    assert int(mc.shard_spec.shape[0]) < 32, "premise: an RM height shard is fewer than 32 sticks"
    tt_x = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=mc,
    )
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, mc)
    desc = pd.create_program_descriptor(
        tt_x, tt_out, device=device, compute_kernel_config=default_compute_kernel_config()
    )
    in_cb = next(cb for cb in desc.cbs if cb.format_descriptors[0].buffer_index == pd.CB_INPUT_TILES)
    assert ttnn.get_cb_address(in_cb) != tt_x.buffer_address()


# ---------------------------------------------------------------------------
# 3. correctness — absolute first, then the reference across the axis surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 256, 512), (1, 1, 2048, 256), (1, 1, 32, 4096), (1, 1, 17, 64), (1, 1, 32, 50)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_height_shard_counts_every_element(device, shape):
    """All-ones => mean(x^2) == 1 exactly, so every output element is
    1/sqrt(1+eps). ABSOLUTE, not correlational: a wrong row->core map, a dropped
    W-tile or a wrong n_reduced only rescales rows, and PCC scores that 0.9999."""
    eps = 1e-6
    mc = _shard(device, shape)
    tt_x = ttnn.from_torch(
        torch.ones(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mc,
    )
    out = ttnn.to_torch(rms_norm(tt_x, epsilon=eps, memory_config=mc)).to(torch.float32)
    expected = 1.0 / (1.0 + eps) ** 0.5
    # Recover the element count the kernel actually summed: out = 1/sqrt(n_used/W + eps).
    W = shape[-1]
    n_used = (out.amin().item() ** -2 - eps) * W
    assert abs(out - expected).max().item() < 5e-3, f"recovered element count {n_used:.2f} of W={W}"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b], ids=["bf16", "fp32", "bf8"])
@pytest.mark.parametrize("has_gamma", [True, False], ids=["gamma", "no_gamma"])
@pytest.mark.parametrize(
    "shape",
    [(1, 1, 256, 512), (4, 8, 32, 256), (1, 1, 17, 64), (2, 512, 1024), (1024, 1024)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_height_shard_matches_reference(device, shape, dtype, has_gamma):
    torch.manual_seed(0)
    mc = _shard(device, shape, dtype=dtype)
    x = torch.randn(shape, dtype=_TORCH[dtype])
    tt_x = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    g = tt_g = None
    if has_gamma:
        g = torch.randn(1, 1, 1, shape[-1], dtype=_TORCH[dtype])
        tt_g = ttnn.from_torch(g, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(rms_norm(tt_x, gamma=tt_g, memory_config=mc)).to(torch.float32)
    assert _pcc(out, _ref(x, g)) > 0.999


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 64, 128), (1, 1, 32, 64), (1, 1, 17, 64)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_row_major_height_shard_matches_reference(device, shape):
    """RM activation on an RM height shard, read through the accessor (see
    test_row_major_height_shard_is_a_non_local_read)."""
    torch.manual_seed(0)
    mc = _shard(device, shape, layout=ttnn.ROW_MAJOR_LAYOUT)
    x = torch.randn(shape, dtype=torch.bfloat16)
    g = torch.randn(1, 1, 1, shape[-1], dtype=torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
    tt_g = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out = ttnn.to_torch(rms_norm(tt_x, gamma=tt_g, memory_config=mc)).to(torch.float32)
    assert _pcc(out, _ref(x, g)) > 0.999


# ---------------------------------------------------------------------------
# 4. the resident shard must not be paid for by shrinking the block
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,dtype",
    [((1, 1, 32, 8192), ttnn.bfloat16), ((1, 1, 32, 4096), ttnn.float32)],
    ids=["bf16-8192", "fp32-4096"],
)
def test_resident_shard_is_not_charged_against_the_cb_budget(device, shape, dtype):
    """A full-W HEIGHT shard puts the input AND output tensor in one core's L1.
    They are ALIASED CBs — the buffer allocator already reserved them — so they
    belong against the L1 bank, not against the program's CB budget. Charging
    them twice makes the halve-and-re-derive loop pay for the shard by
    collapsing WT_CHUNK (measured: 32 -> 1 on bf16 W=8192) or refusing the cell
    outright (fp32 W=4096 missed by 10 KB with 361 KB of the bank still free)."""
    mc = _shard(device, shape, dtype=dtype)
    tt_x = ttnn.from_torch(
        torch.zeros(shape, dtype=_TORCH[dtype]),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mc,
    )
    tt_g = ttnn.from_torch(
        torch.zeros(1, 1, 1, shape[-1], dtype=_TORCH[dtype]), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    p = _placement(device, tt_x)
    blk = pd._derive_blocking(
        tt_x,
        tt_g,
        110,
        p,
        sharded_in=True,
        sharded_out=True,
        l1_total_budget=pd._l1_total_budget(device),
    )
    # Premise: this is the regime where the two budgets differ.
    assert blk.resident_shard_bytes > 0
    assert blk.cb_total_bytes > pd.L1_CB_BUDGET_BYTES, "shape no longer exercises the tight corner"
    # Both walls hold...
    assert blk.program_cb_bytes <= pd.L1_CB_BUDGET_BYTES
    assert blk.cb_total_bytes <= blk.l1_total_budget
    # ...and the block knob did NOT collapse to its minimum to pay for the shard.
    assert blk.wt_chunk > 1, f"WT_CHUNK collapsed to {blk.wt_chunk} (NW={blk.nw})"
    ttnn.deallocate(tt_x)
    ttnn.deallocate(tt_g)


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 32, 7168), (1, 1, 8192, 5120), (1, 1, 64, 128)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_interleaved_blocking_is_unchanged_by_the_two_budget_split(device, shape):
    """An interleaved tensor has no resident shard, so the second wall can never
    bind and every interleaved cell must derive byte-identically to the
    single-budget model. Pinned because that is the whole no-regression argument
    for Refinement 4's budget change."""
    tt_x = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    tt_g = ttnn.from_torch(
        torch.zeros(1, 1, 1, shape[-1], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    grid = device.compute_with_storage_grid_size()
    ht_total, wt_global = pd._tile_geometry(tt_x)
    p = pd._select_placement(device, grid, tt_x, ht_total, wt_global, False)

    def derive(budget):
        b = pd._derive_blocking(tt_x, tt_g, grid.x * grid.y, p, l1_total_budget=budget)
        return (b.wt_chunk, b.nw, b.ht_block, b.x_res_depth, b.gamma_resident, b.cb_total_bytes)

    assert derive(pd._l1_total_budget(device)) == derive(pd.L1_CB_BUDGET_BYTES)
    ttnn.deallocate(tt_x)
    ttnn.deallocate(tt_g)


def test_height_shard_output_keeps_the_input_shard_spec(device):
    """The golden harness passes memory_config=input.memory_config() for every
    sharded cell, so the output must come back with that exact placement."""
    shape = (1, 1, 256, 512)
    mc = _shard(device, shape)
    tt_x = ttnn.from_torch(
        torch.randn(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mc,
    )
    out = rms_norm(tt_x, memory_config=mc)
    assert out.memory_config().memory_layout == HEIGHT
    assert list(out.memory_config().shard_spec.shape) == list(mc.shard_spec.shape)
