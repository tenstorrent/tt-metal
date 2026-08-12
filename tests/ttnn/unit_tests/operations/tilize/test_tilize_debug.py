# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic debugging / structure tests for tilize. DO NOT DELETE.

Three groups:

1. **Deterministic value tests** — all-ones and position-encoded inputs, so a
   DEVICE_PRINT session has hand-calculable expectations and a data-reordering
   bug is visible rather than merely "close but wrong". tilize is a bijection on
   byte positions, so every check here is EXACT (`torch.equal`), not PCC.

2. **Blocking-model tests** (host-only, no device) — pin the block decode and
   the knob derivation for every distribution regime named in op_design.md §5.4,
   including that the linearization degenerates to the pure height split when
   `n_wchunks == 1` and to the pure width split when `nt_h == 1`.

3. **Multi-core distribution tests** — written when Phase 0's SUPPORTED
   rectangle only *accepted* `use_multicore=False`, so they call the
   module-private `_dispatch` (which skips validate) to prove the 2-D split the
   design commits to is wired and correct on all three regimes. Refinement 1
   (A1) flipped the axis; these stay as-is because they also cover the lever
   OFF-arms, which the public entry point deliberately does not expose.

4. **Structural pins** for the lever ledger's argument-based closures.

5. **Refinement 1 (A1 + A5 + A6)** — the same generality axes exercised through
   the PUBLIC entry point, plus the depth-2 L1 fallback the `*_l1` buffer
   directions made load-bearing.
"""

import pytest
import torch

import ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize.tilize import _dispatch
from ttnn.operations.tilize.tilize_program_descriptor import (
    TARGET_READ_BYTES,
    blocking,
    cb_budget_bytes,
    cb_depth_for,
    create_program_descriptor,
    l1_bytes_per_core,
    plan_cores,
    wt_block_max,
)


def _to_device(torch_tensor, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


# ---------------------------------------------------------------------------
# 1. Deterministic value tests
# ---------------------------------------------------------------------------


def test_all_ones_single_tile(device):
    """All-ones: every tile position must read back exactly 1.0, so a dropped or
    stale L1 region shows up as a 0 rather than as a small numeric error."""
    torch_input = torch.ones(1, 1, 32, 32, dtype=torch.bfloat16)
    tt_output = tilize(_to_device(torch_input, device), use_multicore=False)
    result = ttnn.to_torch(tt_output)
    assert torch.equal(result, torch_input), f"mismatch: {(result.float() - 1.0).abs().max()}"


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 32, 32), (1, 1, 64, 128), (1, 1, 32, 512), (2, 3, 64, 96)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_position_encoded_identity(device, shape):
    """t[..., r, c] = r*1000 + c (exact in bf16 for these extents is NOT
    guaranteed, so encode with values that are): every element carries its own
    (row, col), so any face/tile reordering is immediately visible."""
    H, W = shape[-2], shape[-1]
    rows = torch.arange(H).reshape(H, 1) * 32
    cols = torch.arange(W).reshape(1, W) * 0.5
    plane = (rows + cols).to(torch.bfloat16)
    torch_input = plane.expand(*shape).contiguous().to(torch.bfloat16)

    tt_output = tilize(_to_device(torch_input, device), use_multicore=False)
    result = ttnn.to_torch(tt_output)
    assert torch.equal(result, torch_input), (
        f"data reordered; first mismatch at " f"{(result != torch_input).nonzero()[:1].tolist()}"
    )


# ---------------------------------------------------------------------------
# 2. Blocking-model tests (host-only)
# ---------------------------------------------------------------------------


def test_wt_block_max_is_a_byte_target():
    """WT_BLOCK_MAX is derived from TARGET_READ_BYTES, never per-dtype literals."""
    assert wt_block_max(2) == TARGET_READ_BYTES // 64  # bf16 -> 8
    assert wt_block_max(4) == TARGET_READ_BYTES // 128  # fp32 -> 4
    assert wt_block_max(1) == TARGET_READ_BYTES // 32  # uint8 -> 16
    assert wt_block_max(1024) == 2  # narrow floor: row_bytes >= 64 B


@pytest.mark.parametrize(
    "shape, nt_h, Wt",
    [
        ([1, 1, 2048, 2048], 64, 64),  # grid-filling square
        ([1, 1, 32, 16384], 1, 512),  # wide-short: width split only
        ([1, 1, 2048, 64], 64, 2),  # tall-narrow: pure height split
        ([1, 1, 32, 64], 1, 2),  # tiny: one block
        ([2, 3, 64, 96], 12, 3),  # multi-batch fold
        ([1, 2, 3, 64, 32], 12, 1),  # rank 5 folds the same way
    ],
    ids=["square", "wide_short", "tall_narrow", "tiny", "multi_batch", "rank5"],
)
def test_blocking_regimes(shape, nt_h, Wt):
    """Geometry is pinned by the SHAPE; the block count is derived from the KNOB.
    Restating `n_wchunks` as a literal here would just re-hardcode
    TARGET_READ_BYTES in a second place and break the moment the knob is turned —
    which is what this file is supposed to prevent."""
    blk = blocking(shape, tile_height=32, elem_size=2)
    assert (blk["nt_h"], blk["Wt"]) == (nt_h, Wt)

    expected_wt_block = min(Wt, wt_block_max(2))
    n_wchunks = -(-Wt // expected_wt_block)  # ceil
    total_blocks = nt_h * n_wchunks
    assert blk["wt_block"] == expected_wt_block
    assert blk["n_wchunks"] == n_wchunks
    assert blk["total_blocks"] == total_blocks
    # The tail block is never wider than a full block, so the CB (sized for
    # WT_BLOCK) always has room for it.
    assert blk["wt_tail"] <= blk["wt_block"]
    # Every tile is covered exactly once by the block decode.
    covered = sum(
        blk["wt_tail"] if wchunk == n_wchunks - 1 else blk["wt_block"]
        for wchunk in range(n_wchunks)
        for _ in range(nt_h)
    )
    assert covered == nt_h * Wt


def test_blocking_degenerates_to_height_split_when_narrow():
    """With `Wt <= WT_BLOCK_MAX` the block index IS the tile-row index, so the
    2-D linearization is byte-identical to a pure height split (no gate)."""
    blk = blocking([1, 1, 2048, 64], tile_height=32, elem_size=2)
    assert blk["n_wchunks"] == 1
    assert blk["total_blocks"] == blk["nt_h"]
    assert blk["wt_block"] == blk["Wt"]


def test_cb_l1_is_constant_in_w():
    """Per-core CB bytes must not grow with W (risk 2: a CB sized by Wt OOMs the
    wide-short bench)."""
    bound = 2 * wt_block_max(2) * (2048 + 2048)  # CB_DEPTH=2, bf16 tiles: 64 KiB
    footprints = {}
    for W in (64, 2048, 16384, 65536):
        blk = blocking([1, 1, 32, W], tile_height=32, elem_size=2)
        footprints[W] = 2 * blk["wt_block"] * (2048 + 2048)
    assert max(footprints.values()) == bound, footprints
    # Narrow shapes are SMALLER (wt_block = Wt < WT_BLOCK_MAX); nothing grows
    # past the bound, and everything W >= 256 sits exactly on it.
    assert all(v <= bound for v in footprints.values()), footprints
    assert footprints[2048] == footprints[16384] == footprints[65536] == bound, footprints


# ---------------------------------------------------------------------------
# 3. Multi-core distribution (the design's binding work split)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 4096),  # wide-short: nt_h == 1, must NOT collapse to 1 core
        (1, 1, 2048, 64),  # tall-narrow: pure height split
        (1, 1, 256, 256),  # square-ish: 2-D rectangle per core
        (2, 3, 64, 96),  # multi-batch
    ],
    ids=["wide_short", "tall_narrow", "square", "multi_batch"],
)
def test_multicore_identity(device, shape):
    """The 2-D split is wired and correct on every regime, even though Phase 0's
    SUPPORTED rectangle does not yet claim it (hence `_dispatch`)."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape).bfloat16()
    tt_output = _dispatch(_to_device(torch_input, device), use_multicore=True)
    result = ttnn.to_torch(tt_output)
    assert torch.equal(result, torch_input), f"multi-core mismatch on {shape}"


@pytest.mark.parametrize(
    "levers",
    [
        dict(width_split=0),
        dict(row_wise=0),
        dict(target_read_bytes=128),
        dict(target_read_bytes=256),
        dict(target_read_bytes=1024),
        dict(target_read_bytes=2048),
        dict(barrier_per_block=0),
        dict(coalesce_writes=0),
        dict(noc_split=0),
        dict(double_buffer=0),
    ],
    ids=lambda d: "-".join(f"{k}{v}" for k, v in d.items()),
)
def test_lever_off_arms_are_still_correct(device, levers):
    """Each lever's OFF arm must still compute the right answer — a
    counterfactual that produces garbage measures nothing. (The `stub_*` arms are
    excluded: they are ablations whose output is wrong by design.)"""
    torch.manual_seed(42)
    shape = (1, 1, 96, 288)  # 3 tile-rows x 9 tile-columns: exercises the tail block
    torch_input = torch.randn(shape).bfloat16()
    tt_output = _dispatch(_to_device(torch_input, device), use_multicore=True, levers=levers)
    assert torch.equal(ttnn.to_torch(tt_output), torch_input), f"lever arm {levers} is incorrect"


# ---------------------------------------------------------------------------
# 4. Structural pins for the lever ledger's argument-based closures
#    (lever_ledger.json cites these by name; a structural claim that is not
#    mechanically pinned is just an argument)
# ---------------------------------------------------------------------------


def test_b12_multicast_is_structurally_absent(device):
    """master.md B12 (multicast instead of N unicasts) is not merely unapplied —
    it is structurally inapplicable. tilize's map is a BIJECTION on byte
    positions: the single input operand varies along BOTH split axes, so no two
    cores ever read the same byte and there is no fan-out to multicast. Pinned by
    asserting the program declares no semaphores (a multicast handshake needs
    them) and that the per-core block ranges are disjoint and cover the space
    exactly once."""
    torch.manual_seed(0)
    shape = (1, 1, 256, 256)
    tt_input = _to_device(torch.randn(shape).bfloat16(), device)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    pd = create_program_descriptor(tt_input, out, use_multicore=True)
    assert list(pd.semaphores) == [], "a multicast handshake would need semaphores"

    blk = blocking(list(shape), tile_height=32, elem_size=2)
    _cores, _all_cores, per_core = plan_cores(device, blk["total_blocks"], use_multicore=True)
    assert sum(per_core) == blk["total_blocks"], "block ranges must tile the space exactly once"


def test_c17_in_place_is_structurally_impossible(device):
    """master.md C17 (in-place / no-copy) cannot apply: tilize CHANGES byte
    positions, and the tilize LLK helper itself static_asserts
    `input_dfb != output_dfb` (tilize_helpers.inl). Pinned by asserting the two
    CBs are distinct slots and that the op returns a different buffer."""
    from ttnn.operations.tilize.tilize_program_descriptor import CB_INPUT_STICKS, CB_OUTPUT_TILES

    assert CB_INPUT_STICKS != CB_OUTPUT_TILES

    torch.manual_seed(0)
    torch_input = torch.randn(1, 1, 64, 64).bfloat16()
    tt_input = _to_device(torch_input, device)
    tt_output = tilize(tt_input, use_multicore=False)
    assert tt_output.buffer_address() != tt_input.buffer_address(), "layout conversion cannot be in-place"
    assert tt_input.layout == ttnn.ROW_MAJOR_LAYOUT and tt_output.layout == ttnn.TILE_LAYOUT


@pytest.mark.parametrize("elem_size", [2, 4], ids=["2B", "4B"])
def test_b11_transfers_are_alignment_clean(elem_size):
    """master.md B11 (alignment): for every element width in reach of the current
    + next-refinement dtype set (2 B bf16, 4 B fp32/uint32), every transfer this
    op issues is already a multiple of the DRAM alignment unit at EVERY width —
    a read is `WT_BLOCK * 32 * elem` bytes and a write is a whole tile page — so
    there is no misalignment for lever B11 to fix. The 1-byte exception is pinned
    separately below."""
    align = ttnn.get_dram_alignment()
    for Wt in (1, 2, 3, 7, 16, 17, 64, 512):
        blk = blocking([1, 1, 32, Wt * 32], tile_height=32, elem_size=elem_size)
        for w in (blk["wt_block"], blk["wt_tail"]):
            read_bytes = w * 32 * elem_size
            assert read_bytes % align == 0, f"read of {read_bytes} B is not {align} B-aligned"
            assert read_bytes >= align, f"read of {read_bytes} B is below the alignment floor"
        write_bytes = 32 * 32 * elem_size  # one tile page
        assert write_bytes % align == 0


def test_b11_uint8_narrow_stick_is_the_known_alignment_gap():
    """The ONE alignment case this op does not cover, recorded so it cannot ship
    silently: a 1-byte dtype whose `Wt == 1` gives a 32 B read, below this part's
    DRAM alignment unit. WT_BLOCK_MAX's `max(2, ...)` floor does not save it
    because `WT_BLOCK = min(Wt, WT_BLOCK_MAX)` clamps to Wt.

    uint8 is out of SUPPORTED at Phase 0, and refinement A5b explicitly owns the
    alignment-aware narrow-stick reader. If a future change adds uint8 to
    SUPPORTED without that reader, this test is the pin that says so."""
    align = ttnn.get_dram_alignment()
    blk = blocking([1, 1, 32, 32], tile_height=32, elem_size=1)  # uint8, Wt == 1
    assert blk["wt_block"] == 1
    narrow_read = blk["wt_block"] * 32 * 1
    assert narrow_read < align, (
        f"uint8 Wt==1 now reads {narrow_read} B >= alignment {align} B — the gap this "
        "test documents has been closed; update the A5b note and the B11 ledger row"
    )
    # Wt >= 2 is already clean at 1 byte, so the gap is confined to Wt == 1.
    assert (blocking([1, 1, 32, 64], 32, 1)["wt_block"] * 32) % align == 0


def test_multicore_fills_the_grid_on_wide_short(device):
    """A height-only split would strand nt_h == 1 on ONE core. Assert the core
    count directly, do not infer it from the output."""
    grid = device.compute_with_storage_grid_size()
    grid_cores = grid.x * grid.y

    blk = blocking([1, 1, 32, 16384], tile_height=32, elem_size=2)
    cores, _all_cores, per_core = plan_cores(device, blk["total_blocks"], use_multicore=True)
    assert len(cores) == min(blk["total_blocks"], grid_cores), (
        f"wide-short ran on {len(cores)} cores, expected " f"{min(blk['total_blocks'], grid_cores)}"
    )
    assert sum(per_core) == blk["total_blocks"]

    # ... and the single-core parameter value still lands on exactly one core.
    cores_1, _, per_core_1 = plan_cores(device, blk["total_blocks"], use_multicore=False)
    assert len(cores_1) == 1 and per_core_1 == [blk["total_blocks"]]


# ---------------------------------------------------------------------------
# 5. Refinement 1 (A1 + A5 + A6) — the interleaved path at full generality
#
#    The four axes this refinement flipped into SUPPORTED are knob VALUES, not
#    new code paths, so what is worth pinning is (a) that the public entry point
#    (not `_dispatch`) now reaches them, (b) that the grid is actually filled on
#    the golden wide-short cell, and (c) the ONE piece of new host logic: the
#    depth-2 fallback now sees the L1-resident operands the `*_l1` buffer
#    directions introduce.
# ---------------------------------------------------------------------------


def test_a1_wide_short_golden_cell_fills_the_grid(device):
    """The golden suite's wide-short cell (not just the 16384-wide bench shape)
    must occupy `min(total_blocks, grid_cores)` cores. Asserted on the core
    count, never inferred from the output."""
    grid = device.compute_with_storage_grid_size()
    blk = blocking([1, 1, 32, 4096], tile_height=32, elem_size=2)
    cores, _all_cores, per_core = plan_cores(device, blk["total_blocks"], use_multicore=True)

    assert blk["nt_h"] == 1, "the cell must stay the nt_h == 1 regime"
    assert len(cores) == min(blk["total_blocks"], grid.x * grid.y)
    assert len(cores) > 1, "a height-only split would strand this cell on one core"
    assert sum(per_core) == blk["total_blocks"]


@pytest.mark.parametrize(
    "shape",
    [(64, 128), (2, 32, 64), (1, 2, 3, 64, 32)],
    ids=["rank2", "rank3", "rank5"],
)
@pytest.mark.parametrize(
    "in_mem, out_mem",
    [
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
        (ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
        (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
    ],
    ids=["dram_to_l1", "l1_to_l1", "l1_to_dram"],
)
def test_a5_rank_and_buffer_direction_cross(device, shape, in_mem, out_mem):
    """rank x buffer-direction cross through the PUBLIC entry point: `nimg =
    prod(shape[:-2])` is rank-agnostic and the buffer type is a TensorAccessor CT
    arg, so every combination must be exact."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape).bfloat16()
    tt_input = _to_device(torch_input, device, memory_config=in_mem)
    tt_output = tilize(tt_input, memory_config=out_mem)
    assert list(tt_output.shape) == list(shape)
    assert torch.equal(ttnn.to_torch(tt_output), torch_input)


@pytest.mark.parametrize("use_double_buffer", [False, True], ids=["depth1", "depth2"])
def test_a6_double_buffer_is_exact_on_both_depths(device, use_double_buffer):
    """A6: depth-1 halves per-core CB L1 and must stay bit-exact. Uses a shape
    with a TAIL column-block (Wt = 9 > WT_BLOCK_MAX at 1024 B) because depth 1
    leaves the reader exactly `wt_block` pages of slack — the geometry where an
    off-by-one CB size would deadlock rather than merely slow down."""
    torch.manual_seed(42)
    torch_input = torch.randn(1, 1, 96, 288).bfloat16()
    tt_output = tilize(_to_device(torch_input, device), use_double_buffer=use_double_buffer)
    assert torch.equal(ttnn.to_torch(tt_output), torch_input)


def test_a6_depth2_fallback_is_pure_and_monotone():
    """`cb_depth_for` is the whole depth knob: depth 2 only when asked for AND it
    fits. Pure, so the fallback is pinned without a device."""
    assert cb_depth_for(want_depth2=True, depth2_bytes=1000, budget_bytes=1000) == 2
    assert cb_depth_for(want_depth2=True, depth2_bytes=1001, budget_bytes=1000) == 1
    assert cb_depth_for(want_depth2=False, depth2_bytes=0, budget_bytes=1 << 30) == 1


def test_a6_cb_budget_subtracts_the_l1_resident_operands():
    """The A5 buffer directions put operands in the SAME per-core L1 the CBs
    spend, so the budget must shrink with them (Phase 0 could assume DRAM-only
    operands). The fixed fraction still caps it from above."""
    unreserved = 1_500_000
    free = cb_budget_bytes(unreserved, 0)
    assert free == int(unreserved * 0.5), "with no L1 operands the fraction binds"
    # A big L1-resident operand binds instead, and drives depth-2 off.
    tight = cb_budget_bytes(unreserved, 1_450_000)
    assert tight == 50_000 < free
    assert cb_depth_for(want_depth2=True, depth2_bytes=131_072, budget_bytes=tight) == 1
    assert cb_depth_for(want_depth2=True, depth2_bytes=131_072, budget_bytes=free) == 2
    # Never negative: an over-subscribed L1 clamps to 0 rather than going wild.
    assert cb_budget_bytes(unreserved, 2 * unreserved) == 0


def test_a6_l1_bytes_per_core_counts_only_interleaved_l1(device):
    """A DRAM operand costs no worker L1; an L1-interleaved one costs
    `ceil(pages/banks) * aligned_page_size`. (An L1-SHARDED operand is 0 by
    design — Refinement 2 aliases the CB onto the shard, so counting it would
    double-count.)"""
    grid = device.compute_with_storage_grid_size()
    banks = grid.x * grid.y
    torch_input = torch.randn(1, 1, 512, 512).bfloat16()

    dram_t = _to_device(torch_input, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert l1_bytes_per_core(dram_t, banks) == 0

    l1_t = _to_device(torch_input, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    expected = -(-l1_t.buffer_num_pages() // banks) * l1_t.buffer_aligned_page_size()
    assert l1_bytes_per_core(l1_t, banks) == expected > 0
