# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 3b — `g_dram_to_sharded`'s unattributed residual, measured then acted on.

Refinement 3 left 2 340 ns of `g_dram_to_sharded`'s 16 006 unattributed by any
ablation variant and handed the question here. This refinement answers it with a
measurement (lever 1), then acts on the two candidate levers it selects — one
lands, one is refuted.

* **Lever 1 — per-RISC Tracy timeline** (MEASUREMENT, `TILIZE_ZONES=1`). An
  instrumented copy of each kernel's per-block loop with a `DeviceZoneScopedN`
  around every stage. It is the only instrument that can say *which RISC is
  waiting when*: `no_dm` keeps the address-gen sink and `no_compute` keeps the CB
  dance, so the residual falls between their attributions. Measured on the shipped
  `alias_out` plan (`[1,1,2048,512]` -> BLOCK-sharded, 64 cores, 8 blocks/core):

      TRISC0 (unpack) blocked in cb_wait_front   15 592 / 17 316 = 90 %
      NCRISC blocked in cb_reserve_back             350 / 17 017 =  2 %
      BRISC  blocked in cb_wait_front            17 372 / 17 522 = 99 %

  i.e. the reads are the bound, compute never is, and the writer kernel is pure
  launch overhead. Because the zone writes perturb the timing, the shipped
  branches are left byte-for-byte alone and the instrumented loops are a separate
  compile-time branch — which is exactly what these tests pin: the variant must be
  OFF by default, and CORRECT when on (unlike the ablations, it changes no read
  and no CB count, so `torch.equal` still holds).

* **Lever 2 — drop the writer kernel on `alias_out`** (SHIPPED). With the output
  CB aliased, the writer's whole body is one `cb_wait_front` / `cb_pop_front`, and
  the CB never needs recycling: it has exactly `shard_tiles` pages and compute
  pushes exactly `shard_tiles`, so `cb_reserve_back` never blocks and the firmware
  re-zeroes the counters every launch. The program ships TWO kernels.
  Measured -49 ns / -0.71 % on the small alias_out shape (12 rounds, CV 0.4-0.6 %),
  corroborated by the timeline: BRISC-FW end -42 cycles with the NCRISC and TRISC
  spans unchanged. The risk it carries is not perf, it is (a) a cross-launch CB
  leak and (b) losing the program-cache re-binding that the writer's `dst_addr`
  runtime arg used to carry — both pinned below.

* **Lever 3 — hoisted interleaved bank table** (REFUTED, +0.9 %).
  `dataflow_kernel_lib::InterleavedStickBands` primes the bank table ONCE per core
  instead of once per band, removing 84 of 96 `accessor.get_noc_addr` calls, and
  is *slower* than B13, which removes 160 of 256 and is 5.4 % faster than bare:

      variant                       | ns     | vs B13
      ------------------------------|--------|--------
      hoisted table (lever 3)       | 16 062 | 1.009
      B13 per-band table (shipped)  | 15 916 | 1.000
      neither                       | 16 827 | 1.057

  The reading: what B13 buys is the ARMED command buffer, not the arithmetic — and
  the arithmetic is hidden anyway, because `noc_async_read` spins in
  `noc_cmd_buf_ready` on a DRAM-service-bound loop, so row r+1's address math
  overlaps row r's service. The lever's code and gate are kept (gate
  identity-false, `TILIZE_LEVER_BT=2` forces it) so the verdict stays
  re-measurable — and it must stay bit-exact when forced, which is what most of
  the lever-3 tests here check.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import pytest
import torch

import ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize import tilize_program_descriptor as tpd
from ttnn.operations.tilize.tilize_program_descriptor import (
    BANK_TABLE_MIN_BLOCKS,
    bank_table_pays,
    build_plan,
    create_program_descriptor,
    drop_writer_pays,
)

_L1 = ttnn.BufferType.L1
_ROW = ttnn.ShardOrientation.ROW_MAJOR
_COL = ttnn.ShardOrientation.COL_MAJOR
_HEIGHT = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
_WIDTH = ttnn.TensorMemoryLayout.WIDTH_SHARDED
_BLOCK = ttnn.TensorMemoryLayout.BLOCK_SHARDED
DRAM = ttnn.DRAM_MEMORY_CONFIG


def _crs(end_x, end_y):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(end_x, end_y))})


def _shard(scheme, grid, shape, orientation=_ROW):
    return ttnn.MemoryConfig(scheme, _L1, ttnn.ShardSpec(grid, shape, orientation))


@contextmanager
def _env(**kwargs):
    """Force compile-time switches (levers / ablations / zones) for one test."""
    saved = {}
    try:
        for name, value in kwargs.items():
            saved[name] = os.environ.get(name)
            os.environ[name] = str(value)
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _levers(**kwargs):
    return _env(**{f"TILIZE_LEVER_{k.upper()}": v for k, v in kwargs.items()})


def _make(device, shape, in_cfg=DRAM, dtype=ttnn.bfloat16):
    n = 1
    for d in shape:
        n *= d
    if dtype == ttnn.float32:
        torch_input = (torch.arange(n, dtype=torch.float32) % 65536).reshape(shape)
    else:
        torch_input = ((torch.arange(n, dtype=torch.float32) % 4096).reshape(shape)).bfloat16()
    tt_input = ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_cfg
    )
    return torch_input, tt_input


def _plan(device, shape, in_cfg, out_cfg, *, dtype=ttnn.bfloat16, use_multicore=True, use_double_buffer=None):
    _, tt_input = _make(device, shape, in_cfg, dtype)
    tt_output = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dtype, ttnn.TILE_LAYOUT, device, out_cfg)
    return build_plan(tt_input, tt_output, device, use_multicore=use_multicore, use_double_buffer=use_double_buffer)


def _exact(device, shape, in_cfg, out_cfg, *, dtype=ttnn.bfloat16, repeats=1, use_double_buffer=None):
    """tilize is value-preserving, so the oracle is `torch.equal`, not a tolerance.

    `arange` input, not `randn`: every element is unique, so a permutation (the
    failure mode a wrong read order produces) cannot cancel out.
    """
    torch_input, tt_input = _make(device, shape, in_cfg, dtype)
    for _ in range(repeats):
        tt_output = tilize(tt_input, out_cfg, use_double_buffer=use_double_buffer)
        assert tt_output.layout == ttnn.TILE_LAYOUT
        result = ttnn.to_torch(tt_output)
        assert torch.equal(result.float(), torch_input.float()), (
            f"not bit-exact on {shape}: {(result.float() - torch_input.float()).abs().max()} max abs, "
            f"{(result.float() != torch_input.float()).sum()} mismatching elements"
        )
    return tt_input


# The crossover geometries lever 2 changes the program shape of. Every scheme x
# orientation, plus the width-chunked shard (chunks_per_core > 1, i.e. the
# `blocks_row_major` read order) and a shard with a single block per core.
ALIAS_OUT_GEOMETRIES = [
    ("block_row", (1, 1, 256, 128), _shard(_BLOCK, _crs(1, 1), (128, 64), _ROW)),
    ("block_col", (1, 1, 256, 128), _shard(_BLOCK, _crs(1, 1), (128, 64), _COL)),
    ("height_row", (1, 1, 256, 128), _shard(_HEIGHT, _crs(1, 1), (64, 128), _ROW)),
    ("height_col", (1, 1, 256, 128), _shard(_HEIGHT, _crs(1, 1), (64, 128), _COL)),
    ("width_row", (1, 1, 128, 256), _shard(_WIDTH, _crs(1, 1), (128, 64), _ROW)),
    ("width_col", (1, 1, 128, 256), _shard(_WIDTH, _crs(1, 1), (128, 64), _COL)),
    # A shard 4 tiles wide -> chunks_per_core > 1 -> tile-row-outer read order.
    ("block_wide_shard", (1, 1, 256, 256), _shard(_BLOCK, _crs(1, 1), (128, 128), _ROW)),
    # Exactly one tile-row block per core: the CB holds the whole shard in one go.
    ("height_one_block", (1, 1, 128, 64), _shard(_HEIGHT, _crs(3, 0), (32, 64), _ROW)),
]


# ---------------------------------------------------------------------------
# Lever 2 — the writer kernel is gone on `alias_out`
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", ALIAS_OUT_GEOMETRIES, ids=[e[0] for e in ALIAS_OUT_GEOMETRIES])
def test_alias_out_ships_two_kernels_and_stays_bit_exact(device, entry):
    """The whole lever, asserted STRUCTURALLY and then numerically.

    Structural first: a duration delta of 0.7 % is not a proof that the writer is
    gone, so the descriptor's kernel list is what says the lever landed.
    """
    _, shape, out_cfg = entry
    plan = _plan(device, shape, DRAM, out_cfg)
    assert plan["path"] == "alias_out", f"expected the one-sided alias, got {plan['path']}"
    assert plan["drop_writer"] == 1

    _, tt_input = _make(device, shape)
    tt_output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, out_cfg
    )
    descriptor = create_program_descriptor(tt_input, tt_output, plan)
    sources = [k.kernel_source for k in descriptor.kernels]
    assert len(sources) == 2, f"expected reader + compute only, got {len(sources)}: {sources}"
    assert not any("tilize_writer" in s for s in sources), f"the writer kernel is still launched: {sources}"

    _exact(device, shape, DRAM, out_cfg)


@pytest.mark.parametrize("entry", ALIAS_OUT_GEOMETRIES, ids=[e[0] for e in ALIAS_OUT_GEOMETRIES])
def test_alias_out_two_kernel_program_survives_repeat_launches(device, entry):
    """The cross-launch claim: with nothing popping the output CB, the un-popped
    pages of launch N must not block launch N+1.

    Compute pushes exactly the CB's page count and never pops, so `pages_received`
    ends the launch at `shard_tiles` and `pages_acked` at 0. That is only safe
    because the firmware re-runs `setup_local_cb_read_write_interfaces`, which sets
    `tiles_acked_received_init = 0` (`circular_buffer_init.h`), on EVERY launch. If
    it did not, the second launch would hang in `cb_reserve_back` -- so the
    observable of a regression here is a TIMEOUT, and this test has to stay.
    """
    _, shape, out_cfg = entry
    _exact(device, shape, DRAM, out_cfg, repeats=4)


def test_alias_out_keeps_program_cache_rebinding_without_the_writer(device):
    """The runtime arg that used to carry the output base address lived on the
    writer. With the writer gone it rides on the compute kernel; this is the probe
    that says the re-binding still happens (two calls, different shard addresses,
    one cache entry, both bit-exact).
    """
    shape = (1, 1, 256, 128)
    out_cfg = _shard(_BLOCK, _crs(1, 1), (128, 64), _ROW)
    torch_input, tt_input = _make(device, shape)

    first = tilize(tt_input, out_cfg)
    entries = device.num_program_cache_entries()
    second = tilize(tt_input, out_cfg)
    assert device.num_program_cache_entries() == entries, "second call must hit the cache"
    assert first.buffer_address() != second.buffer_address(), "the two calls must land in different shards"
    for out in (first, second):
        assert torch.equal(ttnn.to_torch(out).float(), torch_input.float())


def test_the_dropped_writer_still_emits_a_base_address_runtime_arg(device):
    """...and the arg itself, asserted directly rather than only through the cache
    probe above -- the failure mode is silent (the second call writes into the
    first call's shard), so both witnesses are worth having."""
    shape = (1, 1, 256, 128)
    out_cfg = _shard(_BLOCK, _crs(1, 1), (128, 64), _ROW)
    plan = _plan(device, shape, DRAM, out_cfg)
    _, tt_input = _make(device, shape)
    tt_output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, out_cfg
    )
    descriptor = create_program_descriptor(tt_input, tt_output, plan)
    compute = [k for k in descriptor.kernels if "tilize_compute" in k.kernel_source][0]
    core = plan["work"][0]["core"]
    args = list(compute.runtime_args[core.x][core.y])
    assert len(args) == 2, f"compute should carry num_blocks + the output base address, got {args}"
    assert args[1] == tt_output.buffer_address()


def test_the_writer_survives_everywhere_it_is_still_needed(device):
    """Lever 2 is `alias_out`-only. `alias_in` and the generic path still need the
    writer -- `alias_in` because its OUTPUT CB is a plain CB the writer drains, the
    generic path because it writes.

    Path B was in this list until Refinement 4, which drops BOTH dataflow kernels
    there (its output CB is aliased too, so the writer was as idle as `alias_out`'s).
    That case now lives in `test_tilize_refinement4.py`, asserting the opposite --
    which is the honest bookkeeping: this test's claim about Path B was true of the
    program Refinement 3b shipped, and Refinement 4 changed the program, not the test.
    """
    cases = [
        ("generic", (1, 1, 128, 128), DRAM, DRAM),
        ("alias_in", (1, 1, 256, 128), _shard(_BLOCK, _crs(1, 1), (128, 64), _ROW), DRAM),
    ]
    for name, shape, in_cfg, out_cfg in cases:
        plan = _plan(device, shape, in_cfg, out_cfg)
        assert plan["path"] == name, f"{name}: got path {plan['path']}"
        assert plan["drop_writer"] == 0, f"{name}: the writer must survive"
        _, tt_input = _make(device, shape, in_cfg)
        tt_output = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, out_cfg
        )
        descriptor = create_program_descriptor(tt_input, tt_output, plan)
        assert len(descriptor.kernels) == 3, f"{name}: expected 3 kernels, got {len(descriptor.kernels)}"


def test_the_writer_survives_when_c7_needs_it_as_the_second_reader(device):
    """Lever C7 turns the writer into the second READ issuer on `alias_out`, so the
    two levers are exclusive. C7 is refuted on this path (+11.6 %), but its
    counterfactual bench rows still run it -- and dropping the kernel it depends on
    would turn a measured regression into a HANG (BRISC never signals `sem_done`).
    """
    shape = (1, 1, 128, 64)
    out_cfg = _shard(_HEIGHT, _crs(3, 0), (32, 64), _ROW)
    with _levers(c7=2):
        plan = _plan(device, shape, DRAM, out_cfg, use_double_buffer=False)
        if not plan["split_read"]:
            pytest.skip("C7 is not expressible on this geometry")
        assert plan["drop_writer"] == 0, "C7 uses the writer as the second read issuer"
        _exact(device, shape, DRAM, out_cfg, use_double_buffer=False)


def test_drop_writer_counterfactual_is_bit_exact_too(device):
    """`TILIZE_LEVER_NW=0` is the 3-kernel program of Refinement 3, and it is what
    the ledger's counterfactual row measures -- so it has to keep working."""
    shape = (1, 1, 256, 128)
    out_cfg = _shard(_BLOCK, _crs(1, 1), (128, 64), _ROW)
    with _levers(nw=0):
        plan = _plan(device, shape, DRAM, out_cfg)
        assert plan["drop_writer"] == 0
        _exact(device, shape, DRAM, out_cfg, repeats=2)


def test_drop_writer_gate_is_declared_on(device):
    """The gate itself, so a future edit that turns it off has to say so here."""
    assert drop_writer_pays() is True


# ---------------------------------------------------------------------------
# Lever 3 — the hoisted bank table (REFUTED, but it must stay correct)
# ---------------------------------------------------------------------------


def test_bank_table_gate_is_identity_false(device):
    """Refuted by measurement: +0.9 % against B13 on the regime it was designed for
    and 0.4-1.2 % on the other two. `bank_table_pays` is identity-false so no plan
    ships it; the force flag keeps the counterfactual re-measurable.
    """
    assert bank_table_pays(8, 128) is False
    assert bank_table_pays(64, 64) is False
    assert bank_table_pays(1, 32) is False


BANK_TABLE_SHAPES = [
    # (shape, out_cfg) -- every geometry whose STRUCTURAL clauses the lever meets.
    ((1, 1, 256, 128), _shard(_BLOCK, _crs(1, 1), (128, 64), _ROW)),  # alias_out, 4 blocks
    ((1, 1, 512, 64), _shard(_HEIGHT, _crs(3, 0), (128, 64), _ROW)),  # alias_out, 4 blocks
    ((1, 1, 2048, 32), DRAM),  # generic, tall-narrow
    ((1, 1, 512, 128), DRAM),  # generic, 2 chunks
]


@pytest.mark.parametrize("shape,out_cfg", BANK_TABLE_SHAPES, ids=[f"{s[2]}x{s[3]}" for s, _ in BANK_TABLE_SHAPES])
def test_forced_bank_table_is_bit_exact(device, shape, out_cfg):
    """The affine identity the table rests on -- `addr(first_page + m) ==
    table[m % banks] + (m / banks) * aligned_page` -- is what a wrong hoist breaks,
    and it breaks it SILENTLY: every CB count stays balanced and the block is
    filled with the wrong rows. `arange` input is what makes that visible.

    (The kernel also re-derives the first and last row of every band through the
    accessor under `ASSERT`, so a `--dev` run turns the same bug into an ebreak.)
    """
    with _levers(bt=2, b13=0, b8=0, c7=0, stg=0):
        plan = _plan(device, shape, DRAM, out_cfg)
        if not plan["bank_table"]:
            pytest.skip(f"the lever's structural clauses exclude this plan (path={plan['path']})")
        assert plan["stateful_read"] == 0, "the hoisted table supersedes B13; they must never both be on"
        _exact(device, shape, DRAM, out_cfg)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_forced_bank_table_is_bit_exact_per_dtype(device, dtype):
    """The table strides by the ALIGNED page size, so the dtype (which sets the page
    size) is the axis most likely to expose a stride bug."""
    shape = (1, 1, 256, 128)
    out_cfg = _shard(_BLOCK, _crs(1, 1), (128, 64), _ROW)
    with _levers(bt=2, b13=0, b8=0, c7=0, stg=0):
        plan = _plan(device, shape, DRAM, out_cfg, dtype=dtype)
        if not plan["bank_table"]:
            pytest.skip("the lever's structural clauses exclude this plan")
        _exact(device, shape, DRAM, out_cfg, dtype=dtype)


def test_bank_table_declines_on_a_sharded_source(device):
    """The affine page->bank map only exists on an INTERLEAVED source; a sharded one
    has none, and the helper carries a `static_assert` saying so. The host clause is
    what keeps that assert from ever firing."""
    shape = (1, 1, 256, 128)
    in_cfg = _shard(_BLOCK, _crs(1, 1), (128, 64), _ROW)
    with _levers(bt=2, b13=0, r3=0):
        plan = _plan(device, shape, in_cfg, DRAM)
        assert plan["bank_table"] == 0, "a sharded source has no interleaved bank map"
        _exact(device, shape, in_cfg, DRAM)


def test_bank_table_is_exclusive_with_every_lever_that_owns_the_read_loop(device):
    """All of them reshape the same 32 stick reads, so exactly one may own them. The
    kernel carries the matching `static_assert`; if the host ever emitted two, the
    build would fail rather than silently drop one -- this is the host-side witness.
    """
    shape = (1, 1, 4096, 64)
    for other in ("b8", "c7", "b13"):
        with _levers(bt=2, **{other: 2}):
            plan = _plan(device, shape, DRAM, DRAM)
            owners = [
                plan["bank_table"],
                plan["stateful_read"],
                plan["split_read"],
                1 if plan["prefetch_blocks"] == 2 else 0,
                plan["coalesce_rows"],
                1 if plan["read_group"] > 1 else 0,
            ]
            assert sum(owners) <= 1, f"{other}: two levers own the read loop -> {owners}"


def test_bank_table_min_blocks_is_declared(device):
    """The clause the gate would use if it were ever re-enabled -- one band per core
    means the hoist replaces N calls with N calls plus a table."""
    assert BANK_TABLE_MIN_BLOCKS == 2


# ---------------------------------------------------------------------------
# Lever 1 — the per-RISC timeline variant
# ---------------------------------------------------------------------------


def test_zones_are_off_by_default(device):
    """It is a measurement variant, not a plan: no shipped program may carry it."""
    assert tpd._zone_flag() == 0
    shape = (1, 1, 256, 128)
    out_cfg = _shard(_BLOCK, _crs(1, 1), (128, 64), _ROW)
    plan = _plan(device, shape, DRAM, out_cfg)
    _, tt_input = _make(device, shape)
    tt_output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, out_cfg
    )
    descriptor = create_program_descriptor(tt_input, tt_output, plan)
    reader = [k for k in descriptor.kernels if "tilize_reader" in k.kernel_source][0]
    assert reader.compile_time_args[28] == 0, "the zone variant leaked into a shipped plan"


@pytest.mark.parametrize(
    "shape,out_cfg",
    [((1, 1, 256, 128), DRAM), ((1, 1, 512, 64), DRAM)],
    ids=["two_chunks", "tall"],
)
def test_zone_variant_is_still_bit_exact(device, shape, out_cfg):
    """Unlike the ablations, the timeline variant changes no read, no CB count and
    no trip count -- it only wraps the stages in zones. So it must produce the same
    bytes, and that is what makes its numbers trustworthy as an attribution of the
    REAL kernel rather than of a different one.
    """
    with _env(TILIZE_ZONES=1), _levers(b8=0, c7=0, stg=0, b13=0, bt=0):
        plan = _plan(device, shape, DRAM, out_cfg)
        assert plan["drop_writer"] == 0, "the zone variant instruments the writer's wait/pop"
        _exact(device, shape, DRAM, out_cfg)


def test_zone_variant_reaches_the_alias_out_crossover(device):
    """The regime the whole refinement is about, with the writer kept so its
    99 %-blocked wait is measurable."""
    shape = (1, 1, 256, 128)
    out_cfg = _shard(_BLOCK, _crs(1, 1), (128, 64), _ROW)
    with _env(TILIZE_ZONES=1), _levers(b8=0, c7=0, stg=0, b13=0, bt=0):
        plan = _plan(device, shape, DRAM, out_cfg)
        assert plan["path"] == "alias_out"
        assert plan["drop_writer"] == 0
        _exact(device, shape, DRAM, out_cfg)


def test_zone_variant_is_exclusive_with_lever_2(device):
    """Dropping the writer would delete the zone that prices it, so the zone flag
    forces the 3-kernel program. Asserted here because the two are decided in
    different places (a lever gate vs an env flag)."""
    shape = (1, 1, 256, 128)
    out_cfg = _shard(_BLOCK, _crs(1, 1), (128, 64), _ROW)
    plan_off = _plan(device, shape, DRAM, out_cfg)
    assert plan_off["drop_writer"] == 1
    with _env(TILIZE_ZONES=1):
        assert _plan(device, shape, DRAM, out_cfg)["drop_writer"] == 0


# ---------------------------------------------------------------------------
# Non-regression: the three interleaved bench plans this refinement must not move
# ---------------------------------------------------------------------------


def test_the_interleaved_plans_are_untouched(device):
    """Levers 2 and 3 are `alias_out`-only and identity-false respectively, so the
    interleaved regimes must plan EXACTLY as Refinement 3 left them."""
    expected = {
        # shape                  path       b13 bt nw c7 b8 grp
        (1, 1, 2048, 2048): ("generic", 0, 0, 0, 0, 1, 1),
        (1, 1, 32, 16384): ("generic", 0, 0, 0, 0, 1, 1),
        # d_tall_narrow ships C7 (1 block/core, 64 B reads), not B13 -- Refinement 1c.
        (1, 1, 2048, 32): ("generic", 0, 0, 0, 1, 1, 1),
    }
    for shape, want in expected.items():
        plan = _plan(device, shape, DRAM, DRAM)
        got = (
            plan["path"],
            plan["stateful_read"],
            plan["bank_table"],
            plan["drop_writer"],
            plan["split_read"],
            plan["prefetch_blocks"],
            plan["read_group"],
        )
        assert got == want, f"{shape}: plan moved -> {got}"
