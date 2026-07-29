# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 5 — the run-closing `/perf-ceiling-dm` **Mode-D completeness audit**.

This refinement adds no capability and changes no `SUPPORTED` value: its deliverable
is a ledger in `changelog.md` accounting for **every** `master.md` lever the finished
op does not use. A ledger is prose, and prose rots. These tests are what keep it
honest — each one pins a claim the ledger makes so that a later change which
invalidates the claim fails here instead of leaving a confident, wrong retrospective
in the changelog:

1. **Ledger completeness is machine-checked** against `master.md` itself
   (`test_completeness_ledger_*`): every Part-1 example name and every Part-2 lever ID
   must appear in the Refinement-5 ledger, each with one of the four statuses. When
   `master.md` grows a lever, the audit goes stale and this test says so.
2. **A0 is graded per regime with the measured core count** — the audit's central
   table, asserted here for all five discriminating regimes plus the crossover, so it
   is a property of the planner rather than a row someone typed.
3. **The A0 *balance* term** (new this pass) — the ratios the audit measured, pinned
   with the ns they were measured against, including the distinction between the part
   an assignment could recover and the part that is arithmetically irreducible.
4. **The structural `not-applicable` verdicts** (A4 / B11 / B12 / C17) — each stated
   as the disqualifier the audit gives, not as "we didn't need it". These are the
   entries a future run would otherwise have to re-derive.
"""

from __future__ import annotations

import math
import pathlib
import re

import pytest
import torch

import ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize import tilize_program_descriptor as tpd
from ttnn.operations.tilize.tilize_program_descriptor import build_plan

_ROOT = pathlib.Path(__file__).resolve().parents[5]
_MASTER = _ROOT / "ttnn" / "ttnn" / "operations" / "examples" / "master.md"
_CHANGELOG = _ROOT / "ttnn" / "ttnn" / "operations" / "tilize" / "changelog.md"

# The four Mode-D statuses. Every catalog entry in the ledger carries exactly one.
_STATUSES = ("not-applicable", "deferred", "measured-no-payoff", "missed", "applied")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _catalog():
    """(Part-1 example names, Part-2 lever ids) parsed out of `master.md`.

    Parsed rather than hard-coded on purpose: the point of the test is to notice
    when the catalog grows past what the audit walked.
    """
    text = _MASTER.read_text()
    part1, part2 = text.split("# Part 2 —")
    examples = re.findall(r"^##\s+.*?\[`([a-z0-9_]+)`\]\(", part1, re.M)
    levers = re.findall(r"\*\*([A-E]\d{1,2})\.", part2)
    # De-duplicate, keep the catalog's order.
    seen, ordered = set(), []
    for lever in levers:
        if lever not in seen:
            seen.add(lever)
            ordered.append(lever)
    return examples, ordered


def _ledger_section():
    """The Refinement-5 section of `changelog.md` (where the ledger lives)."""
    text = _CHANGELOG.read_text()
    start = text.index("## Refinement 5 —")
    return text[start:]


def _plan(device, shape, *, dtype=ttnn.bfloat16, out_dtype=None, in_cfg=None, out_cfg=None, multicore=True):
    torch_in = torch.zeros(shape, dtype=torch.float32)
    tt_in = ttnn.from_torch(
        torch_in,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=in_cfg or ttnn.DRAM_MEMORY_CONFIG,
    )
    cfg = out_cfg or tt_in.memory_config()
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), out_dtype or dtype, ttnn.TILE_LAYOUT, device, cfg)
    return build_plan(tt_in, tt_out, device, use_multicore=multicore), tt_in, tt_out


def _blocks(plan):
    return [u["row_count"] * u["chunk_count"] for u in plan["work"]]


def _balance(plan):
    """(raw, recoverable) — the same two ratios `_bench_tilize.work_imbalance` reports."""
    blocks = _blocks(plan)
    total, n, mx = sum(blocks), len(blocks), max(blocks)
    return mx / (total / n), mx / math.ceil(total / n)


def _crs(end_x, end_y):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(end_x, end_y))})


def _shard(scheme, grid, shape):
    return ttnn.MemoryConfig(scheme, ttnn.BufferType.L1, ttnn.ShardSpec(grid, shape, ttnn.ShardOrientation.ROW_MAJOR))


# ---------------------------------------------------------------------------
# 1. The ledger covers the whole catalog, with a status for each entry
# ---------------------------------------------------------------------------


def test_completeness_ledger_covers_every_part2_lever():
    """Every one of `master.md` Part 2's lever IDs is accounted for in the ledger.

    This is the gate's own wording ("covering every master.md Part 1 example and
    Part 2 lever") turned into a check. If a future edit adds `B14` to the catalog,
    this fails — which is the correct outcome: the audit no longer covers the
    catalog it claims to have walked.
    """
    _, levers = _catalog()
    assert len(levers) == 24, f"catalog changed: {len(levers)} Part-2 levers, audit walked 24"
    section = _ledger_section()
    missing = [lever for lever in levers if not re.search(rf"\b{lever}\b", section)]
    assert not missing, f"Refinement-5 ledger does not account for: {missing}"


def test_completeness_ledger_covers_every_part1_example():
    """Same for the Part-1 runnable examples (they are levers with a measured number)."""
    examples, _ = _catalog()
    assert len(examples) == 19, f"catalog changed: {len(examples)} Part-1 examples, audit walked 19"
    section = _ledger_section()
    missing = [name for name in examples if name not in section]
    assert not missing, f"Refinement-5 ledger does not account for the examples: {missing}"


def test_every_ledger_row_carries_one_of_the_four_statuses():
    """A row without a status is an opinion, not an audit entry.

    Checks the ledger's own table rows: each must name exactly one of the four
    Mode-D statuses (plus `applied`, for the levers the op does use).
    """
    section = _ledger_section()
    rows = [
        line
        for line in section.splitlines()
        if line.startswith("|")
        and re.search(r"\*\*(?:[A-E]\d{1,2}|`[a-z0-9_]+`)\*\*", line)
        and not line.startswith("| lever")
        and not line.startswith("| example")
    ]
    assert len(rows) >= 40, f"ledger has only {len(rows)} classified rows; the catalog has 43 entries"
    unstatused = [r for r in rows if not any(s in r for s in _STATUSES)]
    assert not unstatused, "ledger rows without a Mode-D status:\n" + "\n".join(unstatused[:5])


# ---------------------------------------------------------------------------
# 2. A0 graded per regime, against the MEASURED core count
# ---------------------------------------------------------------------------

# name -> (shape, kwargs, expected active cores). The five discriminating regimes
# master.md A0 names, plus the two crossover directions. `None` = "min(grid,
# total_tiles)", computed from the plan so the assert is the criterion, not a constant.
_A0_REGIMES = {
    "square": (dict(shape=(1, 1, 2048, 2048)), None),
    "wide_short": (dict(shape=(1, 1, 32, 16384)), None),  # nt_h == 1: MUST be the grid
    "tall_narrow": (dict(shape=(1, 1, 2048, 32)), None),
    "tiny": (dict(shape=(1, 1, 64, 64)), 4),  # total_tiles = 4 < grid
    "single_core": (dict(shape=(1, 1, 512, 512), multicore=False), 1),
}


@pytest.mark.parametrize("regime", list(_A0_REGIMES))
def test_a0_active_cores_per_regime(device, regime):
    """A0 is a per-regime correctness check, never a holistic one (master.md §A0).

    The audit grades A0 for every regime the op's INPUTS universe reaches; this is
    that grade as a test. `wide_short` is the load-bearing one: `nt_h == 1`, so a
    height-only split would strand it on ONE core while its duration column still
    looked plausible.
    """
    kwargs, expected = _A0_REGIMES[regime]
    plan, _, _ = _plan(device, **kwargs)
    grid = device.compute_with_storage_grid_size()
    want = expected if expected is not None else min(grid.x * grid.y, plan["total_tiles"], tpd.A0_KNEE_CORES)
    assert plan["ncores"] == want, (
        f"A0 violation in the {regime} regime: {plan['ncores']} cores, expected {want} "
        f"(total_tiles={plan['total_tiles']})"
    )


def test_a0_sharded_in_regime_uses_the_shards_own_cores(device):
    """A0's sharded clause: `active == the shard's own cores`, not a re-spread grid.

    Asserted non-tautologically — the shards must tile the tensor exactly, which is
    what makes "the shard's cores" a complete cover rather than a coincidence.
    """
    for scheme, grid, shard, shape in [
        (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, _crs(3, 0), (128, 64), (1, 1, 512, 64)),
        (ttnn.TensorMemoryLayout.BLOCK_SHARDED, _crs(1, 3), (128, 64), (1, 1, 512, 128)),
    ]:
        cfg = _shard(scheme, grid, shard)
        plan, _, _ = _plan(device, shape, in_cfg=cfg, out_cfg=cfg)
        shard_cores = grid.num_cores()
        assert plan["ncores"] == shard_cores, f"{scheme}: {plan['ncores']} cores vs {shard_cores} shards"
        assert plan["shard_tiles"] * plan["ncores"] == plan["total_tiles"]


def test_a0_crossover_regimes_follow_the_sharded_side(device):
    """Both crossover directions launch on the sharded side's cores (A2's clause)."""
    shape = (1, 1, 512, 128)
    cfg = _shard(ttnn.TensorMemoryLayout.BLOCK_SHARDED, _crs(1, 3), (128, 64))
    for in_cfg, out_cfg, want_path in [(None, cfg, "alias_out"), (cfg, ttnn.DRAM_MEMORY_CONFIG, "alias_in")]:
        plan, _, _ = _plan(device, shape, in_cfg=in_cfg, out_cfg=out_cfg)
        assert plan["path"] == want_path, f"expected {want_path}, got {plan['path']}"
        assert plan["ncores"] == 8, f"{want_path}: {plan['ncores']} cores vs the 8 shards"


# ---------------------------------------------------------------------------
# 3. The A0 *balance* term — the finding this pass added, with its measurement
# ---------------------------------------------------------------------------


def test_evenly_divisible_shapes_are_perfectly_balanced(device):
    """Every shape in the pre-Refinement-5 bench set has balance 1.000.

    Which is exactly why the balance term went unmeasured for five refinements: the
    instrument reported "perfect" on every shape anyone looked at.
    """
    for shape in [(1, 1, 2048, 2048), (1, 1, 32, 16384), (1, 1, 2048, 32), (1, 1, 2048, 1024), (1, 1, 2048, 2016)]:
        plan, _, _ = _plan(device, shape)
        raw, recoverable = _balance(plan)
        assert raw == pytest.approx(1.0), f"{shape}: raw imbalance {raw:.3f}"
        assert recoverable == pytest.approx(1.0), f"{shape}: recoverable imbalance {recoverable:.3f}"


# shape -> (raw, recoverable, measured ns, its even reference's ns, achieved GB/s pair)
# Measured this pass, 7 rounds x 10 launches, CV <= 0.6 % (see changelog Refinement 5).
_IMBALANCE_MEASURED = {
    (1, 1, 2080, 2048): (1.97, 1.60, 86479, 85732, (197.0, 195.7)),
    (1, 1, 3072, 2048): (1.33, 1.33, 127751, 85732, (197.0, 195.7)),
    (1, 1, 2080, 1024): (1.97, 1.33, 44576, 44313, (191.1, 189.3)),
    (1, 1, 2080, 32): (1.97, 1.00, 4161, 3441, (64.0, 76.2)),
}


@pytest.mark.parametrize("shape", list(_IMBALANCE_MEASURED))
def test_indivisible_height_balance_ratios_are_the_measured_ones(device, shape):
    """Pin the two ratios per shape, so a distribution change has to re-measure.

    The row split is `_split_contiguous(nt_h, n_h)`: at `nt_h = 64k+1` one core gets
    an extra tile-row, and because a core's work is `row_count * chunk_count` that
    extra row multiplies its block count. The ledger's verdict rests on these exact
    numbers — the raw ratio (vs a fractional ideal) AND the recoverable one (vs the
    best any whole-block assignment could do), because only the second is headroom.
    """
    raw_want, rec_want, *_ = _IMBALANCE_MEASURED[shape]
    plan, _, _ = _plan(device, shape)
    raw, rec = _balance(plan)
    assert raw == pytest.approx(raw_want, abs=0.02), f"{shape}: raw {raw:.3f} != {raw_want}"
    assert rec == pytest.approx(rec_want, abs=0.02), f"{shape}: recoverable {rec:.3f} != {rec_want}"


def test_recoverable_imbalance_only_exists_on_dram_bound_shapes(device):
    """The ledger's structural argument, asserted rather than reasoned.

    A recoverable imbalance needs the *assignment unit* to be bigger than one block,
    i.e. `n_w == 1` (a whole tile-row of `chunk_count` blocks goes to one core) AND
    `chunk_count > 1`. Together those force `nt_h >= ncores` and `Wt > chunk_wt`,
    i.e. a tensor large enough to be DRAM-bandwidth-bound — which is the regime the
    measurement shows absorbs the tail for free (197.0 vs 195.7 GB/s). So the
    "flatten the work list" lever has no regime where it is both available and
    exposed. If this implication ever breaks, the verdict has to be re-derived.
    """
    for shape in [(1, 1, 2080, 2048), (1, 1, 3072, 2048), (1, 1, 2080, 1024), (1, 1, 2080, 32), (1, 1, 1056, 32)]:
        plan, _, _ = _plan(device, shape)
        _, rec = _balance(plan)
        if rec > 1.001:
            chunks = {u["chunk_count"] for u in plan["work"]}
            assert chunks == {plan["wt"] // plan["chunk_wt"]}, f"{shape}: expected n_w == 1, chunk_counts={chunks}"
            assert max(chunks) > 1, f"{shape}: recoverable imbalance with chunk_count == 1 is impossible"
            assert plan["nt_h"] >= plan["ncores"], f"{shape}: nt_h={plan['nt_h']} < ncores={plan['ncores']}"


def test_irreducible_imbalance_is_not_counted_as_headroom(device):
    """`[1,1,2080,32]` costs +21 % of ns — and no assignment can avoid it.

    65 indivisible blocks over 64 cores forces `max == 2` whatever the mapping, so
    its raw 1.97 is NOT a lever. This is the distinction the `rec` column exists to
    make; asserting it stops a future reader treating the raw ratio as a prize.
    """
    plan, _, _ = _plan(device, (1, 1, 2080, 32))
    blocks = _blocks(plan)
    assert sum(blocks) == 65 and len(blocks) == 64
    assert max(blocks) == math.ceil(sum(blocks) / len(blocks)) == 2
    _, rec = _balance(plan)
    assert rec == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. The structural `not-applicable` verdicts, each with its disqualifier
# ---------------------------------------------------------------------------


def test_a4_no_cliff_kernel_exists_to_specialize(device):
    """A4 not-applicable: the program has ONE kernel per role over all cores.

    A4 splits a grid into full cores + a remainder core with its own (cheaper)
    kernel. Here the remainder rides in per-core runtime args — every core runs the
    same binary — so there is no cliff kernel to skip. Asserted on a deliberately
    indivisible shape, i.e. exactly the case A4 exists for.
    """
    plan, tt_in, tt_out = _plan(device, (1, 1, 2080, 2048))
    desc = tpd.create_program_descriptor(tt_in, tt_out, plan)
    assert len(desc.kernels) == 3, f"expected reader/compute/writer only, got {len(desc.kernels)}"
    for kernel in desc.kernels:
        cores = sum(r.grid_size().x * r.grid_size().y for r in kernel.core_ranges.ranges())
        assert cores == plan["ncores"], f"{kernel.kernel_source}: {cores} cores vs plan {plan['ncores']}"
        assert len(kernel.runtime_args) == plan["ncores"], "the remainder must ride in per-core runtime args"


@pytest.mark.parametrize(
    "dtype,shape",
    [
        (ttnn.bfloat16, (1, 1, 2048, 32)),
        (ttnn.bfloat16, (1, 1, 32, 16384)),
        (ttnn.float32, (1, 1, 2048, 2048)),
        (ttnn.uint16, (1, 1, 64, 160)),
        (ttnn.int32, (1, 1, 64, 224)),
    ],
)
def test_b11_every_transfer_offset_is_noc_aligned(device, dtype, shape):
    """B11 applied-automatically, asserted instead of assumed.

    DRAM wants 32 B read / 16 B write alignment. A read is `chunk_wt * 32 *
    elem_bytes` at an offset that is a whole number of those, and the smallest
    possible value is `1 * 32 * 2 = 64 B` — so alignment can only be violated by a
    change that makes the row-chunk sub-tile-wide. Writes are whole tile pages.
    """
    plan, _, _ = _plan(device, shape, dtype=dtype)
    read_bytes = plan["chunk_row_bytes"]
    assert read_bytes % 32 == 0 and read_bytes >= 32, f"read transaction {read_bytes} B is not 32 B-aligned"
    assert read_bytes == plan["chunk_wt"] * 32 * plan["elem_in"]
    assert plan["source_page_bytes"] % 32 == 0, f"source page {plan['source_page_bytes']} B not 32 B-aligned"
    for unit in plan["work"]:
        offset = unit["chunk_start"] * read_bytes
        assert offset % 32 == 0, f"read offset {offset} B is not 32 B-aligned"
    # The writer moves whole TILE pages; `ttnn.tile_size` is the page size the
    # descriptor gives the output CB and the DRAM write.
    write_bytes = plan["tile_out"]
    assert write_bytes % 16 == 0, f"write page {write_bytes} B is not 16 B-aligned"


def test_b12_multicast_cannot_apply_even_in_the_shared_read_regime(device):
    """B12 not-applicable, with the disqualifier the ledger states.

    `nt_h == 1` is the ONE regime where the 64 readers share source pages (each reads
    a slice of the same 32 rows), which looks like multicast's use case. It is not:
    multicast delivers the *same* bytes to every receiver, and here every core needs a
    **disjoint** slice. Broadcasting the whole page instead would move `Wt/chunk_wt`x
    the L1 bytes. Asserted as pairwise-disjoint chunk ranges.
    """
    plan, _, _ = _plan(device, (1, 1, 32, 16384))
    assert plan["nt_h"] == 1 and plan["ncores"] > 1
    spans = [(u["chunk_start"], u["chunk_start"] + u["chunk_count"]) for u in plan["work"]]
    assert len(set(spans)) == len(spans), "chunk ranges are not distinct — mcast might apply"
    for i, (a0, a1) in enumerate(spans):
        for b0, b1 in spans[i + 1 :]:
            assert a1 <= b0 or b1 <= a0, f"overlapping chunk ranges {(a0, a1)} and {(b0, b1)}"
    # The counterfactual the ledger prices: a whole-page broadcast would hand every
    # receiver the whole tile-row, i.e. this many times the bytes it owns.
    assert plan["wt"] // plan["chunk_wt"] == plan["ncores"]


def test_c17_in_place_is_structurally_impossible(device):
    """C17 not-applicable: RM->TILE always moves every byte, so there is no
    no-copy case to detect. The op allocates a distinct output buffer, and the
    compute helper `static_assert`s the two CB formats differ."""
    torch_in = torch.arange(32 * 64, dtype=torch.float32).reshape(1, 1, 32, 64).to(torch.bfloat16)
    tt_in = ttnn.from_torch(torch_in, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_out = tilize(tt_in)
    assert tt_out.buffer_address() != tt_in.buffer_address()
    assert tt_out.layout == ttnn.TILE_LAYOUT and tt_in.layout == ttnn.ROW_MAJOR_LAYOUT
    # Values are untouched (the identity oracle) — only their byte positions moved.
    assert torch.equal(ttnn.to_torch(tt_out), torch_in)


def test_c15_residency_delta_is_the_callers_lever_not_the_ops(device):
    """C15 not-applicable-to-the-op, quantified anyway (the ledger's duty).

    "Prefer sharded over interleaved" is the *caller's* memory config; the op honors
    whatever it is given. But the size of the prize is worth recording, and it is
    measurable from the plan: with both sides L1-sharded the op moves ZERO DRAM
    bytes, versus 2x the tensor for DRAM->DRAM.
    """
    shape = (1, 1, 512, 64)
    cfg = _shard(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, _crs(3, 0), (128, 64))
    sharded, _, _ = _plan(device, shape, in_cfg=cfg, out_cfg=cfg)
    interleaved, _, _ = _plan(device, shape)
    assert sharded["path"] == "alias" and interleaved["path"] == "generic"
    # Path B is a compute-only program: no NoC-capable kernel exists at all, which is
    # a stronger statement than "it issues no transfers".
    assert sharded["drop_reader"] == sharded["drop_writer"] == 1
    assert interleaved["drop_reader"] == interleaved["drop_writer"] == 0


# ---------------------------------------------------------------------------
# 5. Lever A1's counterfactual: the sweep hook, and what it is allowed to change
# ---------------------------------------------------------------------------


def test_a1_hook_is_off_in_production_and_the_shipped_order_is_row_major(device):
    """The op ships A1's row-major line, and the audit hook is inert by default.

    Phase 0 recorded A1 (`row_wise=True`) as KEEP from `noc_placement`'s design
    reference without a device counterfactual on this op; Refinement 5 measured it
    (+2.1 % at 4 cores, +11.5 % at 8, +3.9 % on tall-narrow at 64, ~0 on the
    DRAM-saturated 64-core regimes). That measurement needs a hook, and a hook that
    could be left on is a footgun — so assert both halves: default off, and the
    shipped order is the one that was measured as the fast one.
    """
    assert tpd.CORE_ORDER_OVERRIDE is None, "the A1 sweep hook is set at import — never in production"
    grid = device.compute_with_storage_grid_size()
    for shape in [(1, 1, 64, 64), (1, 1, 256, 32), (1, 1, 2048, 32), (1, 1, 2048, 2048)]:
        plan, _, _ = _plan(device, shape)
        want = ttnn.grid_to_cores(plan["ncores"], grid.x, grid.y, True)
        got = [(int(c.x), int(c.y)) for c in plan["cores"]]
        assert got == [(int(c.x), int(c.y)) for c in want], f"{shape}: not A1's row-major order"


def test_a1_counterfactual_changes_only_the_placement(device):
    """The A/B is attributable: the two arms differ from the shipped plan in the
    core list / range set and in NOTHING else.

    Same discipline as Refinement 1's "the gate never changes the transaction shape":
    a counterfactual that also moved `chunk_wt` or the depth would measure two things
    at once and the ledger row would be meaningless.
    """
    shape = (1, 1, 256, 32)
    shipped, _, _ = _plan(device, shape)
    try:
        for mode in (0, 1):
            tpd.CORE_ORDER_OVERRIDE = mode
            arm, _, _ = _plan(device, shape)
            allowed = {"cores", "core_ranges", "work"}
            for key, value in shipped.items():
                if key in allowed or key in ("in_dtype", "out_dtype"):
                    continue
                assert arm[key] == value, f"arm {mode} also changed {key}: {value} -> {arm[key]}"
            # `work` may only differ in which core each unit landed on.
            assert [(u["row_start"], u["row_count"], u["chunk_start"], u["chunk_count"]) for u in arm["work"]] == [
                (u["row_start"], u["row_count"], u["chunk_start"], u["chunk_count"]) for u in shipped["work"]
            ], f"arm {mode} changed the work decomposition, not just its placement"
            if mode == 1:  # the fragmentation control keeps the placement itself
                assert [(int(c.x), int(c.y)) for c in arm["cores"]] == [(int(c.x), int(c.y)) for c in shipped["cores"]]
            else:
                assert [(int(c.x), int(c.y)) for c in arm["cores"]] != [
                    (int(c.x), int(c.y)) for c in shipped["cores"]
                ], "the column-major arm must actually move the cores"
    finally:
        tpd.CORE_ORDER_OVERRIDE = None


# ---------------------------------------------------------------------------
# 6. The audit's bench-coverage claim
# ---------------------------------------------------------------------------


def test_bench_covers_every_a0_discriminating_regime():
    """The bench set must keep a row for each regime the audit grades.

    Mode D's A0 clause is graded per regime "against measured core counts", so the
    bench has to *have* a row per regime. Four existed; `tiny` and the
    indivisible-height family were added this pass. Dropping one would silently
    narrow the next run's non-regression gate.
    """
    bench = (_ROOT / "tests" / "ttnn" / "unit_tests" / "operations" / "tilize" / "_bench_tilize.py").read_text()
    for regime in (
        "a_square",  # square
        "b_wide_short",  # wide-short (nt_h == 1)
        "d_tall_narrow",  # tall-narrow
        "n_tiny_interleaved",  # tiny (total_tiles < grid)
        "f_sharded_small",  # sharded-in
        "g_dram_to_sharded",  # crossover
        "n_imbal_square_65row",  # indivisible height (recoverable 1.60)
        "n_imbal_tall_narrow_65",  # indivisible height (irreducible)
        "x_eighth_grid_colwise",  # lever A1's counterfactual
        "x_eighth_grid_frag_ctl",  # ...and its fragmentation control
    ):
        assert f'"{regime}"' in bench, f"the bench lost the {regime} regime"
    assert "def work_imbalance" in bench, "the bench lost the A0 balance metric"
