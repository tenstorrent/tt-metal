# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1 — per-core-overhead gating for the low-work-per-core regimes.

Two levers, both *measured* (see `changelog.md` → "Refinement 1" for the tables
and `probes/probe_009.py` / `probe_010.py` for the sweeps):

* **A0 bandwidth-knee term** — `active == min(grid, total_tiles, A0_KNEE_CORES)`.
  The knee was measured on tilize's own transfer shapes and it is the FULL grid,
  so the term is identity. These tests pin that: they assert the criterion holds
  *and* that the constant still encodes "no clamp", so lowering it (re-introducing
  the `dram_saturation` 16-core cap, which measured **2.4× slower** on
  `d_tall_narrow`) fails here instead of silently regressing the bench.

* **C16 depth-2 default gate** — `use_double_buffer=None` (the new default) asks
  the planner for depth-2 only where it pays: below the DRAM
  bandwidth-saturation knee AND with ≥ 2 chunk-blocks per core. `True` / `False`
  still force depth-2 / depth-1, so the public kwarg keeps both values and their
  documented meaning — only the *default* is gated. Gating the default does **not**
  add an axis value: `SUPPORTED["double_buffer"]` stays `[False, True]`, guarded by
  `test_double_buffer_axis_stays_two_valued`.

Correctness is checked with `torch.equal` (tilize is value-preserving), so every
gated path is proven bit-exact, not merely fast.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from ttnn.operations._op_contract import UnsupportedAxisValue
from ttnn.operations.tilize import SUPPORTED, tilize, validate
from ttnn.operations.tilize.tilize_program_descriptor import (
    A0_KNEE_CORES,
    BANDWIDTH_KNEE_CORES,
    DEPTH1_MAX_BLOCKS_PER_CORE,
    MIN_BLOCKS_FOR_DEPTH2,
    L1_CB_BUDGET_BYTES,
    a0_active_cores,
    build_plan,
    depth2_pays,
)

# A wide (large-per-core-work, DRAM-saturated) and a narrow (low-per-core-work)
# shape — the two the refinement's "record per-core CB bytes at the gated
# default" item calls for.
WIDE = (1, 1, 64, 2048)  # nt_h=2  Wt=64  -> 128 tiles
NARROW = (1, 1, 2048, 32)  # nt_h=64 Wt=1   -> 64 tiles, 1 tile/core


def _plan(device, shape, *, use_multicore=True, use_double_buffer=None, dtype=ttnn.bfloat16):
    torch.manual_seed(0)
    tt_input = ttnn.from_torch(
        torch.randn(shape).bfloat16(),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    return build_plan(
        tt_input,
        tt_output,
        device,
        use_multicore=use_multicore,
        use_double_buffer=use_double_buffer,
    )


def _roundtrip(device, shape, *, use_multicore=True, use_double_buffer=None, memory_config=None):
    torch.manual_seed(0)
    torch_input = torch.randn(shape).bfloat16()
    mem = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem
    )
    tt_output = tilize(
        tt_input,
        memory_config,
        use_multicore=use_multicore,
        use_double_buffer=use_double_buffer,
    )
    assert tt_output.layout == ttnn.TILE_LAYOUT
    return torch_input, ttnn.to_torch(tt_output)


# ---------------------------------------------------------------------------
# A0 — the active-core criterion, with the knee term
# ---------------------------------------------------------------------------


def test_a0_knee_term_is_identity_on_this_op(device):
    """The knee term must not clamp: measured 2.4x SLOWER when it does.

    `probes/probe_009.py` swept d_tall_narrow over forced core caps:
    64c 3 623 ns | 32c 5 186 | 16c 8 580 | 8c 14 780 | 4c 27 950 | 1c 107 561 ns.
    Latency is ~linear in tiles-per-core because the op is read-transaction-rate
    bound (64 B DRAM pages on a W=32 RM input), so the DRAM *bandwidth* knee
    never binds. If you lower A0_KNEE_CORES, re-run that probe first.
    """
    grid = device.compute_with_storage_grid_size()
    grid_cores = grid.x * grid.y
    assert A0_KNEE_CORES >= grid_cores, (
        f"A0_KNEE_CORES={A0_KNEE_CORES} would clamp this {grid_cores}-core grid. "
        "Refinement 1 measured a core cap 2.4x SLOWER on d_tall_narrow — re-run "
        "probes/probe_009.py before changing this."
    )
    # The declared criterion is min(grid, total_tiles, knee) and nothing else.
    for total_tiles in (1, 5, 64, 128, 4096):
        assert a0_active_cores(grid_cores, total_tiles) == min(grid_cores, total_tiles)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(NARROW, id="tall_narrow"),
        pytest.param((1, 1, 32, 4096), id="wide_short"),
        pytest.param(WIDE, id="wide"),
    ],
)
def test_a0_active_cores_matches_criterion(device, shape):
    """Every core with work is launched, and no more — in every regime."""
    plan = _plan(device, shape)
    grid = device.compute_with_storage_grid_size()
    expected = min(grid.x * grid.y, plan["total_tiles"], A0_KNEE_CORES)
    assert plan["ncores"] == expected, (
        f"A0: launched {plan['ncores']} cores, expected {expected} " f"(total_tiles={plan['total_tiles']})"
    )


def test_use_multicore_false_still_means_exactly_one_core(device):
    """The knee term is a G clamp inside the multicore path, never a new mode."""
    plan = _plan(device, WIDE, use_multicore=False)
    assert plan["ncores"] == 1


# ---------------------------------------------------------------------------
# C16 — the gated depth-2 default
# ---------------------------------------------------------------------------


def test_depth2_gate_predicate():
    """Every branch of the gate, pinned to the measurement that set it.

    Numbers are in-run depth1/depth2 ratios (7 rounds, CV <= 1.2 %); > 1 means
    depth-2 is faster and must be kept.
    """
    # 1. nothing to overlap -> depth-1 whatever else is true.
    assert depth2_pays(1, 1) is False
    assert depth2_pays(64, 1) is False  # b_wide_short 0.995, g_dram_to_sharded 0.996
    # 2. below the bandwidth knee -> the core's own NoC issue rate is the bound.
    assert depth2_pays(1, 16) is True  # c_single_core 1.321
    assert depth2_pays(1, 32) is True  # x_wide_short_1core 1.360
    assert depth2_pays(BANDWIDTH_KNEE_CORES - 1, MIN_BLOCKS_FOR_DEPTH2) is True
    # 3. at/above the knee -> depth-1 is free only up to 4 block boundaries.
    assert depth2_pays(64, 4) is False  # a_square 0.998, e_square_bf8b_out 1.005
    assert depth2_pays(BANDWIDTH_KNEE_CORES, 4) is False
    assert depth2_pays(64, 8) is True  # e_square_fp32 1.023, fp32_to_bf16 1.019,
    assert depth2_pays(64, 64) is True  # g_sharded_to_dram 1.028
    assert depth2_pays(64, DEPTH1_MAX_BLOCKS_PER_CORE) is False
    assert depth2_pays(64, DEPTH1_MAX_BLOCKS_PER_CORE + 1) is True


def test_gated_default_picks_depth1_when_dram_saturated(device):
    """Grid-filling regimes: depth-1 by default, halving per-core CB L1."""
    gated = _plan(device, WIDE)
    forced2 = _plan(device, WIDE, use_double_buffer=True)

    assert gated["ncores"] >= BANDWIDTH_KNEE_CORES
    assert gated["depth"] == 1, "a DRAM-saturated regime must default to depth-1"
    assert forced2["depth"] == 2, "use_double_buffer=True must still force depth-2"
    # The whole point of the lever: half the L1 for the same chunk width.
    assert gated["chunk_wt"] == forced2["chunk_wt"]
    assert gated["cb_bytes_per_core"] * 2 == forced2["cb_bytes_per_core"]
    assert gated["cb_bytes_per_core"] <= L1_CB_BUDGET_BYTES


@pytest.mark.parametrize(
    "dtype,shape",
    [
        pytest.param(ttnn.bfloat16, WIDE, id="bf16_wide"),
        pytest.param(ttnn.bfloat16, NARROW, id="bf16_narrow"),
        # fp32 is the case that motivated the chunk pin: a naive depth-1 fallback
        # doubles chunk_wt (8 -> 16, so 1024 -> 2048 B reads) and saves NO L1,
        # which measured a 1.3 % loss on e_square_fp32.
        pytest.param(ttnn.float32, (1, 1, 64, 2048), id="fp32_wide"),
    ],
)
def test_gate_never_changes_the_transaction_shape(device, dtype, shape):
    """The gated plan differs from the ungated one in exactly one way: CB pages.

    Same chunk width (== same reader transaction size), same core count, same
    work split — half the per-core CB L1. That makes non-regression structural.
    """
    gated = _plan(device, shape, dtype=dtype)
    forced2 = _plan(device, shape, use_double_buffer=True, dtype=dtype)
    assert gated["depth"] == 1 and forced2["depth"] == 2
    assert gated["chunk_wt"] == forced2["chunk_wt"], "the gate must not re-tune the chunk width"
    assert gated["chunk_row_bytes"] == forced2["chunk_row_bytes"]
    assert gated["ncores"] == forced2["ncores"]
    assert gated["blocks_per_core"] == forced2["blocks_per_core"]
    assert gated["cb_bytes_per_core"] * 2 == forced2["cb_bytes_per_core"]


def test_gated_default_keeps_depth2_past_four_block_boundaries(device):
    """Clause 3 at the plan level: a grid-filling shape with 8 blocks/core.

    `[1,1,4096,2048]` bf16 gives 64 cores x 8 chunk-blocks. Forcing depth-1 there
    measured a real ~2 % loss on three independent 8-block regimes
    (e_square_fp32 1.023, e_square_fp32_to_bf16 1.019, g_sharded_to_dram 1.028),
    so the gate must NOT take depth-1 here even though DRAM is saturated.
    """
    plan = _plan(device, (1, 1, 4096, 2048))
    assert plan["ncores"] == 64
    assert plan["blocks_per_core"] > DEPTH1_MAX_BLOCKS_PER_CORE
    assert plan["depth"] == 2, "past 4 block boundaries the residual overlap still pays"


def test_gated_default_keeps_depth2_when_latency_bound(device):
    """Single-core with blocks to pipeline: the C16 gate must NOT pick depth-1.

    Refinement 2 note: on this exact regime lever B8 (trid double-issue) also fires
    and takes the shipped depth to 3, because its prefetch needs a third window
    (`x_single_core_b8_off` 30 267 ns vs `c_single_core` 27 304 ns). The property
    this test guards is the C16 one — depth-1 here costs +32 %
    (`x_single_core_depth1` 40 013 ns) — so it asserts the *gate's* answer is 2 and
    the shipped depth is at least that, rather than pinning the literal 2 and
    breaking whenever a deeper-window lever lands on the same cell.
    """
    gated = _plan(device, (1, 1, 512, 512), use_multicore=False)
    assert gated["ncores"] == 1
    assert gated["blocks_per_core"] >= MIN_BLOCKS_FOR_DEPTH2
    assert depth2_pays(gated["ncores"], gated["blocks_per_core"]), "C16 must still want depth-2 here"
    assert gated["depth"] >= 2, "the latency-bound single-core regime must never fall back to depth-1"


def test_gated_default_picks_depth1_for_one_block_per_core(device):
    """1 block/core: depth-2 has nothing to overlap, so it is pure L1 cost."""
    gated = _plan(device, NARROW)
    assert gated["blocks_per_core"] == 1
    assert gated["depth"] == 1
    forced2 = _plan(device, NARROW, use_double_buffer=True)
    assert forced2["depth"] == 2
    assert gated["cb_bytes_per_core"] * 2 == forced2["cb_bytes_per_core"]


@pytest.mark.parametrize("use_double_buffer", [None, True, False], ids=["planner", "forced_d2", "forced_d1"])
@pytest.mark.parametrize("shape", [pytest.param(WIDE, id="wide"), pytest.param(NARROW, id="narrow")])
def test_gated_default_is_bit_exact(device, shape, use_double_buffer):
    """Correctness at the gated default and at both forced depths."""
    expected, actual = _roundtrip(device, shape, use_double_buffer=use_double_buffer)
    assert torch.equal(expected, actual), "tilize must be bit-exact at any CB depth"


def test_gated_default_is_bit_exact_single_core(device):
    """The one regime the gate leaves at depth-2."""
    expected, actual = _roundtrip(device, (1, 1, 128, 256), use_multicore=False)
    assert torch.equal(expected, actual)


def test_gate_is_inert_on_the_zero_copy_path(device):
    """Path B's CB *is* the shard, so depth is structurally 1 at any request."""
    mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            (128, 64),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    for request in (None, True, False):
        expected, actual = _roundtrip(device, (1, 1, 512, 64), use_double_buffer=request, memory_config=mem_config)
        assert torch.equal(expected, actual), f"alias path must be bit-exact (request={request})"


# ---------------------------------------------------------------------------
# Registry contract: gating the DEFAULT must not grow the axis
# ---------------------------------------------------------------------------


def test_double_buffer_axis_stays_two_valued():
    """The C16 gate changes the *default*, not the set of supported depths.

    Refinement 1's first attempt declared a third value ``"auto"`` for
    ``use_double_buffer=None``. That was a real registry defect, and this test is
    its regression guard:

    * ``None`` is a **request shape** — "let the planner choose" — and the planner
      only ever chooses depth-1 or depth-2, both already in this axis. There is no
      third CB depth, so there is no third capability to declare.
    * No golden cell can take it: ``tag_double_buffer`` projects the scenario dict
      onto a bool (the golden matrix is 192 ``True`` / 24 ``False``), so the value
      bought exactly zero coverage.
    * A grown SUPPORTED rectangle reclassifies a perf-only refinement as a
      cartesian expansion, which raises the harness's golden-majority bar from
      50 % to 75 % — unreachable for a refinement that unlocks no cell by
      construction. It cost this refinement its completion gate.

    Keep the axis equal to the golden TARGET ``[False, True]``.
    """
    assert sorted(SUPPORTED["double_buffer"], key=str) == [False, True], (
        f'SUPPORTED["double_buffer"] must stay exactly [False, True], got '
        f"{SUPPORTED['double_buffer']!r}. Gating the DEFAULT is not a new axis "
        "value — see this test's docstring."
    )
    # Belt and braces: no sentinel of any spelling.
    assert not any(isinstance(v, str) for v in SUPPORTED["double_buffer"])


@pytest.mark.parametrize("use_double_buffer", [None, True, False], ids=["planner", "d2", "d1"])
def test_validate_accepts_every_depth_request(device, use_double_buffer):
    torch.manual_seed(0)
    tt_input = ttnn.from_torch(
        torch.randn((1, 1, 32, 64)).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    validate(
        tt_input,
        ttnn.DRAM_MEMORY_CONFIG,
        output_dtype=ttnn.bfloat16,
        use_multicore=True,
        use_double_buffer=use_double_buffer,
    )


def test_validate_gates_every_depth_the_planner_may_pick(device, monkeypatch, expect_error):
    """``None`` is *gated*, not waved through.

    Dropping ``"auto"`` must not turn the delegated request into an unchecked one:
    since the planner may pick either depth, ``validate(None)`` has to hold for
    every depth in that set. Proven by shrinking the axis to ``[False]`` — the
    delegated request must then be refused (the planner could still pick depth-2),
    while the pinned ``False`` request stays legal.
    """
    import ttnn.operations.tilize as tilize_mod

    torch.manual_seed(0)
    tt_input = ttnn.from_torch(
        torch.randn((1, 1, 32, 64)).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def _validate(request):
        validate(
            tt_input,
            ttnn.DRAM_MEMORY_CONFIG,
            output_dtype=ttnn.bfloat16,
            use_multicore=True,
            use_double_buffer=request,
        )

    monkeypatch.setitem(tilize_mod.SUPPORTED, "double_buffer", [False])
    with expect_error(UnsupportedAxisValue, "double_buffer=True not in SUPPORTED"):
        _validate(None)  # the planner could still pick depth-2 -> refuse
    with expect_error(UnsupportedAxisValue, "double_buffer=True not in SUPPORTED"):
        _validate(True)
    _validate(False)  # the one depth still declared


def test_per_core_cb_bytes_at_the_gated_default(device):
    """Record (and bound) the per-core CB L1 the gated default actually asks for.

    This is the refinement's "per-core CB bytes recorded at the gated default for
    at least one wide and one narrow shape" deliverable, machine-checked.
    """
    report = {}
    for name, shape in (("wide", WIDE), ("narrow", NARROW)):
        gated = _plan(device, shape)
        forced2 = _plan(device, shape, use_double_buffer=True)
        report[name] = (gated["cb_bytes_per_core"], forced2["cb_bytes_per_core"])
        assert gated["cb_bytes_per_core"] <= L1_CB_BUDGET_BYTES
        assert gated["cb_bytes_per_core"] < forced2["cb_bytes_per_core"], (
            f"{name}: the gate must save L1 ({gated['cb_bytes_per_core']} vs " f"{forced2['cb_bytes_per_core']} B/core)"
        )
    print("\n  per-core CB bytes (gated default -> forced depth-2):")
    for name, (gated_b, forced_b) in report.items():
        print(f"    {name:<8} {gated_b:>7} B  (was {forced_b:>7} B, saved {forced_b - gated_b} B)")
