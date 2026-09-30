# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for the emitter in ``helpers/ulp_sweep.py``.

No device: ``write_table`` is a line-oriented rewrite of a YAML file, and everything
worth pinning is about what it must *not* touch. The table's rows are contracts, so a
regeneration that quietly drops one weakens a gate with nothing to notice.
"""

import math
import os
import re

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat
from helpers.llk_params import ApproximationMode, DestAccumulation, MathOperation
from helpers.ulp import has_ulp_gate
from helpers.ulp_sweep import (
    EMIT_HEADROOM,
    MEASURED,
    _known_lanes,
    export_measured,
    finish_emit,
    flushed_inputs,
    measurable_mask,
    merge_measured,
    nonfinite_failures,
    record,
    stale_excuses,
    write_table,
)

#: An op defined everywhere, so the non-finite tests below turn on the exclusion they
#: mean to test rather than on a domain.
_OP = MathOperation.Abs

#: One op block with a row of each kind write_table has to tell apart.
_TABLE = """Gelu:  # header provenance, ungeneratable
  - {in: Float16, out: Float16, max_ulp: 7}  # superseded
  - {in: Float16_b, out: Float32, max_ulp: 9}  # fp32 output, not swept
  - {in: Float16_b, out: Float16_b, arch: BLACKHOLE, max_ulp: 44}  # another arch

Log1p:
  - {in: Float16, out: Float16_b, metric: tolerance}  # hand-authored non-enrolment
"""


def _refuses(match, kind=ValueError):
    """The suite's ``expect_error`` fixture needs a device; these are host-only tests."""
    return pytest.raises(kind, match=match)  # allow-pytest.raises: host-only test


@pytest.fixture(autouse=True)
def _host_only_arch(monkeypatch):
    """``sweep_cells`` asks the chip which cells promote to a 32-bit Dest, and without
    ``CHIP_ARCH`` that opens a device. These tests run on hosts without one."""
    import helpers.chip_architecture as chip

    monkeypatch.setenv("CHIP_ARCH", os.environ.get("CHIP_ARCH", "wormhole"))
    monkeypatch.setattr(chip, "_cached_chip_architecture", None)


@pytest.fixture
def table(tmp_path):
    path = tmp_path / "budget.yaml"
    path.write_text(_TABLE, encoding="utf-8")
    MEASURED.clear()
    yield path
    MEASURED.clear()


def _rows(path):
    return [
        l.strip() for l in path.read_text().splitlines() if l.strip().startswith("- ")
    ]


def test_only_the_cells_this_run_measured_are_replaced(table):
    """`MEASURED`, not the static SWEEP_FORMATS cross-product. A `-k` run, an interrupt
    or a driver skip must leave every cell it did not measure alone rather than render a
    whole op block from a partial session."""
    record("Gelu", ("Float16", "Float16", "No", "No"), 5)
    assert write_table(table, "today") == (1, [])

    rows = _rows(table)
    assert "{in: Float16, out: Float16, max_ulp: 6}" in rows[0]  # 5 * 1.1, rounded up
    assert "max 5 ULP, today" in rows[0]
    assert not any("max_ulp: 7" in row for row in rows)  # the superseded cell is gone
    # The same op's other (in, out) cell, and the op this run never measured: untouched.
    assert any("{in: Float16_b, out: Float32, max_ulp: 9}" in row for row in rows)
    assert "{in: Float16, out: Float16_b, metric: tolerance}" in rows[-1]


def test_a_row_for_another_architecture_survives_a_regeneration(table):
    """The sweep runs on one arch and the table's header says to re-measure on the
    other. Specificity lets the two rows coexist, so an arch-keyed row is never
    replaceable — even though its `in`/`out` are both swept formats."""
    record("Gelu", ("Float16_b", "Float16_b", "No", "No"), 3)
    write_table(table, "today")
    assert any("arch: BLACKHOLE, max_ulp: 44" in row for row in _rows(table))


def test_a_row_carrying_a_floor_is_kept_rather_than_regenerated(table):
    """`_render` emits only `max_ulp` or `metric: tolerance`, so a `near_zero_atol`
    floor cannot be put back — and it cannot ride along on a demotion either, since
    `AccuracyContract` refuses the field outside the ULP metric. Emitting over it would
    drop it silently; keeping it as well would give the cell two equally specific keys.
    """
    table.write_text(
        "Gelu:\n"
        "  - {in: Float16, out: Float16, max_ulp: 2, near_zero_atol: 5.59e-07}  # floor\n",
        encoding="utf-8",
    )
    record("Gelu", ("Float16", "Float16", "No", "No"), 5)
    assert write_table(table, "today") == (0, ["Gelu"])
    assert "near_zero_atol: 5.59e-07}  # floor" in table.read_text()  # kept verbatim


@pytest.mark.parametrize(
    "wildcard",
    [
        "{metric: tolerance, atol: 0.13, rtol: 0.05}",
        "&lut {metric: tolerance, atol: 0.13, rtol: 0.05}",
        "*lut",
    ],
    ids=["inline", "anchor", "alias"],
)
def test_an_op_wide_declared_tolerance_is_not_shadowed_by_a_rendered_row(
    table, wildcard
):
    """SigmoidAppx's op-wide `atol: 0.13` pins no `(in, out)`, so it is not "covered" by
    any one measured cell -- but every rendered row is more specific than it, and a bare
    `metric: tolerance` there resolves to `atol=None`: the driver's 0.05, not the 0.13
    the table says the LUT needs. Such a row answers for every measured cell, so the op
    is kept verbatim like a floor."""
    anchor = (
        ""
        if not wildcard.startswith("*")
        else ("Anchor:\n  - &lut {metric: tolerance, atol: 0.13, rtol: 0.05}\n\n")
    )
    table.write_text(f"{anchor}Gelu:\n  - {wildcard}\n", encoding="utf-8")
    before = table.read_text()
    for approx in ("No", "Yes"):
        for dest in ("No", "Yes"):
            record("Gelu", ("Float16_b", "Float16_b", approx, dest), 100)
    assert write_table(table, "today") == (0, ["Gelu"])
    assert table.read_text() == before


def test_a_pinned_declared_tolerance_on_an_unmeasured_cell_does_not_block_the_op(table):
    """The widening above is by what a row *pins*: an `atol` row on a cell this run did
    not measure answers for none of its cells, so the op is still regenerated."""
    table.write_text(
        "Gelu:\n  - {in: Float16, out: Float16, metric: tolerance, atol: 0.2}  # kept\n",
        encoding="utf-8",
    )
    record("Gelu", ("Float16_b", "Float16_b", "No", "No"), 1)
    assert write_table(table, "today") == (1, [])
    assert any("atol: 0.2}  # kept" in row for row in _rows(table))


def test_a_measurement_with_nowhere_to_go_is_refused_after_writing_the_rest(table):
    """The key line is passed through verbatim so a header comment survives, so a new
    op's block has to be hand-authored first. Dropping the measurement in silence is
    what left 17 ops' sampled rows in place looking measured.

    Refused only *after* every op that has a key line is written: a whole-table emit
    measures every sweepable op, and one unenrolled op must not throw the rest away."""
    from helpers.ulp_sweep import UnplacedMeasurements

    record("Sqrt", ("Float16", "Float16", "No", "No"), 1)
    record("Gelu", ("Float16", "Float16", "No", "No"), 5)
    with _refuses("no key line") as caught:
        write_table(table, "today")
    assert isinstance(caught.value, UnplacedMeasurements)
    assert (caught.value.written, caught.value.missing) == (1, ["Sqrt"])
    assert "{in: Float16, out: Float16, max_ulp: 6}" in _rows(table)[0]


def test_a_cell_recorded_twice_keeps_the_worst_lane():
    """The key is four-dimensional and a driver enumerates more — `fast_mode` and
    `input_dimensions` are both multi-valued — so one cell is recorded several times.
    Last-write-wins is the polarity that hides error."""
    MEASURED.clear()
    key = ("Float16", "Float16", "No", "No")
    record("Gelu", key, 12)
    record("Gelu", key, 3)
    assert MEASURED["Gelu"][key] == 12
    MEASURED.clear()


def test_an_incomplete_grid_is_refused_rather_than_collapsed(table):
    """`approx` and `dest` droppability is decided per axis. On an anti-diagonal each
    axis sees only singletons, both would be dropped, and one measurement would
    overwrite the other — with `_render` writing the survivor's own figure into the
    provenance comment, so the budget audit could not catch it either."""
    record("Gelu", ("Float16", "Float16", "No", "No"), 4)
    record("Gelu", ("Float16", "Float16", "Yes", "Yes"), 9)
    with _refuses("do not form a full grid"):
        write_table(table, "today")


def test_the_emitted_budget_uses_the_declared_headroom():
    """The factor was hardcoded beside the constant, so tuning it did nothing."""
    from helpers.ulp_sweep import _verdict

    assert _verdict(100, "Float32") == ("ulp", math.ceil(100 * EMIT_HEADROOM))
    # Zero is exact and stays exact: the sweep saw every value.
    assert _verdict(0, "Float32") == ("ulp", 0)
    # A block float never enrols from a sorted sweep, however small the reading.
    assert _verdict(3, "Bfp8_b") == ("block", 3)
    assert _verdict(0, "Bfp8_b") == ("block", 0)


def test_a_nonfinite_disagreement_is_reported_rather_than_only_masked_out():
    """`measurable_mask` drops it because a step count cannot describe it, and the sweep
    driver ranks a distance rather than calling `passed_test` — so without a separate
    report a hardware overflow leaves the statistics clean and passes."""
    src = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    golden = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    result = torch.tensor([1.0, float("inf"), 3.0], dtype=torch.bfloat16)
    fmt = DataFormat.Float16_b

    measurable = measurable_mask(src, golden, result, fmt)
    assert measurable.tolist() == [True, False, True]
    assert nonfinite_failures(_OP, src, golden, result, fmt, fmt).tolist() == [
        False,
        True,
        False,
    ]


def test_a_flushed_subnormal_input_is_not_a_nonfinite_failure():
    """The unpack path flushes it and the golden does not, so a disagreement there is
    the flush, not the op — the same exclusion `measurable_mask` makes."""
    tiny = float(torch.finfo(torch.bfloat16).smallest_normal) / 2
    src = torch.tensor([tiny], dtype=torch.bfloat16)
    golden = torch.tensor([1.0], dtype=torch.bfloat16)
    result = torch.tensor([float("inf")], dtype=torch.bfloat16)
    assert not nonfinite_failures(
        _OP, src, golden, result, DataFormat.Float16_b, DataFormat.Float16_b
    ).any()


def test_a_finite_answer_to_an_infinite_golden_is_a_nonfinite_failure():
    """Only a *saturated* store is excused past the output range. `exp` past overflow
    returning the largest finite bf16 where the golden is `+inf`, or an infinity of the
    wrong sign, is the kernel being wrong -- and the ranking mask drops the lane too, so
    this is the only place it can fail."""
    fmt = DataFormat.Float16_b
    src = torch.tensor([5.0, 5.0, 5.0], dtype=torch.bfloat16)
    golden = torch.full((3,), float("inf"), dtype=torch.bfloat16)
    result = torch.tensor([3.39e38, float("-inf"), float("inf")], dtype=torch.bfloat16)
    assert nonfinite_failures(_OP, src, golden, result, fmt, fmt).tolist() == [
        True,
        True,
        False,
    ]


def test_an_infinite_golden_on_the_singularity_point_is_not_a_nonfinite_failure():
    """`rsqrt(-0)` answers `+inf` against `-inf` because the unpack drops the sign, and
    a 16-bit fp16 Dest answers `log(0)` with -130560 because it has no infinity. The
    value at a pole is a limit, so only the point itself is excused -- a finite answer
    one step off it, where the golden is also infinite in the output format, fails."""
    fmt = DataFormat.Float16_b
    src = torch.tensor([-0.0, 0.0, 1e-30], dtype=torch.bfloat16)
    golden = torch.tensor(
        [float("-inf"), float("inf"), float("inf")], dtype=torch.bfloat16
    )
    result = torch.tensor([float("inf"), -130560.0, 3.0e38], dtype=torch.bfloat16)
    assert nonfinite_failures(
        MathOperation.Rsqrt, src, golden, result, fmt, fmt
    ).tolist() == [False, False, True]


def test_a_golden_past_the_output_range_is_not_a_nonfinite_failure():
    """A full-range sweep of a bf16 input reaches magnitudes a Float16 output cannot
    hold, and `relu_min` passes most of them straight through -- 14,334 lanes of it.
    Saturating there is the store doing what it must, not the kernel being wrong.

    The input stays small and only the golden is large, so nothing but the output-range
    clause can excuse the lane: the same golden into a Float16_b output fails."""
    src = torch.tensor([5.0], dtype=torch.bfloat16)
    golden = torch.tensor([1e5], dtype=torch.bfloat16)
    result = torch.tensor([float("nan")], dtype=torch.bfloat16)
    fmt = DataFormat.Float16_b
    assert not nonfinite_failures(
        _OP, src, golden, result, fmt, DataFormat.Float16
    ).any()
    assert nonfinite_failures(_OP, src, golden, result, fmt, fmt).any()


def test_the_sweeps_own_zero_padding_is_not_data():
    """`generate_full_tensor` pads to the tile count, so the last 257 bf16 lanes are
    zeros the sweep never chose to feed. They inflate every lane count, and on an op
    singular at zero they land on the pole -- 257 copies of one value nobody chose. Both
    masks drop them by position, for every op, whatever its singularities say.

    By position, not by value: one legitimate `0.0` is swept, in the middle.
    """
    from helpers.stimuli_generator.strategies.structured import ulp_sweep_value_count
    from helpers.ulp_sweep import padding_lanes

    fmt = DataFormat.Float16_b
    swept = ulp_sweep_value_count(fmt, float("-inf"), float("inf"))
    pad_lanes = 257
    src = torch.zeros(swept + pad_lanes, dtype=torch.bfloat16)
    pad = padding_lanes(src, fmt)
    assert int(pad.sum()) == pad_lanes
    assert not bool(pad[:swept].any()) and bool(pad[swept:].all())

    # And it reaches both masks: a padded lane is neither measurable nor a failure.
    golden = torch.full_like(src, 1.0)
    result = torch.full_like(src, float("inf"))
    assert not measurable_mask(src, golden, result, fmt)[swept:].any()
    assert not nonfinite_failures(MathOperation.Abs, src, golden, result, fmt, fmt)[
        swept:
    ].any()


def test_an_input_past_a_claim_limit_is_not_a_nonfinite_failure():
    """`Sin` and `Cos` disagree on ~21,000 bf16 lanes past [-pi, pi] -- an argument
    reduction giving up, not a regression. The budget is still measured everywhere;
    only the non-finite answer needs the op to have been claiming something."""
    fmt = DataFormat.Float16_b
    far = torch.tensor([2.6e28], dtype=torch.bfloat16)
    golden = torch.tensor([-1.0], dtype=torch.bfloat16)
    result = torch.tensor([float("inf")], dtype=torch.bfloat16)
    assert not nonfinite_failures(
        MathOperation.Sin, far, golden, result, fmt, fmt
    ).any()
    # Bfp8_b rides on the bfloat16 value set, so it carries bfloat16's limit.
    assert not nonfinite_failures(
        MathOperation.Sin, far, golden, result, DataFormat.Bfp8_b, fmt
    ).any()
    # Inside [-pi, pi] the same disagreement is a failure.
    assert nonfinite_failures(
        MathOperation.Sin,
        torch.tensor([1.0], dtype=torch.bfloat16),
        golden,
        result,
        fmt,
        fmt,
    ).any()


@pytest.mark.parametrize(
    "op", [MathOperation.Sin, MathOperation.Cos], ids=lambda o: o.name
)
def test_the_claim_limit_follows_what_the_stimuli_format_can_reach(op):
    """`sin(2.6e28)` is a bfloat16 value. float16 ends at 65504, inside the range the
    kernel reduces -- Sin and Cos read 1-4 steps over the whole fp16 format -- so an
    fp16 input carries no limit, and a non-finite answer at its very top is a failure.
    Keyed on the op alone, the pi claim covered the fp16 cells too, which are the only
    Sin/Cos cells the table step-gates."""
    fmt = DataFormat.Float16
    top = torch.tensor([65504.0, 4.0], dtype=torch.float16)
    golden = torch.tensor([-1.0, -1.0], dtype=torch.float16)
    result = torch.tensor([float("inf"), float("inf")], dtype=torch.float16)
    assert nonfinite_failures(op, top, golden, result, fmt, fmt).tolist() == [
        True,
        True,
    ]
    # The same two magnitudes from a bfloat16 input: 4.0 is past pi and excused.
    assert nonfinite_failures(
        op,
        top.to(torch.bfloat16),
        golden.to(torch.bfloat16),
        result.to(torch.bfloat16),
        DataFormat.Float16_b,
        DataFormat.Float16_b,
    ).tolist() == [False, False]


def test_a_block_float_input_the_quantizer_flushes_is_the_flush_not_the_op():
    """The sweep's one ``-0.0`` shares a Bfp8_b block with the bf16 subnormals beside
    it; the shared exponent is 0 and the quantizer's forced hidden bit hands the golden
    ``-2**-127``, so ``floor`` reads -1 against the 0 silicon sees: 16,129 steps, the
    whole of Floor's apparent error on every Bfp8_b-input cell. The input as generated
    is a zero, so only the quantized value shows it -- and only on a block format."""
    tiny = 2.0**-133
    below = [-(k * tiny) for k in range(8, 0, -1)]
    above = [k * tiny for k in range(1, 8)]
    src = torch.tensor(below + [-0.0] + above, dtype=torch.bfloat16)
    zero = len(below)
    assert float(src[zero]) == 0.0

    as_bf16 = flushed_inputs(src, DataFormat.Float16_b)
    as_block = flushed_inputs(src, DataFormat.Bfp8_b)
    assert as_bf16.tolist() == [True] * zero + [False] + [True] * len(above)
    assert as_block.all()

    golden = torch.full_like(src, -1.0)
    result = torch.zeros_like(src)
    assert not measurable_mask(src, golden, result, DataFormat.Bfp8_b)[zero]
    assert measurable_mask(src, golden, result, DataFormat.Float16_b)[zero]
    inf = torch.full_like(src, float("inf"))
    assert not nonfinite_failures(
        MathOperation.Floor, src, golden, inf, DataFormat.Bfp8_b, DataFormat.Float16_b
    )[zero]
    assert nonfinite_failures(
        MathOperation.Floor,
        src,
        golden,
        inf,
        DataFormat.Float16_b,
        DataFormat.Float16_b,
    )[zero]


def test_a_known_nonfinite_lane_is_excused_on_its_cell_and_nowhere_else():
    """Celu returns inf for x in 65408..65504 on a 16-bit Float16 Dest (#58607). The
    entry names those inputs on that cell. The lane below the interval is still a
    failure, so is the same lane on a 32-bit Dest, and a named lane that agrees is
    ranked like any other: it is the disagreement that is excused, not the lane."""
    fmt = DataFormat.Float16
    src = torch.tensor([65376.0, 65408.0, 65504.0], dtype=torch.float16)
    golden = src.clone()
    result = torch.full_like(src, float("inf"))
    cell = dict(approx_mode=ApproximationMode.Yes, dest_acc=DestAccumulation.No)
    assert nonfinite_failures(
        MathOperation.Celu, src, golden, result, fmt, fmt, **cell
    ).tolist() == [True, False, False]
    assert nonfinite_failures(
        MathOperation.Celu,
        src,
        golden,
        result,
        fmt,
        fmt,
        approx_mode=ApproximationMode.Yes,
        dest_acc=DestAccumulation.Yes,
    ).all()
    # An op with no entry, and a caller that does not name the cell, get no excuse.
    assert nonfinite_failures(_OP, src, golden, result, fmt, fmt, **cell).all()
    assert nonfinite_failures(MathOperation.Celu, src, golden, result, fmt, fmt).all()
    assert measurable_mask(src, golden, golden.clone(), fmt).all()

    (entry,) = _known_lanes()[MathOperation.Celu]
    assert entry.applies_to(fmt, fmt, ApproximationMode.No, DestAccumulation.No)
    assert not entry.applies_to(
        DataFormat.Float16_b, fmt, ApproximationMode.No, DestAccumulation.No
    )
    assert not entry.applies_to(
        fmt, DataFormat.Float16_b, ApproximationMode.No, DestAccumulation.No
    )


def test_an_entry_no_lane_of_which_disagrees_is_stale():
    """The day the defect is fixed its lanes agree, and the gate has to say so rather
    than keep excusing them."""
    fmt = DataFormat.Float16
    src = torch.tensor([65408.0, 65504.0], dtype=torch.float16)
    golden = src.clone()
    cell = (fmt, fmt, ApproximationMode.No, DestAccumulation.No)
    still_broken = torch.tensor([float("inf"), 65504.0], dtype=torch.float16)
    assert stale_excuses(MathOperation.Celu, src, golden, still_broken, *cell) == []
    fixed = golden.clone()
    assert [
        e.issue for e in stale_excuses(MathOperation.Celu, src, golden, fixed, *cell)
    ] == ["#58607"]
    # On a cell the entry does not name there is nothing to go stale.
    other = (fmt, fmt, ApproximationMode.No, DestAccumulation.Yes)
    assert stale_excuses(MathOperation.Celu, src, golden, fixed, *other) == []
    # An entry another exclusion already covers is stale too: against a NaN golden the
    # same inf answers are excused before the entry is consulted, so it buys nothing.
    nan_golden = torch.full_like(golden, float("nan"))
    assert stale_excuses(MathOperation.Celu, src, nan_golden, still_broken, *cell) == [
        _known_lanes()[MathOperation.Celu][0]
    ]


def test_every_known_lane_entry_names_an_issue_and_a_gateable_cell():
    for op, entries in _known_lanes().items():
        for entry in entries:
            where = f"{op.name}: {entry}"
            assert re.fullmatch(r"#\d+", entry.issue), where
            assert entry.inputs and entry.low <= entry.high and entry.why, where
            # A block output is never step-gated, so there would be nothing to keep gated.
            assert has_ulp_gate(entry.output), where


def test_xdist_workers_measurements_merge_worst_lane_first(table):
    """Under ``-n`` each worker fills its own ``MEASURED``; the controller merges the
    exports. A cell two workers both measured keeps the worse reading, as ``record``
    does within one process, and the merged table is what one process would write."""
    record("Gelu", ("Float16", "Float16", "No", "No"), 5)
    worker_a = export_measured()
    MEASURED.clear()
    record("Gelu", ("Float16", "Float16", "No", "No"), 9)
    record("Gelu", ("Float16", "Float16", "Yes", "No"), 2)
    worker_b = export_measured()
    MEASURED.clear()

    merge_measured(worker_a)
    merge_measured(worker_b)
    assert MEASURED == {
        "Gelu": {
            ("Float16", "Float16", "No", "No"): 9,
            ("Float16", "Float16", "Yes", "No"): 2,
        }
    }


@pytest.mark.parametrize(
    "unmeasurable_first", [True, False], ids=["str,int", "int,str"]
)
def test_an_unmeasurable_cell_survives_the_xdist_merge_in_either_order(
    table, unmeasurable_first
):
    """One worker's reason string and another's step count for the same cell: the
    verdict is "not measurable" whichever export the controller merges first. A number
    must not rescue an overflow, and a string must never reach ``max()``."""
    from helpers.ulp_sweep import record_unmeasurable

    key = ("Float16", "Float16", "No", "No")
    record_unmeasurable("Gelu", key, "3 lane(s) non-finite against a finite golden")
    with_reason = export_measured()
    MEASURED.clear()
    record("Gelu", key, 4)
    with_number = export_measured()
    MEASURED.clear()

    for export in [with_reason, with_number][:: 1 if unmeasurable_first else -1]:
        merge_measured(export)
    assert MEASURED == {"Gelu": {key: "3 lane(s) non-finite against a finite golden"}}


@pytest.mark.parametrize(
    "arch, failed, refusal",
    [
        (ChipArchitecture.BLACKHOLE, 0, "unkeyed rows are read as wormhole"),
        (ChipArchitecture.WORMHOLE, 3, "3 failure"),
    ],
    ids=["other-arch", "failed-session"],
)
def test_emit_refuses_what_the_session_cannot_vouch_for(table, arch, failed, refusal):
    record("Gelu", ("Float16", "Float16", "No", "No"), 5)
    before = table.read_text()
    with _refuses(refusal, RuntimeError):
        finish_emit(arch, failed, table)
    assert table.read_text() == before


def test_emit_writes_on_a_clean_wormhole_session(table):
    for approx in ("No", "Yes"):
        for dest in ("No", "Yes"):
            record("Gelu", ("Float16", "Float16", approx, dest), 5)
    message = finish_emit(ChipArchitecture.WORMHOLE, 0, table)
    assert message == "--ulp-emit: rewrote 1 op block(s) in budget.yaml"
    assert "{in: Float16, out: Float16, max_ulp: 6}" in _rows(table)[0]


def test_the_claim_is_the_whole_format_not_the_drivers_sampling_window():
    """Abs is sampled on (-10, 10) by the functional driver and defined everywhere, so a
    hardware inf at 1e20 against a finite golden is a failure, not an excused lane."""
    src = torch.tensor([1e20], dtype=torch.bfloat16)
    result = torch.tensor([float("inf")], dtype=torch.bfloat16)
    fmt = DataFormat.Float16_b
    assert nonfinite_failures(_OP, src, src.clone(), result, fmt, fmt).any()


@pytest.mark.parametrize(
    "fmt, inputs",
    [(DataFormat.Float16_b, [-2.0, 0.5, 5.0]), (DataFormat.Bfp8_b, [-50.0, 50.0])],
    ids=lambda v: getattr(v, "name", None) or "inputs",
)
def test_an_interval_domain_is_not_clipped_to_the_specs_default_bounds(fmt, inputs):
    """Reciprocal registers its domain as ``intervals``; ANDing the spec's default
    [0, 1] in as well shrank it to [0.1, 1] -- and to nothing for a Bfp8_b input -- so a
    hardware inf on most of the domain was neither counted nor reported."""
    src = torch.tensor(inputs, dtype=torch.bfloat16)
    golden = 1.0 / src
    result = torch.full_like(src, float("inf"))
    op = MathOperation.Reciprocal
    assert nonfinite_failures(op, src, golden, result, fmt, DataFormat.Float16_b).all()
    # Its pole at zero still excuses a lane.
    zero = torch.tensor([0.0], dtype=torch.bfloat16)
    one = torch.tensor([1.0], dtype=torch.bfloat16)
    assert not nonfinite_failures(op, zero, one, result[:1], fmt, fmt).any()


@pytest.mark.parametrize(
    "op, inside, outside",
    [
        (MathOperation.Reciprocal, [1e-7, -1e-7, 1e-30], []),
        (MathOperation.Log, [1e-7, 1e-30], [-1e-7, -2.0]),
        (MathOperation.Rsqrt, [1e-7], [-1e-7]),
        (MathOperation.Asin, [-1.0, 0.5, 1.0], [-1.5, 2.0]),
    ],
    ids=lambda v: getattr(v, "name", None) or "",
)
def test_the_claim_is_the_singularity_not_the_sampling_guard_band(op, inside, outside):
    """``_SFPU_UNDEFINED_RANGES`` keeps a random draw 1e-6 off Reciprocal's pole, and
    that band was the claim: a bf16 ``1/1e-7`` returning inf was excused in both the
    emitted measurement and the gate. The claim is ``_OP_SINGULARITIES`` -- the point,
    and the side the op is undefined on."""
    fmt = DataFormat.Float16_b
    finite = torch.tensor([1.0], dtype=torch.bfloat16)
    inf = torch.tensor([float("inf")], dtype=torch.bfloat16)
    for value, claimed in [(v, True) for v in inside] + [(v, False) for v in outside]:
        src = torch.tensor([value], dtype=torch.bfloat16)
        assert (
            bool(nonfinite_failures(op, src, finite, inf, fmt, fmt).any()) is claimed
        ), value


def test_an_unmeasurable_cell_is_written_as_its_own_verdict(table):
    """Left out, the cell's old row is dropped with nothing replacing it; recorded, it
    becomes a tolerance row that says why."""
    from helpers.ulp_sweep import record_unmeasurable

    record("Gelu", ("Float16", "Float16", "No", "No"), 5)
    record_unmeasurable(
        "Gelu", ("Float16", "Float16", "Yes", "No"), "no measurable lane"
    )
    write_table(table, "today")
    rows = _rows(table)
    assert any("max_ulp: 6" in r and 'approx: "No"' in r for r in rows)
    assert any(
        'approx: "Yes"' in r
        and "metric: tolerance" in r
        and "not measurable: no measurable lane" in r
        for r in rows
    )


def test_emit_refuses_a_partly_measured_grid(table):
    """`write_table` replaces every row of a touched (in, out); a run narrowed inside
    one would drop the rows of the cells it never reached."""
    record("Gelu", ("Float16", "Float16", "No", "No"), 5)
    before = table.read_text()
    with _refuses("Gelu Float16->Float16: 3 cell", RuntimeError):
        finish_emit(ChipArchitecture.WORMHOLE, 0, table)
    assert table.read_text() == before


def test_an_op_with_an_unregenerable_row_is_kept_while_the_rest_are_written(tmp_path):
    """A floor row the sweep covers cannot be re-derived from a measurement. Its op
    keeps its block verbatim, every other op is written, and the caller is told."""
    path = tmp_path / "budget.yaml"
    path.write_text(
        "Erfinv:\n"
        "  - {in: Float16, out: Float16, max_ulp: 2, near_zero_atol: 1.0e-07}  # floor\n"
        "\n"
        "Gelu:\n"
        "  - {in: Float16, out: Float16, max_ulp: 7}  # superseded\n",
        encoding="utf-8",
    )
    MEASURED.clear()
    record("Erfinv", ("Float16", "Float16", "No", "No"), 5)
    record("Gelu", ("Float16", "Float16", "No", "No"), 5)
    written, kept = write_table(path, "today")
    MEASURED.clear()
    text = path.read_text()
    assert (written, kept) == (1, ["Erfinv"])
    assert "near_zero_atol: 1.0e-07}  # floor" in text
    assert "max_ulp: 7" not in text and "max_ulp: 6" in text


def test_a_rewritten_block_keeps_one_blank_line_before_the_next(table):
    record("Gelu", ("Float16", "Float16", "No", "No"), 5)
    write_table(table, "today")
    assert "\n\n\n" not in table.read_text()
