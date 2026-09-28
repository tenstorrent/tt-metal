# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for the emitter in ``helpers/ulp_sweep.py``.

No device: ``write_table`` is a line-oriented rewrite of a YAML file, and everything
worth pinning is about what it must *not* touch. The table's rows are contracts, so a
regeneration that quietly drops one weakens a gate with nothing to notice.
"""

import math

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat
from helpers.llk_params import MathOperation
from helpers.ulp_sweep import (
    EMIT_HEADROOM,
    MEASURED,
    export_measured,
    finish_emit,
    measurable_mask,
    merge_measured,
    nonfinite_failures,
    record,
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
    # The op this run never measured, untouched.
    assert any("Log1p" in l for l in table.read_text().splitlines())
    assert "{in: Float16, out: Float16_b, metric: tolerance}" in rows[-1]


def test_a_row_for_another_architecture_survives_a_regeneration(table):
    """The sweep runs on one arch and the table's header says to re-measure on the
    other. Specificity lets the two rows coexist, so an arch-keyed row is never
    replaceable — even though its `in`/`out` are both swept formats."""
    record("Gelu", ("Float16_b", "Float16_b", "No", "No"), 3)
    write_table(table, "today")
    assert any("arch: BLACKHOLE, max_ulp: 44" in row for row in _rows(table))


def test_a_row_carrying_a_floor_is_refused_rather_than_regenerated(table):
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


def test_a_measurement_with_nowhere_to_go_is_refused(table):
    """The key line is passed through verbatim so a header comment survives, so a new
    op's block has to be hand-authored first. Dropping the measurement in silence is
    what left 17 ops' sampled rows in place looking measured."""
    record("Sqrt", ("Float16", "Float16", "No", "No"), 1)
    with _refuses("no key line"):
        write_table(table, "today")


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
    singular at zero they would read as a real failure -- `reciprocal` returns `Inf`
    there against a finite golden clamp, and only its registered domain excluding zero
    keeps those lanes out of the verdict today.

    By position, not by value: one legitimate `0.0` is swept, in the middle.
    """
    from helpers.stimuli_generator.strategies.structured import ulp_sweep_value_count
    from helpers.ulp_sweep import padding_lanes

    fmt = DataFormat.Float16_b
    swept = ulp_sweep_value_count(fmt, float("-inf"), float("inf"))
    src = torch.zeros(swept + 257, dtype=torch.bfloat16)
    pad = padding_lanes(src, fmt)
    assert int(pad.sum()) == 257
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
    # Inside [-pi, pi] the same disagreement is a failure.
    assert nonfinite_failures(
        MathOperation.Sin,
        torch.tensor([1.0], dtype=torch.bfloat16),
        golden,
        result,
        fmt,
        fmt,
    ).any()


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
    # Its undefined hole around zero still excuses a lane.
    zero = torch.tensor([0.0], dtype=torch.bfloat16)
    one = torch.tensor([1.0], dtype=torch.bfloat16)
    assert not nonfinite_failures(op, zero, one, result[:1], fmt, fmt).any()


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
