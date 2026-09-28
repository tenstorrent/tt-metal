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
from helpers.format_config import DataFormat
from helpers.ulp_sweep import (
    EMIT_HEADROOM,
    MEASURED,
    measurable_mask,
    nonfinite_failures,
    record,
    write_table,
)

#: One op block with a row of each kind write_table has to tell apart.
_TABLE = """Gelu:  # header provenance, ungeneratable
  - {in: Float16, out: Float16, max_ulp: 7}  # superseded
  - {in: Float16_b, out: Float32, max_ulp: 9}  # fp32 output, not swept
  - {in: Float16_b, out: Float16_b, arch: BLACKHOLE, max_ulp: 44}  # another arch

Log1p:
  - {in: Float16, out: Float16_b, metric: tolerance}  # hand-authored non-enrolment
"""


def _refuses(match):
    """The suite's ``expect_error`` fixture needs a device; these are host-only tests."""
    return pytest.raises(ValueError, match=match)  # allow-pytest.raises: host-only test


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
    assert write_table(table, "today") == 1

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
    with _refuses("cannot regenerate"):
        write_table(table, "today")
    assert "near_zero_atol" in table.read_text()  # and nothing was written


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
    assert nonfinite_failures(src, golden, result, fmt).tolist() == [False, True, False]


def test_a_flushed_subnormal_input_is_not_a_nonfinite_failure():
    """The unpack path flushes it and the golden does not, so a disagreement there is
    the flush, not the op — the same exclusion `measurable_mask` makes."""
    tiny = float(torch.finfo(torch.bfloat16).smallest_normal) / 2
    src = torch.tensor([tiny], dtype=torch.bfloat16)
    golden = torch.tensor([1.0], dtype=torch.bfloat16)
    result = torch.tensor([float("inf")], dtype=torch.bfloat16)
    assert not nonfinite_failures(src, golden, result, DataFormat.Float16_b).any()
