# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for the ``--ulp-measure`` recorder in ``helpers/utils.py``: what
``passed_test`` files, under which variant, and what it must not."""

import subprocess
import sys

import helpers.chip_architecture as chip
import helpers.sfpu_accuracy_budget as budget
import helpers.utils as utils
import pytest
import torch
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, MathOperation
from helpers.sfpu_accuracy_budget import accuracy_contract
from helpers.ulp import ulp_distance
from helpers.ulp_budget_diff import read_cells
from helpers.ulp_table import MEASURED_ARCH, TABLE_PATH, BudgetKey, read_measurements
from helpers.utils import RecordDropped, _record_ulp_measurement, passed_test

BF16 = DataFormat.Float16_b


@pytest.fixture
def rows(tmp_path, monkeypatch):
    """The recorder pointed at a file, and the reader of what it wrote."""
    path = tmp_path / "measure.jsonl"
    monkeypatch.setattr(utils, "_ULP_MEASURE_PATH", str(path))
    return lambda: read_measurements(path)


def _lookup(**query):
    accuracy_contract(MathOperation.Abs, arch=MEASURED_ARCH, **query)


def test_a_record_the_sweep_requires_fails_rather_than_being_dropped(rows):
    """The headroom report judges only the cells it reads, so a sweep record that would
    be dropped fails its cell; a functional driver's is only lost."""
    golden = torch.full((32,), 1.5, dtype=torch.bfloat16)
    reading = dict(
        distance=ulp_distance(golden, golden.clone()),
        mask=torch.ones(32, dtype=torch.bool),
    )
    _record_ulp_measurement(**reading)  # no query pending: dropped
    with pytest.raises(RecordDropped):  # allow-pytest.raises: host-only test
        _record_ulp_measurement(**reading, required=True)
    assert rows() == []
    _lookup(output_format=BF16)
    _record_ulp_measurement(**reading, required=True)
    assert len(rows()) == 1


def test_the_recorder_files_one_row_under_the_variant_just_resolved(rows):
    """A comparison is tagged with `accuracy_contract`'s last query and consumes it: a
    second comparison in the same test that never went through the registry must not
    inherit the variant, and the row names the variant asked for rather than the
    tensors' format."""
    golden = torch.full((32,), 1.5, dtype=torch.bfloat16)
    _lookup(
        output_format=BF16,
        input_format=DataFormat.Float16,
        dest_acc=DestAccumulation.Yes,
    )
    assert passed_test(golden, golden.clone(), BF16)
    assert passed_test(golden, golden.clone(), BF16)  # no lookup
    (row,) = rows()
    assert row.op is MathOperation.Abs
    assert row.key == BudgetKey(
        input_format=DataFormat.Float16,
        output_format=BF16,
        dest_acc=DestAccumulation.Yes,
        arch=MEASURED_ARCH,
    )
    assert (row.worst, row.lanes, row.unmeasurable, row.nonfinite) == (0, 32, 0, 0)


def test_the_recorder_writes_the_nonfinite_count_the_headroom_report_reads(rows):
    golden = torch.full((32,), 1.5, dtype=torch.bfloat16)
    _lookup(output_format=BF16, input_format=BF16)
    _record_ulp_measurement(
        ulp_distance(golden, golden.clone()),
        mask=torch.ones(32, dtype=torch.bool),
        nonfinite=7,
    )
    assert [cell.nonfinite for cell in read_cells(rows()).values()] == [7]


def test_the_recorder_ranks_the_lanes_the_verdict_ranks(rows):
    """Without the lanes a `near_zero_atol` floor rescued: those carry the largest step
    counts by construction, and filing them would fold a floor-carried pass back as a
    budget."""
    golden = torch.full((32,), 1.5, dtype=torch.bfloat16)
    golden[0] = 1e-6
    result = golden.clone()
    result[0] = 3e-6  # about 200 steps away, 2e-6 in absolute terms
    result[1] = 1.5078125  # one bf16 step above 1.5
    _lookup(output_format=BF16)
    assert passed_test(golden, result, BF16, max_ulp=1, near_zero_atol=1e-5)
    (row,) = rows()
    assert row.worst == 1, row


def test_the_recorder_drops_an_ambiguous_or_promoted_variant(rows, monkeypatch):
    """Two lookups and then one comparison cannot say which variant it was, and a
    variant TestConfig promoted to a 32-bit Dest ran another kernel than it names.
    Neither is filed."""
    monkeypatch.setenv("CHIP_ARCH", "wormhole")
    monkeypatch.setattr(chip, "_cached_chip_architecture", None)
    golden = torch.full((32,), 1.5, dtype=torch.bfloat16)
    for _ in range(2):
        _lookup(output_format=BF16)
    assert passed_test(golden, golden.clone(), BF16)
    assert rows() == []
    half = golden.to(torch.float16)
    cell = dict(
        output_format=DataFormat.Float16,
        input_format=BF16,
        dest_acc=DestAccumulation.No,
    )
    _lookup(**cell)
    assert passed_test(half, half.clone(), DataFormat.Float16)
    assert rows() == []
    # The promotion is the resolved architecture's, not the process's: Quasar runs the
    # 16-bit Dest it was asked for, so the same cell resolved for it is filed, under it.
    accuracy_contract(MathOperation.Abs, arch=ChipArchitecture.QUASAR, **cell)
    assert passed_test(half, half.clone(), DataFormat.Float16)
    (row,) = rows()
    assert (row.key.arch, row.key.dest_acc) == (
        ChipArchitecture.QUASAR,
        DestAccumulation.No,
    )


def test_every_comparison_spends_the_query_it_was_resolved_for(rows):
    """A path that writes no row (a shape mismatch, an output with no per-element ULP)
    still spends the lookup, or the next comparison in the test would inherit the
    variant and the next lookup read as ambiguous."""
    golden = torch.full((32,), 1.5, dtype=torch.bfloat16)
    for result, fmt in (
        (golden.clone().reshape(1, 32), BF16),  # broadcast compare
        (golden.clone(), DataFormat.Bfp4_b),  # no per-element ULP
    ):
        _lookup(output_format=fmt)
        passed_test(golden, result, fmt)
        assert (budget.LAST_QUERY, budget.PENDING_AMBIGUOUS) == (None, False), fmt


def test_a_comparison_does_not_load_the_budget_table():
    """`passed_test` spends the registry's pending query on every call but must not
    import the registry to do it: every caller would then load and validate the whole
    table. In a fresh interpreter, since this one has it."""
    code = (
        f"import sys, torch; from {utils.__name__} import passed_test; "
        "from helpers.format_config import DataFormat; "
        "golden = torch.ones(32, dtype=torch.bfloat16); "
        "assert passed_test(golden, golden.clone(), DataFormat.Float16_b); "
        f"sys.exit({budget.__name__!r} in sys.modules)"
    )
    done = subprocess.run([sys.executable, "-c", code], cwd=TABLE_PATH.parent.parent)
    assert done.returncode == 0


def test_a_tolerance_reading_ranks_fp16_subnormals_flushed(rows):
    """Every step-budget gate on Float16 ranks the subnormal band flushed, so a reading
    from a tolerance cell must too, or a folded-back figure is looser than its gate."""
    golden = torch.full((32,), 1.5, dtype=torch.float16)
    golden[0] = 6e-08  # the smallest fp16 subnormal: one step from 0 unflushed
    result = golden.clone()
    result[0] = 0.0
    _lookup(output_format=DataFormat.Float16)
    assert passed_test(golden, result, DataFormat.Float16)
    (row,) = rows()
    assert row.worst == 0, row


def test_the_measure_file_is_truncated_at_session_start_or_refused(
    tmp_path, monkeypatch
):
    """A stale file is truncated so runs do not fold together, a missing parent is
    created, and a directory raises for the plugin to turn into a usage error."""
    target = tmp_path / "nested" / "measure.jsonl"
    monkeypatch.setattr(utils, "_ULP_MEASURE_PATH", str(target))
    target.parent.mkdir()
    target.write_text("stale\n", encoding="utf-8")
    utils.prepare_ulp_measure_file()
    assert target.read_text() == ""
    fresh = tmp_path / "out" / "dir" / "measure.jsonl"
    monkeypatch.setattr(utils, "_ULP_MEASURE_PATH", str(fresh))
    utils.prepare_ulp_measure_file()
    assert fresh.read_text() == ""
    monkeypatch.setattr(utils, "_ULP_MEASURE_PATH", str(tmp_path))  # a directory
    with pytest.raises(OSError):  # allow-pytest.raises: host-only test
        utils.prepare_ulp_measure_file()


def test_a_recorder_write_failure_does_not_fail_the_comparison(tmp_path, monkeypatch):
    """Reporting only: an unwritable path is warned about once (a path that stops
    taking writes mid-run would otherwise warn on every comparison of every worker)
    and otherwise ignored."""
    monkeypatch.setattr(utils, "_ULP_MEASURE_PATH", str(tmp_path))  # a directory
    monkeypatch.setattr(utils, "_ULP_MEASURE_WARNED", False)
    warnings = []
    monkeypatch.setattr(
        utils.logger, "warning", lambda text, *args: warnings.append(text.format(*args))
    )
    golden = torch.full((32,), 1.5, dtype=torch.bfloat16)
    for _ in range(3):
        _lookup(output_format=BF16)
        assert passed_test(golden, golden.clone(), BF16)
    assert len(warnings) == 1 and "--ulp-measure: cannot write" in warnings[0]
