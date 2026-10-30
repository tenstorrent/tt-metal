"""Whether a rung was TRIED is history; whether its verdict still counts is not.

The resume filter drops every attempt whose `baseline_at_record` differs from the baseline measured
now, and REWRITES the live log with what survives (run.py:3049-3061):

    stamp = r.get("baseline_at_record")
    if stamp is None or base is None or round(float(stamp), 4) != base:
        continue
    ...
    Path(kernel_log).write_text(json.dumps(keep))

That rule is right for a VERDICT. A "no gain" earned against a different baseline has not been shown
to hold now, so it must not seed a skip.

It is wrong for the question the ladder actually asks. `_op_ladder_status` counts rungs to decide
what to hand out next, and it reads the filtered file:

    perf_mcp.py:4057   attempts = [a for a in _load_attempts() if a.get("kernel_detected_in_source")]
    perf_mcp.py:532    _load_attempts() -> _KERNEL_LOG_PATH        # the FILTERED live log

so an attempt that genuinely happened reads as never having happened once the baseline moves. The
cumulative archive that remembers everything is read in exactly ONE place -- _rebuild_optimize_report
-- to draw RUN_REPORT.md. No decision path consults it.

Measured on gemma-3-12b-it: the baseline lands on 381.186 / 381.222 / 381.263 / 381.291 / 381.311 on
successive runs, and the filter compares with `round(stamp, 4) != base`, exact equality. So a
different subset survives every run, different rungs reappear, and MinimalMatmul 1024x3840x8192 was
measured at `grid` in runs 22, 24, 25 and 26. Run 25 went from 179 rows to 121 -- the dropped rows
included that op's structural/tt-lang/cpp attempts, which is what un-capped its knob retries.

The split: "was it tried" comes from the union of archive + live and is permanent; the verdict still
comes from the baseline-matched rows and can still expire.
"""

import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))


@pytest.fixture()
def mcp(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_KERNEL_LOG", str(tmp_path / "kl.json"))
    import models.experimental.perf_automation.cc_optimize.perf_mcp as m

    importlib.reload(m)
    return m


def _row(sig, kind, stamp, **kw):
    r = {"op_signature": sig, "kernel_kind": kind, "baseline_at_record": stamp, "measured_ms": 400.0}
    r.update(kw)
    return r


def _write(mcp, live, cum):
    Path(mcp._KERNEL_LOG_PATH).write_text(json.dumps(live))
    Path(str(mcp._KERNEL_LOG_PATH) + ".cumulative").write_text(json.dumps(cum))


# ---------------------------------------------------------------- the reported case


def test_a_rung_dropped_by_the_filter_still_counts_as_tried(mcp):
    """The gemma3 case: cpp was tried at an older baseline, the filter removed it from the live log,
    and the op then looked like it had never gone deep."""
    live = [_row("Matmul A", "grid", 381.31)]
    cum = live + [_row("Matmul A", "cpp", 381.22)]
    _write(mcp, live, cum)
    kinds = {(a.get("kernel_kind") or "").lower() for a in mcp._load_attempts_all()}
    assert "cpp" in kinds, kinds


def test_the_live_log_alone_would_have_missed_it(mcp):
    """Guards the premise -- if the filtered log already carried it, there is nothing to fix."""
    live = [_row("Matmul A", "grid", 381.31)]
    _write(mcp, live, live + [_row("Matmul A", "cpp", 381.22)])
    assert "cpp" not in {(a.get("kernel_kind") or "").lower() for a in mcp._load_attempts()}


def test_an_unstamped_attempt_still_counts_as_tried(mcp):
    """Rows written before the stamp existed, and the ones a crashed run leaves behind, happened too."""
    _write(mcp, [], [_row("Matmul A", "shard", None)])
    assert len(mcp._load_attempts_all()) == 1


# ---------------------------------------------------------------- the union is a union


def test_live_rows_survive_when_the_archive_is_missing(mcp):
    """First run for a model: no archive yet. Must not lose the live rows."""
    Path(mcp._KERNEL_LOG_PATH).write_text(json.dumps([_row("Matmul A", "grid", 1.0)]))
    assert len(mcp._load_attempts_all()) == 1


def test_a_corrupt_archive_does_not_lose_the_live_rows(mcp):
    """This runs every termination_check; a bad archive must degrade, not erase history."""
    Path(mcp._KERNEL_LOG_PATH).write_text(json.dumps([_row("Matmul A", "grid", 1.0)]))
    Path(str(mcp._KERNEL_LOG_PATH) + ".cumulative").write_text("{not json")
    assert len(mcp._load_attempts_all()) == 1


def test_the_same_attempt_in_both_files_is_not_double_counted(mcp):
    """_fold_cumulative copies live rows into the archive, so overlap is the normal case. Counting a
    rung twice would spend its retry allowance on one attempt."""
    r = _row("Matmul A", "grid", 381.31)
    _write(mcp, [r], [r])
    grid = [a for a in mcp._load_attempts_all() if a.get("kernel_kind") == "grid"]
    assert len(grid) == 1, grid


def test_two_genuinely_distinct_attempts_at_one_rung_both_count(mcp):
    """A second grid variant is a real second attempt -- collapsing them would hand out a third."""
    a1 = _row("Matmul A", "grid", 381.31, measured_ms=400.0, note="first")
    a2 = _row("Matmul A", "grid", 381.31, measured_ms=390.0, note="second")
    _write(mcp, [a1, a2], [a1, a2])
    assert len([a for a in mcp._load_attempts_all() if a.get("kernel_kind") == "grid"]) == 2


def test_no_files_at_all_is_empty_not_an_error(mcp):
    assert mcp._load_attempts_all() == []


# ---------------------------------------------------------------- verdicts stay baseline-scoped


def test_the_verdict_reader_is_untouched(mcp):
    """_load_attempts still returns ONLY the filtered rows. A verdict earned against another
    baseline must not start counting as settled -- that is the half the filter gets right."""
    live = [_row("Matmul A", "grid", 381.31)]
    _write(mcp, live, live + [_row("Matmul A", "cpp", 381.22)])
    assert len(mcp._load_attempts()) == 1
