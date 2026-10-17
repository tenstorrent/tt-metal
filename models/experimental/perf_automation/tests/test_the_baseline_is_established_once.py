"""The baseline is established ONCE per model, and afterwards moves only downward.

Two defects, one root cause: `perf_mcp_baseline_<model>_<task>.json` was doing two jobs.

    the PICTURE   per-op buckets, tags, roofline gaps -- the agent needs these fresh every
                  iteration to choose what to work on next
    the BAR       device_ms -- what measure_candidate grades every candidate against

profile_model wrote both, unconditionally, on every call:

    _baseline_path().write_text(json.dumps(prof))        # perf_mcp.py:1826

and its own docstring invites re-running it ("call this again whenever you want a fresh picture").
On gemma-3-12b-it it ran ~44 times in a single run. So refreshing the picture silently redefined
the bar.

What that produces:

  1. The agent applies an edit that makes the model SLOWER, re-profiles for a fresh picture, and the
     slower number becomes the baseline. measure_candidate now grades against it -- so REVERTING the
     bad edit reads as a win, and gets banked as one.

  2. The baseline drifts across runs on unchanged code: 381.186 / 381.222 / 381.263 / 381.291 /
     381.311. The resume filter compares `baseline_at_record` with EXACT equality
     (run.py:3049), so a different subset of attempt history survived every run. That is the
     upstream cause of the 38% (op, rung) repeat rate -- MinimalMatmul 1024x3840x8192/grid was
     measured in four separate runs.

The full-pipeline bar has ratcheted since 9358229fa8. This is the same rule for the steering metric,
plus the stronger guarantee the operator asked for: on a SECOND optimize of the same model, the
baseline is not re-measured at all -- it is read from the first run.

That second part is enforced in the code path (before_loop takes a branch that skips the
measurement), NOT as guidance in the agent prompt. Advice gets worked around; a branch does not.
PERF_MCP_FORCE_REBASELINE=1 is the deliberate escape hatch.
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
    monkeypatch.delenv("PERF_MCP_FORCE_REBASELINE", raising=False)
    import models.experimental.perf_automation.cc_optimize.perf_mcp as m

    importlib.reload(m)
    return m


def _prof(ms, layers="all", buckets=None):
    return {
        "device_ms": ms,
        "wall_ms": ms * 10,
        "perf_layers": layers,
        "buckets": buckets if buckets is not None else [{"id": "matmul", "count": 10, "ms": ms * 0.7}],
    }


def _bar(mcp):
    return json.loads(mcp._baseline_path().read_text())


# ---------------------------------------------------------------- the bar only ratchets down


def test_the_first_profile_establishes_the_bar(mcp):
    mcp._promote_baseline(_prof(381.26))
    assert _bar(mcp)["device_ms"] == 381.26


def test_a_slower_reprofile_does_not_move_the_bar(mcp):
    """The reported case: re-profiling after a bad edit must not redefine what wins are graded
    against, or reverting that edit reads as a win."""
    mcp._promote_baseline(_prof(381.26))
    mcp._promote_baseline(_prof(400.44))
    assert _bar(mcp)["device_ms"] == 381.26


def test_a_faster_reprofile_does_move_the_bar(mcp):
    """A committed win SHOULD lower it -- that is the whole point of a ratchet."""
    mcp._promote_baseline(_prof(381.26))
    mcp._promote_baseline(_prof(376.76))
    assert _bar(mcp)["device_ms"] == 376.76


def test_an_equal_reading_is_accepted(mcp):
    mcp._promote_baseline(_prof(381.26))
    mcp._promote_baseline(_prof(381.26))
    assert _bar(mcp)["device_ms"] == 381.26


def test_the_drift_that_broke_the_resume_filter_is_gone(mcp):
    """381.186 / 381.222 / 381.263 / 381.291 / 381.311 on unchanged code, compared with exact
    equality by the resume filter. Only the best survives now, so the stamp is stable."""
    for ms in (381.186, 381.222, 381.263, 381.291, 381.311):
        mcp._promote_baseline(_prof(ms))
    assert _bar(mcp)["device_ms"] == 381.186


# ---------------------------------------------------------------- the picture still refreshes


def test_the_per_op_picture_is_refreshed_even_when_the_bar_holds(mcp):
    """The agent needs current buckets to pick its next target; only the BAR is frozen."""
    mcp._promote_baseline(_prof(381.26, buckets=[{"id": "matmul", "count": 10, "ms": 200.0}]))
    mcp._promote_baseline(_prof(400.44, buckets=[{"id": "reduction", "count": 99, "ms": 300.0}]))
    doc = _bar(mcp)
    assert doc["device_ms"] == 381.26
    assert doc["buckets"][0]["id"] == "reduction", doc["buckets"]


def test_the_rejected_reading_is_still_visible(mcp):
    """Held back, not thrown away -- a reader can still see what the board actually reported."""
    mcp._promote_baseline(_prof(381.26))
    out = mcp._promote_baseline(_prof(400.44))
    assert out["observed_device_ms"] == 400.44 and out["device_ms"] == 381.26


# ---------------------------------------------------------------- a shape change is not a regression


def test_changing_the_profiled_depth_rebaselines(mcp):
    """A 4-layer profile and a 48-layer profile are different units. Refusing the slower one would
    pin the bar to a number from a different measurement entirely."""
    mcp._promote_baseline(_prof(30.0, layers="4"))
    mcp._promote_baseline(_prof(381.26, layers="all"))
    assert _bar(mcp)["device_ms"] == 381.26


# ---------------------------------------------------------------- established once, then reused


def test_baseline_exists_reports_the_established_bar(mcp):
    assert mcp.baseline_exists() is False
    mcp._promote_baseline(_prof(381.26))
    assert mcp.baseline_exists() is True


def test_a_zero_baseline_does_not_count_as_established(mcp):
    """A degenerate capture must not suppress the first real measurement."""
    mcp._baseline_path().write_text(json.dumps({"device_ms": 0.0, "buckets": []}))
    assert mcp.baseline_exists() is False


def test_a_corrupt_baseline_does_not_count_as_established(mcp):
    mcp._baseline_path().write_text("{not json")
    assert mcp.baseline_exists() is False


def test_the_bar_is_written_atomically(mcp):
    """A reader catching a truncated write is how the full-pipeline bar was lost once already."""
    src = Path(mcp.__file__).read_text()
    i = src.index("def _promote_baseline")
    body = src[i : src.index("\ndef ", i + 1)]
    assert "os.replace" in body


def test_no_temp_file_is_left_behind(mcp):
    mcp._promote_baseline(_prof(381.26))
    leftovers = [p.name for p in mcp._baseline_path().parent.iterdir() if ".tmp" in p.name]
    assert leftovers == [], leftovers


# ---------------------------------------------------------------- reuse is ENFORCED, not advised


def test_the_second_run_skips_the_measurement_entirely():
    """The operator's requirement: a model that already has a baseline does not get a new one, and
    that must be a branch the run cannot take -- not a line in the prompt the agent can ignore.

    Asserted against the source because the alternative is standing up the whole before_loop
    harness (device, tracy, stages) to observe one branch.
    """
    bl = Path(__file__).resolve().parent.parent / "agent" / "before_loop.py"
    src = bl.read_text()
    # Anchored on the REUSE BRANCH ITSELF rather than a byte window after the first mention of
    # `_stored_baseline`. The window version broke the moment anything was inserted between the two
    # -- which is a test failing on layout, not on the behaviour it names.
    assert "PERF_MCP_FORCE_REBASELINE" in src, "no escape hatch"
    reuse = src.index('stages.start("tracy_baseline", "Reusing')
    measure = src.index("_measure_baseline(", reuse)
    assert src.rindex("if _stored_baseline is not None:", 0, reuse) < reuse, "reuse is not a branch"
    assert reuse < measure, "the measurement must be skipped by the reuse branch, not run before it"
    assert "else:" in src[reuse:measure], "the measurement must sit on the ELSE arm"


def test_the_baseline_file_is_keyed_from_the_model_directory():
    """PERF_MCP_MODEL_NAME is not set until AFTER discover() takes this baseline, so reading it
    first yields "" and the key falls back to the literal "model" -- which is how one gemma3 run
    wrote its anchor to perf_measurements_model_main.jsonl while every other writer used
    perf_measurements_gemma3_main.jsonl. Two ledgers for one run."""
    bl = Path(__file__).resolve().parent.parent / "agent" / "before_loop.py"
    src = bl.read_text()
    line = [ln for ln in src.splitlines() if "_bl_model = " in ln][0]
    assert line.index("Path(model_root).name") < line.index("PERF_MCP_MODEL_NAME"), line
