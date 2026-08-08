"""A run's own starting measurement is not a result, and must not displace one.

The full-pipeline bookend records into the ledger with a write-once BEFORE: the first reading claims
the BEFORE slot, everything later lands as an AFTER. That rule is right for RESULTS -- it keeps the
original baseline alive across reruns so the headline reads 84.05 -> x rather than resetting.

It is wrong for the next run's own BEFORE bookend. That reading is a baseline, not an outcome, and
filing it as an AFTER makes it the newest "current state":

    fullpipe_e2e  before  84.0539   fullpipe-gate:BEFORE
    fullpipe_e2e  after   34.9909   fullpipe-gate:committed-best   (run 20, a real committed result)
    fullpipe_e2e  after   36.2548   fullpipe-gate:BEFORE           (run 21's cold baseline)

Note the source string on the last row still says BEFORE -- correctly labelled, filed in the wrong
phase. Readers take the last AFTER as the current state, so a fresh cold measurement displaced a
committed one: the report showed 36.25, and the next run's bar became 36.25 instead of 34.99. Both
wrong in the same direction, and the run then measures warm and "beats" its own cold start.

So a BEFORE bookend recorded when a BEFORE already exists is DROPPED, not reclassified. The original
anchor stays, the committed best stays, and the run's starting measurement lives where it belongs --
in the gate's own baseline file, which is per-run and not a record of progress.
"""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))


@pytest.fixture()
def run_mod(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_MODEL_NAME", "gemma3")
    monkeypatch.setenv("PERF_MCP_TASK", "main")
    import models.experimental.perf_automation.cc_optimize.run as R

    importlib.reload(R)
    return R


def _rows(run_mod):
    led = run_mod._ledger()
    return [r for r in led.rows(led.KIND_FULLPIPE, model="gemma3", task="main")]


def _phases(run_mod):
    return [(r["phase"], r["value_ms"]) for r in _rows(run_mod)]


# ---------------------------------------------------------------- the reported case


def test_the_first_before_claims_the_anchor(run_mod):
    run_mod._record_fullpipe_bookend(84.0539, "trace+1cq", "BEFORE")
    assert _phases(run_mod) == [("before", 84.0539)]


def test_a_second_before_is_dropped_not_reclassified(run_mod):
    """Run 21's cold 36.2548 must not become the newest 'after'."""
    run_mod._record_fullpipe_bookend(84.0539, "trace+1cq", "BEFORE")
    run_mod._record_fullpipe_bookend(34.9909, "trace+1cq", "committed-best")
    run_mod._record_fullpipe_bookend(36.2548, "trace+1cq", "BEFORE")
    assert _phases(run_mod) == [("before", 84.0539), ("after", 34.9909)]


def test_the_committed_best_survives_a_rerun(run_mod):
    """The bar for the next run stays the best committed result, not the cold restart."""
    run_mod._record_fullpipe_bookend(84.0539, "trace+1cq", "BEFORE")
    run_mod._record_fullpipe_bookend(34.9909, "trace+1cq", "committed-best")
    run_mod._record_fullpipe_bookend(36.2548, "trace+1cq", "BEFORE")
    led = run_mod._ledger()
    assert led.last(led.KIND_FULLPIPE, led.PHASE_AFTER, model="gemma3", task="main")["value_ms"] == 34.9909


# ---------------------------------------------------------------- results still record


def test_results_still_land_as_after(run_mod):
    """Only the BEFORE bookend is dropped. Committed results are the whole point of the AFTER slot."""
    run_mod._record_fullpipe_bookend(84.0539, "trace+1cq", "BEFORE")
    for ms in (60.2, 46.5, 34.99):
        run_mod._record_fullpipe_bookend(ms, "committed-best", "trace+1cq")
    assert _phases(run_mod) == [("before", 84.0539), ("after", 60.2), ("after", 46.5), ("after", 34.99)]


def test_a_result_slower_than_the_best_still_records(run_mod):
    """Dropping is about the BEFORE label, never about the number being unwelcome."""
    run_mod._record_fullpipe_bookend(84.0539, "trace+1cq", "BEFORE")
    run_mod._record_fullpipe_bookend(34.99, "trace+1cq", "committed-best")
    run_mod._record_fullpipe_bookend(37.10, "trace+1cq", "committed-best")
    assert ("after", 37.10) in _phases(run_mod)


def test_a_first_run_with_no_anchor_still_gets_one(run_mod):
    """On a fresh model the BEFORE bookend IS the anchor and must be kept."""
    run_mod._record_fullpipe_bookend(36.2548, "trace+1cq", "BEFORE")
    assert _phases(run_mod) == [("before", 36.2548)]


def test_an_unlabelled_bookend_is_treated_as_a_result(run_mod):
    """Fail toward RECORDING: a reading whose provenance is unknown must not be silently discarded."""
    run_mod._record_fullpipe_bookend(84.0539, "trace+1cq", "BEFORE")
    run_mod._record_fullpipe_bookend(35.0, "trace+1cq", "")
    assert ("after", 35.0) in _phases(run_mod)
