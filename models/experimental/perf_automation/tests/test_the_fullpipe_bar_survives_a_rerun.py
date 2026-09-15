"""The tok/s/u bar is established once per model and survives every rerun.

There are TWO baselines, and only one of them was protected:

    Step 9/10 "Measuring the baseline latency"   device_ms tracy profile   -- reuse enforced
    run.py:3143 _fullpipe_e2e(..., "BEFORE")     full-pipeline trace+1cq   -- NOT protected

The second is the tok/s/u number every win is banked against, and it was re-established from
scratch on every optimize run, because the line immediately before it DELETED the file:

    _reset_fullpipe_baselines()                  # unlink(), unconditionally
    before_ms, before_mode = _fullpipe_e2e(...)  # measure, and become the new bar

That one line defeated every protection built around the bar. The ratchet in
_promote_fullpipe_pending had nothing to ratchet against. The "is the bar readable" guard was moot:
the file was not corrupt, it was absent, and establishing a new bar when none exists is correct.

The damage, on gemma-3-12b-it:

    run 30   68.3241 ms   a thermally clamped reading (14.6 tok/s/u vs a true ~34) became the
                          anchor; every candidate that run would have been graded against it
    run 33   35.9253 ms   replaced a committed 33.981 (29.4 -> 27.8 tok/s/u)

Both were restored by hand. Neither should have been possible.

The delete was written to stop a fresh optimize inheriting a stale best from a DIFFERENT model or
module. That risk is already handled -- _fullpipe_1cq_name() keys the file by (model, task). The
unlink was vestigial protection from when the file was global, and it was destroying the very thing
it was meant to protect.

So a usable bar for THIS (model, task) is kept and the BEFORE bookend is skipped entirely -- which
also saves a multi-minute full-model run. Anything not usable (absent, unparseable, non-positive) is
cleared so the run establishes a fresh one. PERF_MCP_FORCE_REBASELINE=1 restores the old behaviour.
"""

import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))


@pytest.fixture()
def run(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_MODEL_NAME", "gemma3")
    monkeypatch.setenv("PERF_MCP_TASK", "main")
    monkeypatch.delenv("PERF_MCP_FORCE_REBASELINE", raising=False)
    import models.experimental.perf_automation.cc_optimize.run as r

    importlib.reload(r)
    return r


def _bar_path(run):
    return Path(run.state_dir()) / run._fullpipe_1cq_name()


def _write_bar(run, ms, mode="trace+1cq"):
    _bar_path(run).write_text(json.dumps({"full_pipeline_ms": ms, "mode": mode, "method": "trace"}))


# ---------------------------------------------------------------- an established bar is kept


def test_an_established_bar_is_not_deleted(run):
    """The committed 33.981 that run 33 destroyed."""
    _write_bar(run, 33.981)
    run._reset_fullpipe_baselines()
    assert _bar_path(run).exists()
    assert json.loads(_bar_path(run).read_text())["full_pipeline_ms"] == 33.981


def test_the_bar_is_readable_back_after_the_reset_call(run):
    """_read_fullpipe_best_1cq is what the call site uses to skip the bookend."""
    _write_bar(run, 33.981)
    run._reset_fullpipe_baselines()
    ms, mode = run._read_fullpipe_best_1cq()
    assert ms == 33.981 and mode == "trace+1cq"


# ---------------------------------------------------------------- an unusable one is cleared


def test_no_bar_stays_no_bar(run):
    """Nothing to preserve; the run must establish one."""
    run._reset_fullpipe_baselines()
    assert not _bar_path(run).exists()
    assert (run._read_fullpipe_best_1cq() or (None, None))[0] in (None, 0, 0.0)


def test_a_corrupt_bar_is_cleared(run):
    """Keeping an unparseable file would make the run skip the bookend AND have no number."""
    _bar_path(run).write_text("{not json")
    run._reset_fullpipe_baselines()
    assert not _bar_path(run).exists()


def test_a_zero_bar_is_cleared(run):
    _write_bar(run, 0.0)
    run._reset_fullpipe_baselines()
    assert not _bar_path(run).exists()


# ---------------------------------------------------------------- the escape hatch


def test_force_rebaseline_deletes_it(run, monkeypatch):
    monkeypatch.setenv("PERF_MCP_FORCE_REBASELINE", "1")
    _write_bar(run, 33.981)
    run._reset_fullpipe_baselines()
    assert not _bar_path(run).exists()


# ---------------------------------------------------------------- the bookend is actually skipped


def test_the_before_bookend_is_skipped_when_a_bar_exists():
    """Keeping the file is only half of it -- the multi-minute BEFORE run must not happen either,
    because measuring it is what moves the bar. Asserted against the source, since exercising the
    real call site needs a device, a worktree and an agent.
    """
    src = Path(__file__).resolve().parent.parent / "cc_optimize" / "run.py"
    text = src.read_text()
    i = text.index("_reset_fullpipe_baselines()\n    #")
    block = text[i : i + 1200]
    assert "_read_fullpipe_best_1cq()" in block, "the bar is not read before measuring"
    assert "REUSED from the established" in block, "no reuse path"
    assert block.index("_read_fullpipe_best_1cq()") < block.index(
        "_fullpipe_e2e(repo_root"
    ), "the BEFORE measurement must sit on the else arm, after the reuse check"
