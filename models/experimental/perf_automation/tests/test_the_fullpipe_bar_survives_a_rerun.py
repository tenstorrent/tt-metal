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


def _write_bar(run, ms, mode="trace+1cq", sha=None):
    doc = {"full_pipeline_ms": ms, "mode": mode, "method": "trace"}
    if sha is not None:
        doc["sha"] = sha
    _bar_path(run).write_text(json.dumps(doc))


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
    i = text.index("_reset_fullpipe_baselines(start_sha)\n    #")
    block = text[i : i + 1200]
    assert "_read_fullpipe_best_1cq()" in block, "the bar is not read before measuring"
    assert "REUSED from the established" in block, "no reuse path"
    assert block.index("_read_fullpipe_best_1cq()") < block.index(
        "_fullpipe_e2e(repo_root"
    ), "the BEFORE measurement must sit on the else arm, after the reuse check"


# ---------------------------------------------------------------- a bar from another tree is not a bar


def test_a_bar_measured_at_the_commit_the_run_starts_from_is_kept(run):
    """The plain --persist relaunch after a run: HEAD is the last win, the bar was measured there."""
    _write_bar(run, 89.3623, sha="9048d7eeab15a8a57ea1224980c2923df31b9c4a")
    run._reset_fullpipe_baselines("9048d7eeab15a8a57ea1224980c2923df31b9c4a")
    assert _bar_path(run).exists()


def test_a_bar_measured_on_a_different_tree_is_dropped_and_said(run, capsys):
    """Voxtral 2026-10-05: two levers undone by hand and tt-metal updated; the code measured 90.6 ms
    against a stored best of 89.36 measured on the old tree. Same launch command, no env var."""
    _write_bar(run, 89.3623, sha="9048d7eeab15a8a57ea1224980c2923df31b9c4a")
    run._reset_fullpipe_baselines("596f9d7664e0000000000000000000000000000000")
    assert not _bar_path(run).exists()
    out = capsys.readouterr().out
    assert "different tree" in out and "9048d7eeab1" in out and "596f9d7664e" in out
    assert "nothing else the run remembers is touched" in out


def test_a_bar_without_a_recorded_commit_is_kept(run):
    """Written before the stamp existed: nothing to compare, so the old rule applies."""
    _write_bar(run, 33.981)
    run._reset_fullpipe_baselines("596f9d7664e0000000000000000000000000000000")
    assert _bar_path(run).exists()


def test_an_unreadable_head_keeps_the_bar(run):
    """If git cannot say where the run starts, keeping the bar is the conservative choice, as before."""
    _write_bar(run, 33.981, sha="9048d7eeab15a8a57ea1224980c2923df31b9c4a")
    run._reset_fullpipe_baselines("")
    assert _bar_path(run).exists()


def test_the_call_site_hands_the_reset_the_commit_the_run_starts_at():
    src = (Path(__file__).resolve().parent.parent / "cc_optimize" / "run.py").read_text()
    assert 'start_sha = _git(repo_root, "rev-parse", "HEAD")' in src
    assert "_reset_fullpipe_baselines(start_sha)" in src


# ---------------------------------------------------------------- a pending reading from another tree is not a reading


def _pending_path(run):
    return _bar_path(run).with_suffix(".pending.json")


def _write_pending(run, ms, sha, stages=None):
    _pending_path(run).write_text(
        json.dumps({"full_pipeline_ms": ms, "mode": "trace+1cq", "method": "trace", "sha": sha, "stages": stages or {}})
    )


_OLD = "024f03283e0aea5a844cc0c8fb83c9116517ff01"
_NEW = "cb33ed02ef5000000000000000000000000000000"


def test_a_pending_reading_from_another_tree_is_dropped_with_the_bar(run, capsys):
    """Devstral 2026-10-22: run 7's last reading (prefill 933 ms) sat pending at 024f03283e0 when the
    image e2e fix moved HEAD and prefill to 1293 ms. Both files describe code that is gone."""
    _write_bar(run, 13.017, sha="f279226bc0c53a882db2641254a91301bc68aeae")
    _write_pending(run, 13.0155, _OLD, {"prefill": 933.143})
    run._reset_fullpipe_baselines(_NEW)
    assert not _bar_path(run).exists()
    assert not _pending_path(run).exists()
    out = capsys.readouterr().out
    assert "pending full-pipeline reading dropped" in out and "024f03283e0" in out and "cb33ed02ef5" in out


def test_a_pending_reading_at_the_commit_the_run_starts_from_is_kept(run):
    """The plain --persist relaunch: the reading was taken on this very tree; a commit will promote it."""
    _write_bar(run, 13.017, sha=_NEW)
    _write_pending(run, 13.0155, _NEW)
    run._reset_fullpipe_baselines(_NEW)
    assert _bar_path(run).exists() and _pending_path(run).exists()


def test_a_pending_reading_without_a_stamp_or_without_a_head_is_kept(run):
    """Nothing to compare: the conservative choice, as for the bar."""
    _write_pending(run, 13.0155, "")
    run._reset_fullpipe_baselines(_NEW)
    assert _pending_path(run).exists()
    _write_pending(run, 13.0155, _OLD)
    run._reset_fullpipe_baselines("")
    assert _pending_path(run).exists()


def test_force_rebaseline_drops_the_pending_reading_too(run, monkeypatch):
    monkeypatch.setenv("PERF_MCP_FORCE_REBASELINE", "1")
    _write_pending(run, 13.0155, _NEW)
    run._reset_fullpipe_baselines(_NEW)
    assert not _pending_path(run).exists()


def test_perf_mcp_cannot_promote_a_stale_pending_reading_after_the_reset(run, monkeypatch):
    """The leak, end to end: perf_mcp promotes a pending reading whenever HEAD differs from its stamp,
    which a relaunch from a new commit satisfies before measuring anything. Without the reset the
    933 ms reading becomes the bar; with it there is nothing left to promote."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "perf_mcp_pending_leak", str(Path(__file__).resolve().parents[1] / "cc_optimize" / "perf_mcp.py")
    )
    perf_mcp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(perf_mcp)
    monkeypatch.setattr(perf_mcp, "_FULLPIPE_BASELINE_1CQ_PATH", _bar_path(run))
    monkeypatch.setattr(perf_mcp, "_head_sha_quiet", lambda: _NEW)
    assert perf_mcp._fullpipe_pending_path() == _pending_path(run), "run.py and perf_mcp must name the same file"

    _write_pending(run, 13.0155, _OLD, {"prefill": 933.143})
    assert perf_mcp._promote_fullpipe_if_committed() is True, "the leak this reset guards against"
    assert json.loads(_bar_path(run).read_text())["stages"]["prefill"] == 933.143
    _bar_path(run).unlink()

    _write_pending(run, 13.0155, _OLD, {"prefill": 933.143})
    run._reset_fullpipe_baselines(_NEW)
    assert perf_mcp._promote_fullpipe_if_committed() is False
    assert not _bar_path(run).exists()
