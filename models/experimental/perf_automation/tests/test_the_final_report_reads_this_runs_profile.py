"""The final report must price this run, not whichever run last wrote a profile.

_read_baseline_profile_for_report rglob'd the WHOLE runs/ tree and took the newest file by mtime. A
run that writes no profile under runs/ -- voxtral 2026-09-03 wrote none; runs/latest has no
profiles/ directory at all -- then inherited a capture from 2026-08-28, six days earlier. That build
ran HiFi4 while this one had been moved to LoFi/HiFi2 by its own fidelity wins, so the final report's
fidelity ladder marked HiFi4 "in use" on all three stacks and every per-stage roof moved with it.
The LIVE report, which reads the state file, was right the whole time: one run, two reports, and the
wrong one was the final.
"""

from __future__ import annotations

import importlib.util as _ilu
import json
import sys
from pathlib import Path

PERF = Path(__file__).resolve().parents[1]
for _p in (PERF, PERF / "cc_optimize"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_spec = _ilu.spec_from_file_location("_cc_run_prof", PERF / "cc_optimize" / "run.py")
_run = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_run)


def _make_run(root: Path, name: str, marker: str, *, latest: bool = False):
    d = root / _run.PERF_DIR / "runs" / name / "profiles"
    d.mkdir(parents=True, exist_ok=True)
    (d / "baseline_profile.json").write_text(json.dumps({"marker": marker}))
    if latest:
        link = root / _run.PERF_DIR / "runs" / "latest"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(name)
    return d


def test_the_state_file_wins_over_any_run_directory(tmp_path, monkeypatch):
    """The owner is keyed by (model, task) and written by THIS run."""
    _make_run(tmp_path, "2026-08-28T20-30-39-m", "six-days-old")

    class _M:
        @staticmethod
        def _read_baseline_profile():
            return {"marker": "this-run"}

    monkeypatch.setattr(_run, "_perf_mcp", lambda: _M)
    assert _run._read_baseline_profile_for_report(tmp_path) == {"marker": "this-run"}


def test_a_foreign_run_is_never_inherited(tmp_path, monkeypatch):
    """THE DEFECT. No state, and the only profile on disk belongs to another run."""
    _make_run(tmp_path, "2026-08-28T20-30-39-m", "six-days-old")
    monkeypatch.setattr(_run, "_perf_mcp", lambda: None)
    got = _run._read_baseline_profile_for_report(tmp_path)
    assert got is None, "a stale run's capture must not stand in for this one: %r" % (got,)


def test_this_runs_own_directory_is_still_read(tmp_path, monkeypatch):
    """runs/latest names the run that is happening, so it is a safe fallback."""
    _make_run(tmp_path, "2026-08-28T20-30-39-m", "six-days-old")
    _make_run(tmp_path, "2026-09-03T10-00-00-m", "this-run", latest=True)
    monkeypatch.setattr(_run, "_perf_mcp", lambda: None)
    assert _run._read_baseline_profile_for_report(tmp_path) == {"marker": "this-run"}


def test_nothing_anywhere_reads_as_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(_run, "_perf_mcp", lambda: None)
    assert _run._read_baseline_profile_for_report(tmp_path) is None


def test_a_state_reader_that_raises_falls_through_rather_than_failing(tmp_path, monkeypatch):
    _make_run(tmp_path, "2026-09-03T10-00-00-m", "this-run", latest=True)

    class _Boom:
        @staticmethod
        def _read_baseline_profile():
            raise RuntimeError("state unreadable")

    monkeypatch.setattr(_run, "_perf_mcp", lambda: _Boom)
    assert _run._read_baseline_profile_for_report(tmp_path) == {"marker": "this-run"}


def test_the_unbounded_glob_is_gone():
    src = (PERF / "cc_optimize" / "run.py").read_text(encoding="utf-8")
    assert 'rglob("baseline_profile.json")' not in src, "a tree-wide search is how another run's capture got in"
