# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A live report must not be rewound by git_revert.

RUN_REPORT.md was written into the model directory INSIDE the optimize worktree, where git can see
it. The first commit of a run sweeps up untracked files, so the report landed in a commit beside the
lever's source change. Later commits stage only the file a lever touched, so it was never
re-committed -- permanently "modified, unstaged". git_revert discards exactly that, restoring the
committed blob.

Observed on gemma-3-12b-it:

    21:47:05  render: 30 attempts, e2e 87.93 -> 45.27 ms   (38,856 bytes)
    21:47:10  revert: 7 attempts,  e2e "after not measured yet"  (5,626 bytes)

and it stayed rewound until the next attempt re-rendered it. The measurements were never lost -- the
ledger and kernel log live in /tmp, outside git -- but the report a human reads was wrong for most of
the run, and read as a measurement error.

  g1  the git mechanism itself, on a real repo -- tracked file rewinds, ignored file does not
  g2  the report resolves into the git-ignored run directory when one exists
  g3  it falls back to the model directory when there is none
  g4  end to end: a render survives a revert
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def _summary():
    spec = importlib.util.spec_from_file_location("summary_revert", str(_PA / "cc_optimize" / "summary.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def _repo(tmp_path):
    r = tmp_path / "repo"
    (r / "models" / "demos" / "m").mkdir(parents=True)
    (r / "pkg").mkdir()
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "t@t")
    _git(r, "config", "user.name", "t")
    return r


# --------------------------------------------------------------------------- g1 THE MECHANISM
def test_g1_a_tracked_report_is_rewound_by_revert(tmp_path):
    """The defect, reproduced on a real repo: this is what the model directory does."""
    r = _repo(tmp_path)
    rep = r / "models" / "demos" / "m" / "RUN_REPORT.md"
    rep.write_text("7 attempts")
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "first commit sweeps up the untracked report")

    rep.write_text("30 attempts")  # later renders, never re-staged
    _git(r, "checkout", "--", ".")  # what git_revert does

    assert rep.read_text() == "7 attempts", "fixture must reproduce the rewind"


def test_g1_an_ignored_report_survives_revert(tmp_path):
    """The fix: git cannot restore what it never tracked."""
    r = _repo(tmp_path)
    (r / ".gitignore").write_text("pkg/runs/\n")
    runs = r / "pkg" / "runs" / "2026-07-31T18-09-23"
    runs.mkdir(parents=True)
    rep = runs / "RUN_REPORT.md"
    rep.write_text("7 attempts")
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "first commit")
    assert _git(r, "ls-files", "pkg/runs").stdout.strip() == "", "the run dir must not be tracked"

    rep.write_text("30 attempts")
    _git(r, "checkout", "--", ".")
    _git(r, "reset", "-q", "--hard")  # the harsher form, too

    assert rep.read_text() == "30 attempts", "an ignored report was still rewound"


# --------------------------------------------------------------------------- g2/g3 RESOLUTION
def test_g2_resolves_into_the_run_dir_when_one_exists(tmp_path, monkeypatch):
    m = _summary()
    runs = tmp_path / "runs"
    (runs / "latest").mkdir(parents=True)
    monkeypatch.setattr(m, "_runs_root", lambda: runs)
    p = m.report_path(tmp_path / "model")
    assert p == runs / "latest" / "RUN_REPORT.md"


def test_g3_falls_back_to_the_model_dir_without_a_run(tmp_path, monkeypatch):
    m = _summary()
    monkeypatch.setattr(m, "_runs_root", lambda: tmp_path / "nope")
    p = m.report_path(tmp_path / "model")
    assert p == tmp_path / "model" / "RUN_REPORT.md"


def test_g3_a_broken_latest_symlink_falls_back(tmp_path, monkeypatch):
    """`latest` is a symlink; a dangling one must not send the report nowhere."""
    m = _summary()
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "latest").symlink_to(runs / "gone")
    monkeypatch.setattr(m, "_runs_root", lambda: runs)
    assert m.report_path(tmp_path / "model") == tmp_path / "model" / "RUN_REPORT.md"


def test_g3_the_real_runs_dir_is_git_ignored():
    """The protection is only real if the shipped .gitignore actually covers runs/."""
    gi = _PA / ".gitignore"
    assert gi.is_file(), "perf_automation/.gitignore is missing; runs/ would be git-visible"
    assert any(ln.strip().rstrip("/") == "runs" for ln in gi.read_text().splitlines())


# --------------------------------------------------------------------------- g4 END TO END
def test_g4_a_rendered_report_survives_a_revert(tmp_path, monkeypatch):
    """The whole path: render through upsert_report_section, then revert the worktree."""
    m = _summary()
    r = _repo(tmp_path)
    (r / ".gitignore").write_text("pkg/runs/\n")
    runs = r / "pkg" / "runs"
    (runs / "latest").mkdir(parents=True)
    monkeypatch.setattr(m, "_runs_root", lambda: runs)
    model_dir = r / "models" / "demos" / "m"

    m.upsert_report_section(model_dir, "optimize", "7 lever attempts")
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "first commit")

    m.upsert_report_section(model_dir, "optimize", "30 lever attempts")
    _git(r, "checkout", "--", ".")
    _git(r, "reset", "-q", "--hard")

    written = m.report_path(model_dir)
    assert written == runs / "latest" / "RUN_REPORT.md", "report did not go to the run dir"
    assert "30 lever attempts" in written.read_text(), "the revert rewound the live report"
    assert not (model_dir / "RUN_REPORT.md").exists(), "a git-visible copy is still being written"
