# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""--fresh forgot the one file that decides what gets measured.

WHAT IT COST, 2026-08-15. A --fresh run cleared 20 items and then REFUSED TO START:

    [optimize/cc] --fresh: cleared 20 item(s) of run memory:
    ...
    [optimize/cc] preflight FAILED
      FAILED tests/test_the_workload_comes_from_the_model.py::test_the_real_generated_test_is_caught
    [optimize/cc] refusing to start against a tool whose own tests fail.

The check reads the generated perf test on disk. --fresh had removed its `.trace_caps.json` sidecar
and left the test itself, so the run inherited the PREVIOUS run's generated workload -- declaring
TT_PERF_FLUSH_EVERY where the tool expected TT_PERF_AUDIO_STREAMS -- from a run that had asked to
forget everything. Clearing the sidecar but keeping the test is a half-measure: the test is what
defines the measurement.

TRACKED-ONLY-IF-UNTRACKED IS THE SUBTLETY. These same filenames were COMMITTED model source on the
older lineage. Deleting them there would break this module's one promise -- nothing tracked by git
is touched -- and fight the run's own reset step, which restores tracked files. On a branch that
generates them they are run output like any other, and regenerating costs only what it cost to write
them the first time.
"""

import subprocess
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def _repo(tmp_path, tracked=(), untracked=()):
    """A model dir inside a real git repo, so the tracked/untracked question has a real answer."""
    root = tmp_path / "repo"
    (root / "tests" / "e2e").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)
    for n in tracked:
        (root / "tests" / "e2e" / n).write_text("# tracked\n")
    if tracked:
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-qm", "init"], cwd=root, check=True)
    for n in untracked:
        (root / "tests" / "e2e" / n).write_text("# generated\n")
    return root


def test_a_generated_perf_test_is_cleared(tmp_path):
    """THE BUG: it survived --fresh and decided the next run's workload."""
    from agent import fresh_start

    root = _repo(tmp_path, untracked=["test_main_perf.py"])
    names = [p.name for p in fresh_start.plan(None, model_dir=root)]
    assert "test_main_perf.py" in names


def test_a_tracked_perf_test_is_never_touched(tmp_path):
    """The older lineage COMMITTED these. Deleting model source is not --fresh's job."""
    from agent import fresh_start

    root = _repo(tmp_path, tracked=["test_main_perf.py"])
    names = [p.name for p in fresh_start.plan(None, model_dir=root)]
    assert "test_main_perf.py" not in names


def test_a_mixed_directory_is_split_correctly(tmp_path):
    from agent import fresh_start

    root = _repo(tmp_path, tracked=["test_e2e_pipeline_perf.py"], untracked=["test_main_perf.py"])
    names = [p.name for p in fresh_start.plan(None, model_dir=root)]
    assert "test_main_perf.py" in names
    assert "test_e2e_pipeline_perf.py" not in names


def test_the_sidecar_still_goes_too(tmp_path):
    from agent import fresh_start

    root = _repo(tmp_path, untracked=["test_main_perf.py"])
    (root / "tests" / "e2e" / "test_main_perf.py.trace_caps.json").write_text("{}")
    names = [p.name for p in fresh_start.plan(None, model_dir=root)]
    assert "test_main_perf.py.trace_caps.json" in names


def test_a_non_repo_keeps_everything(tmp_path):
    """No git to ask means no way to tell generated from shipped. Refusing to delete is the
    recoverable mistake; deleting someone's model source is not."""
    from agent import fresh_start

    root = tmp_path / "plain"
    (root / "tests" / "e2e").mkdir(parents=True)
    (root / "tests" / "e2e" / "test_main_perf.py").write_text("# who knows\n")
    names = [p.name for p in fresh_start.plan(None, model_dir=root)]
    assert "test_main_perf.py" not in names


def test_wipe_actually_removes_it(tmp_path):
    from agent import fresh_start

    root = _repo(tmp_path, untracked=["test_main_perf.py"])
    f = root / "tests" / "e2e" / "test_main_perf.py"
    fresh_start.wipe(None, model_dir=root)
    assert not f.exists(), "--fresh planned the removal but did not perform it"
