"""I-4 tests: run dir + write-once manifest (PLAN section 2 & 5)."""

import pytest

from agent.run import ManifestExistsError, Run


def test_new_run_creates_dirs_and_latest_symlink(tmp_path):
    runs_root = tmp_path / "runs"
    run = Run.create(runs_root, run_id="2026-06-09T14-22")

    assert run.dir.is_dir()
    assert run.profiles_dir.is_dir()

    latest = runs_root / "latest"
    assert latest.is_symlink()
    assert latest.resolve() == run.dir.resolve()

    # latest() resolves back to the same run.
    assert Run.latest(runs_root).run_id == "2026-06-09T14-22"


def test_latest_repoints_on_new_run(tmp_path):
    runs_root = tmp_path / "runs"
    Run.create(runs_root, run_id="run-a")
    Run.create(runs_root, run_id="run-b")
    assert Run.latest(runs_root).run_id == "run-b"


def test_manifest_write_once(tmp_path):
    runs_root = tmp_path / "runs"
    run = Run.create(runs_root, config={"target_ms": 12.0}, run_id="r1")
    assert run.manifest.read()["target_ms"] == 12.0
    # Second write is rejected.
    with pytest.raises(ManifestExistsError):
        run.manifest.write({"target_ms": 99.0})
    # Original preserved.
    assert run.manifest.read()["target_ms"] == 12.0


def test_latest_none_when_absent(tmp_path):
    assert Run.latest(tmp_path / "runs") is None
