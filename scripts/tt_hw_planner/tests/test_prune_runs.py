import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "optimize_under_test",
    str(Path(__file__).resolve().parents[1] / "commands" / "optimize.py"),
)
optimize = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(optimize)


def _mk_run(perf_dir, ts, mtime):
    d = perf_dir / "runs" / ts
    (d / "profiles" / "tracy_out" / ".logs").mkdir(parents=True)
    (d / "manifest.json").write_text("{}")
    (d / "profiles" / "baseline_profile.json").write_text("{}")
    (d / "profiles" / "iter_baseline_report.csv").write_text("x")
    (d / "profiles" / "tracy_out" / ".logs" / "profile_log_device.csv").write_text("y" * 1000)
    import os

    os.utime(d, (mtime, mtime))
    return d


def test_keeps_recent_full_strips_middle_deletes_oldest(tmp_path, monkeypatch):
    monkeypatch.setattr(optimize, "_RUNS_KEEP_FULL", 2)
    monkeypatch.setattr(optimize, "_RUNS_KEEP_TOTAL", 4)
    perf = tmp_path / "perf"
    dirs = [_mk_run(perf, "2026-07-21T%02d-00-00" % i, 1000 + i) for i in range(6)]

    optimize._prune_runs(perf)

    assert (dirs[5] / "profiles" / "tracy_out").is_dir()
    assert (dirs[4] / "profiles" / "tracy_out").is_dir()
    assert not (dirs[3] / "profiles" / "tracy_out").exists()
    assert (dirs[3] / "manifest.json").is_file()
    assert (dirs[2] / "manifest.json").is_file()
    assert not dirs[1].exists()
    assert not dirs[0].exists()


def test_preserves_reader_inputs_for_stripped_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(optimize, "_RUNS_KEEP_FULL", 1)
    monkeypatch.setattr(optimize, "_RUNS_KEEP_TOTAL", 10)
    perf = tmp_path / "perf"
    dirs = [_mk_run(perf, "2026-07-21T%02d-00-00" % i, 1000 + i) for i in range(3)]

    optimize._prune_runs(perf)

    for d in dirs:
        assert (d / "manifest.json").is_file()
        assert (d / "profiles" / "baseline_profile.json").is_file()
        assert (d / "profiles" / "iter_baseline_report.csv").is_file()


def test_noop_when_no_runs_dir(tmp_path):
    optimize._prune_runs(tmp_path / "perf")
