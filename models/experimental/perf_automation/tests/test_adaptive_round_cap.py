import importlib.util
import json
from pathlib import Path


PERF_REL = "models/experimental/perf_automation"


def _load_run():
    spec = importlib.util.spec_from_file_location(
        "cc_run", str(Path(__file__).resolve().parents[1] / "cc_optimize" / "run.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _run_dir(repo_root, name="r1"):
    d = repo_root / PERF_REL / "runs" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_manifest(run_dir, timeout=10800):
    (run_dir / "manifest.json").write_text(json.dumps({"config": {"timeout": timeout}}))


def _write_baseline(run_dir, seconds):
    lines = [
        json.dumps({"stage": "tracy_baseline", "event": "start", "seconds": None}),
        json.dumps({"stage": "tracy_baseline", "event": "done", "seconds": seconds}),
    ]
    (run_dir / "events.jsonl").write_text("\n".join(lines) + "\n")


def test_no_manifest_returns_floor(tmp_path, monkeypatch):
    m = _load_run()
    monkeypatch.delenv("PERF_MCP_ROUND_MAX_SEC", raising=False)
    assert m._round_hard_cap(tmp_path, 600) == 2400


def test_fast_model_lands_on_floor(tmp_path, monkeypatch):
    m = _load_run()
    monkeypatch.delenv("PERF_MCP_ROUND_MAX_SEC", raising=False)
    rd = _run_dir(tmp_path)
    _write_manifest(rd)
    _write_baseline(rd, 100.0)
    assert m._round_hard_cap(tmp_path, 600) == 2400


def test_heavy_model_scales_up(tmp_path, monkeypatch):
    m = _load_run()
    monkeypatch.delenv("PERF_MCP_ROUND_MAX_SEC", raising=False)
    rd = _run_dir(tmp_path)
    _write_manifest(rd, timeout=10800)
    _write_baseline(rd, 2167.92)
    assert m._round_hard_cap(tmp_path, 600) == int(3 * 2167.92)


def test_ceiling_clamps_pathological_baseline(tmp_path, monkeypatch):
    m = _load_run()
    monkeypatch.delenv("PERF_MCP_ROUND_MAX_SEC", raising=False)
    rd = _run_dir(tmp_path)
    _write_manifest(rd, timeout=10800)
    _write_baseline(rd, 9000.0)
    assert m._round_hard_cap(tmp_path, 600) == 10800


def test_manifest_timeout_is_the_ceiling(tmp_path, monkeypatch):
    m = _load_run()
    monkeypatch.delenv("PERF_MCP_ROUND_MAX_SEC", raising=False)
    rd = _run_dir(tmp_path)
    _write_manifest(rd, timeout=5000)
    _write_baseline(rd, 2167.92)
    assert m._round_hard_cap(tmp_path, 600) == 5000


def test_ceiling_never_below_floor(tmp_path, monkeypatch):
    m = _load_run()
    monkeypatch.delenv("PERF_MCP_ROUND_MAX_SEC", raising=False)
    rd = _run_dir(tmp_path)
    _write_manifest(rd, timeout=100)
    _write_baseline(rd, 100.0)
    assert m._round_hard_cap(tmp_path, 600) == 2400


def test_env_override_wins(tmp_path, monkeypatch):
    m = _load_run()
    monkeypatch.setenv("PERF_MCP_ROUND_MAX_SEC", "999")
    rd = _run_dir(tmp_path)
    _write_manifest(rd)
    _write_baseline(rd, 2167.92)
    assert m._round_hard_cap(tmp_path, 600) == 999


def test_bad_override_falls_through_to_adaptive(tmp_path, monkeypatch):
    m = _load_run()
    monkeypatch.setenv("PERF_MCP_ROUND_MAX_SEC", "not-an-int")
    rd = _run_dir(tmp_path)
    _write_manifest(rd, timeout=10800)
    _write_baseline(rd, 2167.92)
    assert m._round_hard_cap(tmp_path, 600) == int(3 * 2167.92)


def test_corrupt_events_falls_to_floor(tmp_path, monkeypatch):
    m = _load_run()
    monkeypatch.delenv("PERF_MCP_ROUND_MAX_SEC", raising=False)
    rd = _run_dir(tmp_path)
    _write_manifest(rd)
    (rd / "events.jsonl").write_text("{not json\n\n{}\n")
    assert m._round_hard_cap(tmp_path, 600) == 2400


def test_stall_scales_the_floor(tmp_path, monkeypatch):
    m = _load_run()
    monkeypatch.delenv("PERF_MCP_ROUND_MAX_SEC", raising=False)
    rd = _run_dir(tmp_path)
    _write_manifest(rd, timeout=10800)
    _write_baseline(rd, 100.0)
    assert m._round_hard_cap(tmp_path, 1000) == 4000
