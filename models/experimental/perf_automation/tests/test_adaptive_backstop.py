import importlib.util
import json
from pathlib import Path

from agent.probes import adaptive_backstop

PERF_REL = "models/experimental/perf_automation"


def _load_run():
    spec = importlib.util.spec_from_file_location(
        "cc_run", str(Path(__file__).resolve().parents[1] / "cc_optimize" / "run.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write_run(dirpath, timeout=10800, baseline=None):
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "manifest.json").write_text(json.dumps({"config": {"timeout": timeout}}))
    if baseline is not None:
        (dirpath / "events.jsonl").write_text(
            json.dumps({"stage": "tracy_baseline", "event": "done", "seconds": baseline}) + "\n"
        )


def test_probes_no_manifest_returns_floor(monkeypatch):
    monkeypatch.delenv("PERF_MCP_MANIFEST", raising=False)
    monkeypatch.delenv("PERF_MCP_MEASURE_BACKSTOP", raising=False)
    assert adaptive_backstop(3600) == 3600


def test_probes_fast_model_floor(tmp_path, monkeypatch):
    monkeypatch.delenv("PERF_MCP_MEASURE_BACKSTOP", raising=False)
    rd = tmp_path / "runs" / "r1"
    _write_run(rd, baseline=100.0)
    monkeypatch.setenv("PERF_MCP_MANIFEST", str(rd / "manifest.json"))
    assert adaptive_backstop(3600) == 3600


def test_probes_heavy_model_scales(tmp_path, monkeypatch):
    monkeypatch.delenv("PERF_MCP_MEASURE_BACKSTOP", raising=False)
    rd = tmp_path / "runs" / "r1"
    _write_run(rd, timeout=10800, baseline=2167.92)
    monkeypatch.setenv("PERF_MCP_MANIFEST", str(rd / "manifest.json"))
    assert adaptive_backstop(3600) == int(3 * 2167.92)


def test_probes_ceiling_clamps(tmp_path, monkeypatch):
    monkeypatch.delenv("PERF_MCP_MEASURE_BACKSTOP", raising=False)
    rd = tmp_path / "runs" / "r1"
    _write_run(rd, timeout=10800, baseline=9000.0)
    monkeypatch.setenv("PERF_MCP_MANIFEST", str(rd / "manifest.json"))
    assert adaptive_backstop(3600) == 10800


def test_probes_manifest_timeout_is_ceiling(tmp_path, monkeypatch):
    monkeypatch.delenv("PERF_MCP_MEASURE_BACKSTOP", raising=False)
    rd = tmp_path / "runs" / "r1"
    _write_run(rd, timeout=5000, baseline=2167.92)
    monkeypatch.setenv("PERF_MCP_MANIFEST", str(rd / "manifest.json"))
    assert adaptive_backstop(3600) == 5000


def test_probes_env_override_wins(tmp_path, monkeypatch):
    rd = tmp_path / "runs" / "r1"
    _write_run(rd, baseline=2167.92)
    monkeypatch.setenv("PERF_MCP_MANIFEST", str(rd / "manifest.json"))
    monkeypatch.setenv("PERF_MCP_MEASURE_BACKSTOP", "1234")
    assert adaptive_backstop(3600) == 1234


def test_probes_corrupt_events_floor(tmp_path, monkeypatch):
    monkeypatch.delenv("PERF_MCP_MEASURE_BACKSTOP", raising=False)
    rd = tmp_path / "runs" / "r1"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "manifest.json").write_text(json.dumps({"config": {"timeout": 10800}}))
    (rd / "events.jsonl").write_text("{bad\n\n{}\n")
    monkeypatch.setenv("PERF_MCP_MANIFEST", str(rd / "manifest.json"))
    assert adaptive_backstop(3600) == 3600


def test_run_measure_backstop_floor(tmp_path, monkeypatch):
    m = _load_run()
    monkeypatch.delenv("PERF_MCP_MEASURE_BACKSTOP", raising=False)
    assert m._measure_backstop(tmp_path) == 3600


def test_run_measure_backstop_scales(tmp_path, monkeypatch):
    m = _load_run()
    monkeypatch.delenv("PERF_MCP_MEASURE_BACKSTOP", raising=False)
    rd = tmp_path / PERF_REL / "runs" / "r1"
    _write_run(rd, timeout=10800, baseline=2167.92)
    assert m._measure_backstop(tmp_path) == int(3 * 2167.92)


def test_run_measure_backstop_ceiling(tmp_path, monkeypatch):
    m = _load_run()
    monkeypatch.delenv("PERF_MCP_MEASURE_BACKSTOP", raising=False)
    rd = tmp_path / PERF_REL / "runs" / "r1"
    _write_run(rd, timeout=10800, baseline=9000.0)
    assert m._measure_backstop(tmp_path) == 10800


def test_run_measure_backstop_env_override(tmp_path, monkeypatch):
    m = _load_run()
    monkeypatch.setenv("PERF_MCP_MEASURE_BACKSTOP", "1500")
    rd = tmp_path / PERF_REL / "runs" / "r1"
    _write_run(rd, baseline=2167.92)
    assert m._measure_backstop(tmp_path) == 1500


def test_run_round_cap_and_backstop_share_reader(tmp_path, monkeypatch):
    m = _load_run()
    monkeypatch.delenv("PERF_MCP_MEASURE_BACKSTOP", raising=False)
    monkeypatch.delenv("PERF_MCP_ROUND_MAX_SEC", raising=False)
    rd = tmp_path / PERF_REL / "runs" / "r1"
    _write_run(rd, timeout=10800, baseline=2167.92)
    assert m._round_hard_cap(tmp_path, 600) == int(3 * 2167.92)
    assert m._measure_backstop(tmp_path) == int(3 * 2167.92)
