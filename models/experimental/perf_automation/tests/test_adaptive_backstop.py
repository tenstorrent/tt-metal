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


def test_run_measure_backstop_cold_start_is_bounded(tmp_path, monkeypatch):
    """CONTRACT CHANGED 2026-07-25 (BUG 4): with no history there is nothing to scale from,
    so a bounded cold-start value is used instead of the old absolute 3600 s floor. The old
    floor made a 3 ms module wait an hour before a hang was noticed (1139x its real work)."""
    m = _load_run()
    monkeypatch.delenv("PERF_MCP_MEASURE_BACKSTOP", raising=False)
    v = m._measure_backstop(tmp_path)
    assert 30 <= v <= 3600


def test_run_measure_backstop_scales_with_observed_cost(tmp_path, monkeypatch):
    """CONTRACT CHANGED 2026-07-25 (BUG 4): the budget is a multiple of the OBSERVED cost of
    the operation it governs (profile), clamped by the operator ceiling -- not a fixed 3x of
    a proxy. It must scale with the model and stay inside the ceiling."""
    m = _load_run()
    monkeypatch.delenv("PERF_MCP_MEASURE_BACKSTOP", raising=False)
    rd = tmp_path / PERF_REL / "runs" / "r1"
    _write_run(rd, timeout=10800, baseline=2167.92)
    big = m._measure_backstop(tmp_path)
    rd2 = tmp_path / "small" / PERF_REL / "runs" / "r1"
    _write_run(rd2, timeout=10800, baseline=3.16)
    small = m._measure_backstop(tmp_path / "small")
    assert big > small, "a slow model must get a bigger budget than a 3 s module"
    assert big <= 10800 and small >= 30


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
    # CONTRACT CHANGED 2026-07-25 (BUG 4): round cap and measure backstop now derive from
    # DIFFERENT observed operations (a round cycle vs one profile), so they are no longer
    # required to be equal -- that equality was the bug (a round budgeted from a profile).
    assert m._round_hard_cap(tmp_path, 600) >= m._measure_backstop(tmp_path)
