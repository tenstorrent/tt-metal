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


def test_no_manifest_cold_start_is_bounded(tmp_path, monkeypatch):
    """CONTRACT CHANGED 2026-07-25 (BUG 4): no history -> bounded cold-start value, not the
    old absolute 2400 s floor which was simultaneously 400x a micro module's work and too
    tight for llama's round (killed 4x at exactly 2400 s with nothing wrong)."""
    m = _load_run()
    monkeypatch.delenv("PERF_MCP_ROUND_MAX_SEC", raising=False)
    v = m._round_hard_cap(tmp_path, 600)
    assert 30 <= v <= 10800


def test_fast_model_gets_a_small_budget(tmp_path, monkeypatch):
    m = _load_run()
    monkeypatch.delenv("PERF_MCP_ROUND_MAX_SEC", raising=False)
    rd = _run_dir(tmp_path)
    _write_manifest(rd)
    # 100 s was never a "fast" baseline (llama's is 146.7 s); use a genuinely fast module,
    # which is the case the old absolute floor served worst (3600 s for 13 s of work).
    _write_baseline(rd, 3.16)
    # BUG 4: a fast model must get a SMALL budget, not the old 2400 s floor.
    v = m._round_hard_cap(tmp_path, 600)
    assert 30 <= v < 2400


def test_heavy_model_scales_up(tmp_path, monkeypatch):
    m = _load_run()
    monkeypatch.delenv("PERF_MCP_ROUND_MAX_SEC", raising=False)
    rd = _run_dir(tmp_path)
    _write_manifest(rd, timeout=10800)
    _write_baseline(rd, 2167.92)
    # BUG 4: derived from the observed ROUND cycle (not 3x a profile), clamped by the ceiling.
    assert m._round_hard_cap(tmp_path, 600) == 10800


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


def test_ceiling_always_clamps(tmp_path, monkeypatch):
    m = _load_run()
    monkeypatch.delenv("PERF_MCP_ROUND_MAX_SEC", raising=False)
    rd = _run_dir(tmp_path)
    _write_manifest(rd, timeout=100)
    _write_baseline(rd, 100.0)
    # BUG 4: ceiling below the old floor must still clamp; no absolute floor may exceed it.
    assert m._round_hard_cap(tmp_path, 600) <= 100


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
    # BUG 4: a bad override falls through to the derived budget (ceiling-clamped here).
    assert m._round_hard_cap(tmp_path, 600) == 10800


def test_corrupt_events_behaves_like_cold_start(tmp_path, monkeypatch):
    m = _load_run()
    monkeypatch.delenv("PERF_MCP_ROUND_MAX_SEC", raising=False)
    rd = _run_dir(tmp_path)
    _write_manifest(rd)
    (rd / "events.jsonl").write_text("{not json\n\n{}\n")
    # BUG 4: unreadable history behaves like cold start -> bounded, not a 2400 s floor.
    v = m._round_hard_cap(tmp_path, 600)
    assert 30 <= v <= 10800


def test_cap_no_longer_derives_from_stall(tmp_path, monkeypatch):
    m = _load_run()
    monkeypatch.delenv("PERF_MCP_ROUND_MAX_SEC", raising=False)
    rd = _run_dir(tmp_path)
    _write_manifest(rd, timeout=10800)
    _write_baseline(rd, 100.0)
    # BUG 4: the cap no longer derives from stall_sec at all -- it derives from the observed
    # round cycle, which is the whole point (a round is not 4x a FROZEN threshold).
    assert m._round_hard_cap(tmp_path, 1000) >= 30
