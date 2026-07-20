import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cc_optimize import perf_mcp as m


def _mk_model(tmp_path, body="import ttnn\n"):
    root = tmp_path / "model"
    (root / "_stubs").mkdir(parents=True)
    (root / "tt").mkdir(parents=True)
    (root / "_stubs" / "a.py").write_text(body)
    (root / "tt" / "pipeline.py").write_text("x = 1\n")
    return root


def _wire(monkeypatch, tmp_path, model_root):
    calls = {"n": 0}

    def fake_measure_runs(ctx):
        calls["n"] += 1
        return [{"device_ms": float(calls["n"])}]

    monkeypatch.setattr(m, "_MODEL_ROOT", model_root)
    monkeypatch.setattr(m, "_PROFILE_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(m, "measure_runs", fake_measure_runs)
    monkeypatch.setattr(
        m, "_Ctx", lambda: types.SimpleNamespace(run=types.SimpleNamespace(dir="/tmp", profiles_dir="/tmp"))
    )
    monkeypatch.setattr(m.roofline, "annotate_profile", lambda p, e: p)
    monkeypatch.setattr(m, "_detect_partial_capture", lambda d: None)
    monkeypatch.setattr(m, "_persist_artifacts", lambda p: p)
    monkeypatch.setattr(m, "_reap_measurement_dir", lambda d: None)
    monkeypatch.delenv("PERF_MCP_NO_PROFILE_CACHE", raising=False)
    return calls


def test_fingerprint_stable_then_changes(tmp_path, monkeypatch):
    root = _mk_model(tmp_path)
    monkeypatch.setattr(m, "_MODEL_ROOT", root)
    fp1 = m._model_source_fingerprint()
    fp2 = m._model_source_fingerprint()
    assert fp1 and fp1 == fp2
    (root / "_stubs" / "a.py").write_text("import ttnn\nx = 2\n")
    fp3 = m._model_source_fingerprint()
    assert fp3 != fp1


def test_fingerprint_empty_when_no_source(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "_MODEL_ROOT", tmp_path / "nope")
    assert m._model_source_fingerprint() == ""


def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "_PROFILE_CACHE_DIR", tmp_path / "c")
    assert m._profile_cache_get("fp") is None
    m._profile_cache_put("fp", {"device_ms": 5.0})
    assert m._profile_cache_get("fp") == {"device_ms": 5.0}


def test_profile_once_dedups_unchanged_model(tmp_path, monkeypatch):
    root = _mk_model(tmp_path)
    calls = _wire(monkeypatch, tmp_path, root)
    p1 = m._profile_once(cq=2)
    p2 = m._profile_once(cq=1)
    assert calls["n"] == 1
    assert p1["device_ms"] == p2["device_ms"] == 1.0


def test_profile_once_reprofiles_after_source_change(tmp_path, monkeypatch):
    root = _mk_model(tmp_path)
    calls = _wire(monkeypatch, tmp_path, root)
    m._profile_once(cq=1)
    assert calls["n"] == 1
    (root / "_stubs" / "a.py").write_text("import ttnn\nx = 99\n")
    m._profile_once(cq=1)
    assert calls["n"] == 2


def test_profile_once_no_cache_when_disabled(tmp_path, monkeypatch):
    root = _mk_model(tmp_path)
    calls = _wire(monkeypatch, tmp_path, root)
    monkeypatch.setenv("PERF_MCP_NO_PROFILE_CACHE", "1")
    m._profile_once(cq=1)
    m._profile_once(cq=1)
    assert calls["n"] == 2
