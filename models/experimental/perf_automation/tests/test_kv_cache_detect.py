import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cc_optimize import perf_mcp as m


def test_scaling_recompute_when_time_scales_with_capacity():
    assert m.kv_cache_needed_by_scaling(10.0, 20.0) is True
    assert m.kv_cache_needed_by_scaling(10.0, 16.0) is True


def test_scaling_kv_present_when_flat():
    assert m.kv_cache_needed_by_scaling(10.0, 10.5) is False
    assert m.kv_cache_needed_by_scaling(10.0, 11.0) is False


def test_scaling_none_on_bad_input():
    assert m.kv_cache_needed_by_scaling(0.0, 5.0) is None
    assert m.kv_cache_needed_by_scaling("x", 5.0) is None
    assert m.kv_cache_needed_by_scaling(None, None) is None


def _mk_pipeline(tmp_path, body):
    demo = tmp_path / "d"
    (demo / "tt").mkdir(parents=True)
    (demo / "tt" / "pipeline.py").write_text(body)
    return demo


def test_recompute_detected_when_no_kv(tmp_path):
    demo = _mk_pipeline(
        tmp_path,
        "def decode_step(self):\n    return self.model(h, use_cache=False, past_key_value=None)\n",
    )
    assert m._decode_is_recompute(demo) is True


def test_recompute_false_when_kv_write_present(tmp_path):
    demo = _mk_pipeline(
        tmp_path,
        "def decode_step(self):\n    self.update_cache(k, v)\n    return self.model(h, use_cache=False)\n",
    )
    assert m._decode_is_recompute(demo) is False


def test_recompute_false_when_no_pipeline(tmp_path):
    assert m._decode_is_recompute(tmp_path / "missing") is False


def test_decode_gate_fires_on_recompute_even_when_traced(tmp_path, monkeypatch):
    monkeypatch.delenv("TT_PERF_MODULE_LEVEL", raising=False)
    monkeypatch.setattr(m, "_decode_is_recompute", lambda root: True)
    prof = {"decode_status": "traced", "per_token_ms": 5.0}
    out = m._decode_gate(prof, [])
    assert out is not None
    assert out["next_rung"] == "structural-decode"
    assert "recompute" in out["reason"]


def test_decode_gate_none_when_traced_and_kv_present(tmp_path, monkeypatch):
    monkeypatch.delenv("TT_PERF_MODULE_LEVEL", raising=False)
    monkeypatch.setattr(m, "_decode_is_recompute", lambda root: False)
    prof = {"decode_status": "traced", "per_token_ms": 5.0}
    assert m._decode_gate(prof, []) is None


def test_decode_gate_still_fires_on_repeat_prefill(tmp_path, monkeypatch):
    monkeypatch.delenv("TT_PERF_MODULE_LEVEL", raising=False)
    monkeypatch.setattr(m, "_decode_is_recompute", lambda root: False)
    prof = {"decode_status": "repeat_prefill", "per_token_ms": 5.0}
    out = m._decode_gate(prof, [])
    assert out is not None and out["next_rung"] == "structural-decode"


def test_decode_gate_scaling_overrides_static(tmp_path, monkeypatch):
    monkeypatch.delenv("TT_PERF_MODULE_LEVEL", raising=False)
    monkeypatch.setattr(m, "_decode_is_recompute", lambda root: True)
    prof = {"decode_status": "traced", "decode_ms_at_c": 10.0, "decode_ms_at_2c": 10.2, "per_token_ms": 5.0}
    assert m._decode_gate(prof, []) is None
