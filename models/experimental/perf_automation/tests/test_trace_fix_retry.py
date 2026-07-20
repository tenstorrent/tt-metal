import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "perf_mcp_tracefix",
    str(Path(__file__).resolve().parents[1] / "cc_optimize" / "perf_mcp.py"),
)
perf_mcp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(perf_mcp)
ladder = perf_mcp._op_ladder_status
NT = perf_mcp._MAX_TRACE_FIX_RETRIES


def _op(grid="full", wdtype="bf8_b", bound="memory"):
    return {"grid": grid, "weight_dtype": wdtype, "bound_by": bound}


def _att(kind, n=1, wedged=False):
    return [{"op_signature": "MatmulDeviceOperation", "kernel_kind": kind, "wedged": wedged} for _ in range(n)]


def test_rung_state_clean_is_done():
    done, wedged = perf_mcp._rung_state(_att("tt-lang", 1, wedged=False), "tt-lang", NT)
    assert done and wedged == 0


def test_rung_state_wedged_below_budget_not_done():
    done, wedged = perf_mcp._rung_state(_att("tt-lang", 1, wedged=True), "tt-lang", NT)
    assert (not done) and wedged == 1


def test_rung_state_wedged_at_budget_is_done():
    done, wedged = perf_mcp._rung_state(_att("tt-lang", NT, wedged=True), "tt-lang", NT)
    assert done and wedged == NT


def test_wedged_ttlang_keeps_rung_open_with_fix_feedback(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_ttl_available", lambda: True)
    done, rung, reason = ladder(_op(), "MatmulDeviceOperation", _att("tt-lang", 1, wedged=True))
    assert (not done) and rung == "tt-lang"
    assert ("trace-fix 1/%d" % NT) in reason and "trace-safe" in reason


def test_wedged_ttlang_advances_after_budget(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_ttl_available", lambda: True)
    _, rung, _ = ladder(_op(), "MatmulDeviceOperation", _att("tt-lang", NT, wedged=True))
    assert rung != "tt-lang"


def test_clean_ttlang_advances_immediately(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_ttl_available", lambda: True)
    _, rung, _ = ladder(_op(), "MatmulDeviceOperation", _att("tt-lang", 1, wedged=False))
    assert rung != "tt-lang"


def test_wedged_cpp_keeps_rung_open_with_fix_feedback(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_ttl_available", lambda: True)
    atts = _att("tt-lang", 1, wedged=False) + _att("cpp", 1, wedged=True)
    done, rung, reason = ladder(_op(), "MatmulDeviceOperation", atts)
    assert (not done) and rung == "cpp"
    assert ("trace-fix 1/%d" % NT) in reason


def test_trace_compat_feedback_enriches_custom_rung(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_load_target", lambda: {"rung": "tt-lang"})
    out = perf_mcp._trace_compat_feedback("boom")
    assert "boom" in out and "trace-safe" in out


def test_trace_compat_feedback_passthrough_for_knob(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_load_target", lambda: {"rung": "grid"})
    assert perf_mcp._trace_compat_feedback("boom") == "boom"
