import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "perf_mcp_tracefix",
    str(Path(__file__).resolve().parents[1] / "cc_optimize" / "perf_mcp.py"),
)
perf_mcp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(perf_mcp)
ladder = perf_mcp._op_ladder_status


def _op(grid="full", wdtype="bf8_b", bound="memory"):
    return {"grid": grid, "weight_dtype": wdtype, "bound_by": bound}


def _att(kind, n=1, wedged=False):
    return [{"op_signature": "MatmulDeviceOperation", "kernel_kind": kind, "wedged": wedged} for _ in range(n)]


def test_rung_state_clean_true_when_measured():
    clean, wedged = perf_mcp._rung_state(_att("tt-lang", 1, wedged=False), "tt-lang")
    assert clean and wedged == 0


def test_rung_state_wedged_counts_and_not_clean():
    clean, wedged = perf_mcp._rung_state(_att("tt-lang", 2, wedged=True), "tt-lang")
    assert (not clean) and wedged == 2


def test_wedged_ttlang_keeps_rung_open_with_fix_feedback(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_ttl_available", lambda: True)
    done, rung, reason = ladder(_op(), "MatmulDeviceOperation", _att("tt-lang", 1, wedged=True))
    assert (not done) and rung == "tt-lang"
    assert "trace-fix attempt 2" in reason
    assert "override_runtime_args" in reason and "ISOLATION" in reason and "cache+reuse" in reason
    assert "do NOT switch to cpp" in reason
    assert "HANG or TIMEOUT" in reason and "no attempt limit" in reason


def test_author_reason_instructs_isolation_first(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_ttl_available", lambda: True)
    _, rung, reason = ladder(_op(), "MatmulDeviceOperation", [])
    assert rung == "tt-lang"
    assert "ISOLATION" in reason and "EAGER first" in reason


def test_feedback_instructs_eager_first_and_stage_capture(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_ttl_available", lambda: True)
    _, _, author = ladder(_op(), "MatmulDeviceOperation", [])
    assert "EAGER first" in author and "BEGIN_CAPTURE" in author
    assert "localizes to the LAST printed stage" in author
    _, _, wedge = ladder(_op(), "MatmulDeviceOperation", _att("tt-lang", 1, wedged=True))
    assert "EAGER first" in wedge and "RUN_OP" in wedge


def test_wedged_ttlang_holds_indefinitely_never_cpp(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_ttl_available", lambda: True)
    for n in (1, 3, 10, 50):
        _, rung, _ = ladder(_op(), "MatmulDeviceOperation", _att("tt-lang", n, wedged=True))
        assert rung == "tt-lang", "with %d wedges expected tt-lang held (no budget), got %s" % (n, rung)


def test_clean_ttlang_advances_to_cpp(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_ttl_available", lambda: True)
    _, rung, _ = ladder(_op(), "MatmulDeviceOperation", _att("tt-lang", 1, wedged=False))
    assert rung == "cpp"


def test_wedged_cpp_holds_indefinitely_after_clean_ttlang(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_ttl_available", lambda: True)
    for n in (1, 5, 20):
        atts = _att("tt-lang", 1, wedged=False) + _att("cpp", n, wedged=True)
        done, rung, reason = ladder(_op(), "MatmulDeviceOperation", atts)
        assert (not done) and rung == "cpp"
    assert "trace-fix attempt" in reason


def test_trace_compat_feedback_enriches_custom_rung(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_load_target", lambda: {"rung": "tt-lang"})
    out = perf_mcp._trace_compat_feedback("boom")
    assert "boom" in out and "ISOLATION" in out and "override_runtime_args" in out


def test_trace_compat_feedback_passthrough_for_knob(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_load_target", lambda: {"rung": "grid"})
    assert perf_mcp._trace_compat_feedback("boom") == "boom"
