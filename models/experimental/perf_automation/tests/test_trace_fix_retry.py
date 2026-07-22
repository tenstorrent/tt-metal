import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "perf_mcp_tracefix",
    str(Path(__file__).resolve().parents[1] / "cc_optimize" / "perf_mcp.py"),
)
perf_mcp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(perf_mcp)
_ladder_raw = perf_mcp._op_ladder_status


def _op(grid="full", wdtype="bf8_b", bound="memory"):
    return {"grid": grid, "weight_dtype": wdtype, "bound_by": bound}


def _att(kind, n=1, wedged=False):
    return [{"op_signature": "MatmulDeviceOperation", "kernel_kind": kind, "wedged": wedged} for _ in range(n)]


_KDONE = _att("shard", perf_mcp._MAX_KNOB_RETRIES)


def ladder(op, op_code, attempts):
    return _ladder_raw(op, op_code, _KDONE + list(attempts))


def test_rung_state_clean_true_when_measured():
    clean, wedged = perf_mcp._rung_state(_att("tt-lang", 1, wedged=False), "tt-lang")
    assert clean and wedged == 0


def test_rung_state_wedged_counts_and_not_clean():
    clean, wedged = perf_mcp._rung_state(_att("tt-lang", 2, wedged=True), "tt-lang")
    assert (not clean) and wedged == 2


def test_wedged_ttlang_feedback_gives_proven_recipe(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_ttl_available", lambda: True)
    monkeypatch.setenv("TT_PERF_TRACE", "1")
    done, rung, reason = ladder(_op(), "MatmulDeviceOperation", _att("tt-lang", 1, wedged=True))
    assert (not done) and rung == "tt-lang"
    assert "attempt 2" in reason
    assert "PROVEN trace-safe recipe" in reason and "do NOT switch to cpp" in reason
    assert "PERSISTENT input buffer" in reason and "CACHE it" in reason


def test_author_reason_instructs_isolation_smoke_test(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_ttl_available", lambda: True)
    monkeypatch.setenv("TT_PERF_TRACE", "1")
    _, rung, reason = ladder(_op(), "MatmulDeviceOperation", [])
    assert rung == "tt-lang"
    assert "ISOLATION" in reason and "eager+trace" in reason and "PERSISTENT input buffer" in reason


def test_trace_off_author_reason_is_eager_no_recipe(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_ttl_available", lambda: True)
    monkeypatch.setenv("TT_PERF_TRACE", "0")
    _, rung, reason = ladder(_op(), "MatmulDeviceOperation", [])
    assert rung == "tt-lang"
    assert "TT_PERF_TRACE=0 (eager)" in reason
    assert "trace-safe recipe" not in reason and "ISOLATION" not in reason


def test_trace_off_wedged_reason_is_eager_crash(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_ttl_available", lambda: True)
    monkeypatch.setenv("TT_PERF_TRACE", "0")
    _, rung, reason = ladder(_op(), "MatmulDeviceOperation", _att("tt-lang", 1, wedged=True))
    assert rung == "tt-lang"
    assert "crashed in EAGER" in reason and "trace-safe recipe" not in reason


def test_trace_off_compat_feedback_passthrough(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_load_target", lambda: {"rung": "tt-lang"})
    monkeypatch.setenv("TT_PERF_TRACE", "0")
    assert perf_mcp._trace_compat_feedback("boom") == "boom"


def test_wedged_ttlang_holds_until_cap_then_cpp(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_ttl_available", lambda: True)
    cap = perf_mcp._MAX_KERNEL_WEDGES
    for n in range(1, cap):
        _, rung, _ = ladder(_op(), "MatmulDeviceOperation", _att("tt-lang", n, wedged=True))
        assert rung == "tt-lang", "%d < cap wedges should hold tt-lang, got %s" % (n, rung)
    for n in (cap, cap + 7):
        _, rung, _ = ladder(_op(), "MatmulDeviceOperation", _att("tt-lang", n, wedged=True))
        assert rung == "cpp", "%d >= cap wedges should advance to cpp, got %s" % (n, rung)


def test_clean_ttlang_advances_to_cpp(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_ttl_available", lambda: True)
    _, rung, _ = ladder(_op(), "MatmulDeviceOperation", _att("tt-lang", 1, wedged=False))
    assert rung == "cpp"


def test_wedged_cpp_holds_until_cap_then_structural(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_ttl_available", lambda: True)
    monkeypatch.setenv("TT_PERF_TRACE", "1")
    cap = perf_mcp._MAX_KERNEL_WEDGES
    for n in range(1, cap):
        atts = _att("tt-lang", 1, wedged=False) + _att("cpp", n, wedged=True)
        done, rung, reason = ladder(_op(), "MatmulDeviceOperation", atts)
        assert (not done) and rung == "cpp"
        assert "attempt" in reason and "PROVEN trace-safe recipe" in reason
    for n in (cap, cap + 3):
        atts = _att("tt-lang", 1, wedged=False) + _att("cpp", n, wedged=True)
        _, rung, _ = ladder(_op(), "MatmulDeviceOperation", atts)
        assert rung == "structural"


def test_trace_compat_feedback_enriches_custom_rung(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_load_target", lambda: {"rung": "tt-lang"})
    monkeypatch.setenv("TT_PERF_TRACE", "1")
    out = perf_mcp._trace_compat_feedback("boom")
    assert "boom" in out and "CACHE it" in out and "PERSISTENT input buffer" in out


def test_trace_compat_feedback_passthrough_for_knob(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_load_target", lambda: {"rung": "grid"})
    assert perf_mcp._trace_compat_feedback("boom") == "boom"
