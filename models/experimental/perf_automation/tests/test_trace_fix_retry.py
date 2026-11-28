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


def test_trace_off_compat_feedback_passthrough(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_load_target", lambda: {"rung": "tt-lang"})
    monkeypatch.setenv("TT_PERF_TRACE", "0")
    assert perf_mcp._trace_compat_feedback("boom") == "boom"


def test_trace_compat_feedback_enriches_custom_rung(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_load_target", lambda: {"rung": "tt-lang"})
    monkeypatch.setenv("TT_PERF_TRACE", "1")
    out = perf_mcp._trace_compat_feedback("boom")
    assert "boom" in out and "CACHE it" in out and "PERSISTENT input buffer" in out


def test_trace_compat_feedback_passthrough_for_knob(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_load_target", lambda: {"rung": "grid"})
    assert perf_mcp._trace_compat_feedback("boom") == "boom"


# REMOVED 2026-07-25: the tests below asserted the PRE-REORDER ladder (kernels straight after
# knobs). The ladder was deliberately changed to run structural/algorithmic levers BEFORE tt-lang/C++
# -- and gated so the harness demands a real attempt -- because the KV-cache rung was never being
# picked up. These assertions encoded the old order, so they contradicted the intended design rather
# than protecting it. Removed: test_author_reason_instructs_isolation_smoke_test, test_clean_ttlang_advances_to_cpp, test_trace_off_author_reason_is_eager_no_recipe, test_trace_off_wedged_reason_is_eager_crash, test_wedged_cpp_holds_until_cap_then_structural, test_wedged_ttlang_feedback_gives_proven_recipe, test_wedged_ttlang_holds_until_cap_then_cpp
