import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "perf_mcp_gapa",
    str(Path(__file__).resolve().parents[1] / "cc_optimize" / "perf_mcp.py"),
)
perf_mcp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(perf_mcp)
host_gate = perf_mcp._host_gate


def _prof(host_ms=24.7, source="op_gap"):
    return {
        "device_ms": 856.0,
        "buckets": [
            {"id": "matmul", "device_ms": 800.0},
            {"id": "host_overhead", "device_ms": host_ms, "tags": {"source": source}},
        ],
    }


def test_never_routes_unavailable_host():
    r = host_gate(_prof(source="unavailable"), blocking=[], attempts=[])
    assert r is None


def test_never_routes_subthreshold_host():
    r = host_gate(_prof(host_ms=0.1), blocking=[], attempts=[])
    assert r is None


def test_not_cleared_by_an_attempt_that_measured_nothing():
    """SUPERSEDED CONTRACT. This asserted `r is None` -- one structural row, of any outcome, closed
    the dispatch axis. That is the defect test_the_host_rung_clears_on_a_measurement.py documents:
    on gemma-3-12b-it a single trace attempt measuring 400.44 ms with beat_baseline=False sealed the
    rung, and across 158 later attempts the axis had ONE entry while 20.92 ms of host_overhead sat
    in every profile. "Attempted" is not "resolved"."""
    attempts = [{"op_signature": "host_overhead", "kernel_kind": "structural"}]
    r = host_gate(_prof(), blocking=[], attempts=attempts)
    assert r is not None and r.get("next_rung") == "trace-capture"


def test_cleared_by_a_measured_win():
    """What DOES clear it: a lever that actually reduced cost."""
    attempts = [
        {
            "op_signature": "host_overhead",
            "kernel_kind": "structural",
            "beat_baseline": True,
            "measured_ms": 312.5,
        }
    ]
    assert host_gate(_prof(), blocking=[], attempts=attempts) is None


def test_cleared_after_the_attempt_cap():
    """...or after N real attempts, so an infeasible lever cannot loop forever."""
    attempts = [
        {"op_signature": "host_overhead", "kernel_kind": k, "measured_ms": 400.0, "beat_baseline": False}
        for k in ("trace", "structural", "trace-capture")
    ]
    assert host_gate(_prof(), blocking=[], attempts=attempts) is None


# REMOVED 2026-07-25: the tests below asserted the PRE-REORDER ladder (kernels straight after
# knobs). The ladder was deliberately changed to run structural/algorithmic levers BEFORE tt-lang/C++
# -- and gated so the harness demands a real attempt -- because the KV-cache rung was never being
# picked up. These assertions encoded the old order, so they contradicted the intended design rather
# than protecting it. Removed: test_never_routes_while_device_still_blocking, test_routes_host_when_device_ops_rung_exhausted
