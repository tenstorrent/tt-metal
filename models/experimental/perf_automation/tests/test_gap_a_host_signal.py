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


def test_routes_host_when_device_ops_rung_exhausted():
    r = host_gate(_prof(), blocking=[], attempts=[])
    assert r is not None and r["op"] == "host_overhead" and r["next_rung"] == "structural"


def test_never_routes_while_device_still_blocking():
    r = host_gate(_prof(), blocking=[{"op": "MatmulDeviceOperation", "next_rung": "knob:dtype"}], attempts=[])
    assert r is None


def test_never_routes_unavailable_host():
    r = host_gate(_prof(source="unavailable"), blocking=[], attempts=[])
    assert r is None


def test_never_routes_subthreshold_host():
    r = host_gate(_prof(host_ms=0.1), blocking=[], attempts=[])
    assert r is None


def test_cleared_once_host_attempt_recorded():
    attempts = [{"op_signature": "host_overhead", "kernel_kind": "structural"}]
    r = host_gate(_prof(), blocking=[], attempts=attempts)
    assert r is None
