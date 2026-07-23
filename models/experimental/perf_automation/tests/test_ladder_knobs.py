import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "perf_mcp_ladderknobs",
    str(Path(__file__).resolve().parents[1] / "cc_optimize" / "perf_mcp.py"),
)
perf_mcp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(perf_mcp)
ladder = perf_mcp._op_ladder_status
MAX = perf_mcp._MAX_KNOB_RETRIES

MM = "MatmulDeviceOperation"


def _att(kind, n=1, wedged=False):
    return [{"op_signature": MM, "kernel_kind": kind, "wedged": wedged} for _ in range(n)]


def test_grid_knob_fires_when_grid_not_full():
    _, rung, _ = ladder({"grid": "partial", "bound_by": "memory"}, MM, [])
    assert rung == "knob:grid"


def test_grid_knob_bounded_by_retry_counter():
    _, rung, _ = ladder({"grid": "partial", "bound_by": "memory"}, MM, _att("grid", MAX))
    assert rung != "knob:grid"


def test_fidelity_knob_fires_on_compute_bound():
    _, rung, _ = ladder({"grid": "full", "bound_by": "compute"}, MM, [])
    assert rung == "knob:fidelity"


def test_fidelity_bounded_by_retry_counter():
    _, rung, _ = ladder({"grid": "full", "bound_by": "compute"}, MM, _att("fidelity", MAX))
    assert rung != "knob:fidelity"


def test_dtype_knob_fires_when_weight_dtype_unknown_failopen():
    _, rung, _ = ladder({"grid": "full", "bound_by": "memory", "weight_dtype": ""}, MM, [])
    assert rung == "knob:dtype"


def test_dtype_knob_skips_when_weight_already_low():
    _, rung, _ = ladder({"grid": "full", "bound_by": "memory", "weight_dtype": "bf8_b"}, MM, [])
    assert rung == "knob:shard"


def test_shard_knob_fires_on_memory_bound_after_dtype():
    _, rung, _ = ladder({"grid": "full", "bound_by": "memory", "weight_dtype": "bf8_b"}, MM, _att("shard", 0))
    assert rung == "knob:shard"


def test_shard_bounded_by_retry_counter(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_ttl_available", lambda: True)
    atts = _att("shard", MAX)
    _, rung, _ = ladder({"grid": "full", "bound_by": "memory", "weight_dtype": "bf8_b"}, MM, atts)
    assert rung != "knob:shard"


def test_shard_fires_for_memory_bound_nonmatmul():
    _, rung, _ = ladder({"grid": "full", "bound_by": "memory"}, "LayerNormDeviceOperation", [])
    assert rung == "knob:shard"


def test_ladder_order_grid_before_fidelity_before_kernels(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_ttl_available", lambda: True)
    _, rung, _ = ladder({"grid": "partial", "bound_by": "compute"}, MM, [])
    assert rung == "knob:grid"
    _, rung, _ = ladder({"grid": "full", "bound_by": "compute"}, MM, _att("fidelity", MAX))
    assert rung in ("tt-lang", "cpp", "tp-fracture", "structural")


def test_residual_report_propagates_grid_fidelity_weight_dtype():
    from agent import roofline

    prof = {
        "device_ms": 10.0,
        "buckets": [
            {
                "id": "matmul",
                "device_ms": 10.0,
                "top_ops": [
                    {
                        "op_code": MM,
                        "shape": "32x2048@2048x6144",
                        "device_ms": 10.0,
                        "count": 1,
                        "grid": "partial",
                        "fidelity": "hifi4",
                        "weight_dtype": "bf16",
                    }
                ],
            }
        ],
    }
    rep = roofline.residual_report(prof, {})
    row = rep["rows"][0]
    assert "grid" in row and "fidelity" in row and "weight_dtype" in row
    assert row["grid"] == "partial" and row["fidelity"] == "hifi4"


def test_host_gate_not_starved_by_blocking_device_ops():
    prof = {
        "buckets": [
            {"id": "host_overhead", "device_ms": 202.0, "tags": {"source": "op_gap"}},
        ]
    }
    block = perf_mcp._host_gate(prof, [{"op": "MatmulDeviceOperation"}], [])
    assert block is not None
