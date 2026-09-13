import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "perf_mcp_knob",
    str(Path(__file__).resolve().parents[1] / "cc_optimize" / "perf_mcp.py"),
)
perf_mcp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(perf_mcp)
ladder = perf_mcp._op_ladder_status
N = perf_mcp._MAX_KNOB_RETRIES


def _op(grid="partial", wdtype="bf16", bound="memory"):
    return {"grid": grid, "weight_dtype": wdtype, "bound_by": bound}


def _att(n, kind):
    return [{"op_signature": "MatmulDeviceOperation", "kernel_kind": kind} for _ in range(n)]


def test_grid_stays_open_after_one_weak_attempt():
    _, rung, _ = ladder(_op(grid="partial"), "MatmulDeviceOperation", _att(1, "grid"))
    assert rung == "knob:grid"


def test_grid_done_when_actually_applied():
    _, rung, _ = ladder(_op(grid="full"), "MatmulDeviceOperation", _att(1, "grid"))
    assert rung != "knob:grid"


def test_grid_retires_after_N_attempts_still_partial():
    _, rung, _ = ladder(_op(grid="partial"), "MatmulDeviceOperation", _att(N, "grid"))
    assert rung != "knob:grid"


def test_dtype_stays_open_after_one_attempt():
    _, rung, _ = ladder(_op(grid="full", wdtype="bf16"), "MatmulDeviceOperation", _att(1, "dtype"))
    assert rung == "knob:dtype"


def test_dtype_done_when_actually_lowered():
    _, rung, _ = ladder(_op(grid="full", wdtype="bf8_b"), "MatmulDeviceOperation", _att(1, "dtype"))
    assert rung not in ("knob:grid", "knob:dtype")


def test_dtype_retires_after_N_attempts():
    _, rung, _ = ladder(_op(grid="full", wdtype="bf16"), "MatmulDeviceOperation", _att(N, "dtype"))
    assert rung != "knob:dtype"
