# Dynamic throttle emulation, mode 4 (see perf_matmul_dyn_base.py).
import pytest
from perf_matmul_dyn_base import CASES, run_dyn

pytestmark = (
    pytest.mark.llk_sweep
)  # exhaustive sweep: CI keeps a thin subset (LLK_SWEEP_ALL=1 for all)
LLK_SWEEP_KEEP = frozenset(  # ids kept besides the thin subset: one per value of each swept dimension the subset misses
    [
        "test_perf_matmul_dyn4[fmt2-DestSync.Half-dims2-MathFidelity.HiFi4]",
    ]
)


@pytest.mark.perf
@pytest.mark.parametrize("fmt,sync,dims,mf", CASES)
def test_perf_matmul_dyn4(perf_report, fmt, sync, dims, mf):
    run_dyn(perf_report, fmt, sync, dims, mf, 4)
