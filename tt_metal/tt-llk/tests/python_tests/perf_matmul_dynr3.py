import pytest
from dynr_base import CASES, run_dyn

pytestmark = (
    pytest.mark.llk_sweep
)  # exhaustive sweep: CI keeps a thin subset (LLK_SWEEP_ALL=1 for all)
LLK_SWEEP_KEEP = frozenset(  # ids kept besides the thin subset: one per value of each swept dimension the subset misses
    [
        "test_perf_matmul_dynr3[fmt0-DestSync.Half-dims0-MathFidelity.LoFi]",
        "test_perf_matmul_dynr3[fmt9-DestSync.Full-dims9-MathFidelity.HiFi2]",
    ]
)


@pytest.mark.perf
@pytest.mark.parametrize("fmt,sync,dims,mf", CASES)
def test_perf_matmul_dynr3(perf_report, fmt, sync, dims, mf):
    run_dyn(perf_report, fmt, sync, dims, mf, 3)
