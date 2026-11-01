# Dynamic throttle emulation, mode 5 (see perf_matmul_dyn_base.py).
import pytest
from perf_matmul_dyn_base import CASES, run_dyn


@pytest.mark.perf
@pytest.mark.parametrize("fmt,sync,dims,mf", CASES)
def test_perf_matmul_dyn5(perf_report, fmt, sync, dims, mf):
    run_dyn(perf_report, fmt, sync, dims, mf, 5)
