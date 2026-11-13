# Dynamic throttle emulation, mode 3 (see perf_matmul_dyn_base.py).
import pytest
from conftest import skip_for_wormhole
from perf_matmul_dyn_base import CASES, run_dyn

pytestmark = (
    pytest.mark.llk_sweep
)  # exhaustive sweep: CI keeps a thin subset (LLK_SWEEP_ALL=1 for all)


@skip_for_wormhole
@pytest.mark.perf
@pytest.mark.parametrize("fmt,sync,dims,mf", CASES)
def test_perf_matmul_dyn3(perf_report, fmt, sync, dims, mf):
    run_dyn(perf_report, fmt, sync, dims, mf, 3)
