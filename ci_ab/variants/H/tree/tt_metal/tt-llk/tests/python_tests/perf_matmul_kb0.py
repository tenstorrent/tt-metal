# Local sweep extension (not committed): the kernel-broadcast cases of perf_matmul_kb with broadcast off, same source.
import pytest
from perf_matmul_ext import run_mm
from perf_matmul_kb import CASES, MM_KB
from helpers.llk_params import DestAccumulation


@pytest.mark.perf
@pytest.mark.parametrize("fmt,sync,dims,mf", [c[:4] for c in CASES])
def test_perf_matmul_kb0(perf_report, fmt, sync, dims, mf):
    rt, ct, kt = dims
    run_mm(perf_report, fmt, DestAccumulation.No, sync, [rt * 32, kt * 32], [kt * 32, ct * 32], mf, extra=[MM_KB(0, 0)])
