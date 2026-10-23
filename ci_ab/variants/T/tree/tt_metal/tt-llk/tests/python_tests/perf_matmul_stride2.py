# Local sweep extension (not committed): in1 column stride 2 (see perf_matmul_stride_base.py).
import pytest
from perf_matmul_stride_base import CASES, run_stride


@pytest.mark.perf
@pytest.mark.parametrize("fmt,sync,dims,mf", [c for c in CASES if (c[2][1] - 1) * 2 < 16])
def test_perf_matmul_stride2(perf_report, fmt, sync, dims, mf):
    run_stride(perf_report, fmt, sync, dims, mf, 2)
