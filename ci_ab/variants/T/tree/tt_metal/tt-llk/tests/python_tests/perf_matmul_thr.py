# Local sweep extension (not committed): every throttle level on matmul_test.cpp.
import pytest
from perf_matmul_ext import run_mm
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync, MathFidelity

F = DataFormat
CASES = [
    (fmt, sync, (rt, ct, kt), mf, thr)
    for fmt in (InputOutputFormat(F.Float16_b, F.Float16_b), InputOutputFormat(F.Bfp8_b, F.Bfp8_b))
    for sync in (DestSync.Half, DestSync.Full)
    for rt, ct in ((1, 1), (1, 8), (4, 2), (8, 1))
    for kt in (1, 32)
    for mf in (MathFidelity.LoFi, MathFidelity.HiFi2, MathFidelity.HiFi4)
    for thr in range(6)
]


@pytest.mark.perf
@pytest.mark.parametrize("fmt,sync,dims,mf,thr", CASES)
def test_perf_matmul_thr(perf_report, fmt, sync, dims, mf, thr):
    rt, ct, kt = dims
    run_mm(perf_report, fmt, DestAccumulation.No, sync, [rt * 32, kt * 32], [kt * 32, ct * 32], mf, throttle=thr)
