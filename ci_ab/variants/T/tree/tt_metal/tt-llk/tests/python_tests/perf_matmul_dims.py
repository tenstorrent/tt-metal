# Local sweep extension (not committed): the production subblocks (SUBBLOCK_HW_CHOICES) and inner dims kt 1 to 128,
# plus the matmul_tiles path (1x1 block per call) with in1 transpose. Sampled: four formats, LoFi/HiFi2/HiFi4.
import pytest
from perf_matmul_ext import run_mm
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync, MathFidelity, Transpose
from helpers.matmul_sweep import unpack_matmul_fits_perf_ring

F = DataFormat
FMTS = [(InputOutputFormat(F.Float16_b, F.Float16_b), DestAccumulation.No),
        (InputOutputFormat(F.Bfp8_b, F.Bfp8_b), DestAccumulation.No),
        (InputOutputFormat(F.Bfp8_b, F.Float16_b, input_format_B=F.Float16_b), DestAccumulation.No),
        (InputOutputFormat(F.Float32, F.Float32), DestAccumulation.Yes)]
SUB = [(7, 1), (1, 7), (5, 1), (1, 5), (3, 1), (1, 3), (3, 2), (2, 3), (6, 1), (1, 6), (8, 1), (1, 8), (4, 2), (2, 4),
       (1, 1), (2, 2), (4, 1), (1, 4), (2, 1), (1, 2)]
KT = (1, 2, 3, 4, 5, 7, 8, 16, 32, 64, 128)
CAP = {(DestSync.Half, False): 8, (DestSync.Full, False): 16, (DestSync.Half, True): 4, (DestSync.Full, True): 8}
CASES = [
    (fmt, acc, sync, (rt, ct, kt), mf, tr)
    for fmt, acc in FMTS
    for sync in (DestSync.Half, DestSync.Full)
    for rt, ct in SUB
    if rt * ct <= CAP[(sync, acc == DestAccumulation.Yes)]
    for kt in KT
    if unpack_matmul_fits_perf_ring(rt, ct, kt)
    for mf in (MathFidelity.LoFi, MathFidelity.HiFi2, MathFidelity.HiFi4)
    for tr in ((Transpose.No, Transpose.Yes) if (rt, ct) == (1, 1) else (Transpose.No,))
]


@pytest.mark.perf
@pytest.mark.parametrize("fmt,acc,sync,dims,mf,tr", CASES)
def test_perf_matmul_dims(perf_report, fmt, acc, sync, dims, mf, tr):
    rt, ct, kt = dims
    run_mm(perf_report, fmt, acc, sync, [rt * 32, kt * 32], [kt * 32, ct * 32], mf, transpose=tr)
