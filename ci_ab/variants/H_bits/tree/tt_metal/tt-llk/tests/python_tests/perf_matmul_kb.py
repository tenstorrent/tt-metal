# Local sweep extension (not committed): kernel broadcast of the streamed operand (A when rt > ct, B when ct >= rt).
import pytest
from dataclasses import dataclass
from perf_matmul_ext import run_mm
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync, MathFidelity
from helpers.test_variant_parameters import TemplateParameter

F = DataFormat


# field-free so the perf report schema gets no extra columns
def MM_KB(a, b):
    @dataclass(repr=False)
    class _KB(TemplateParameter):
        def __repr__(self) -> str:  # the variant hash reads str() of the templates
            return f"MM_KB({a},{b})"

        def convert_to_cpp(self) -> str:
            return f"#define MM_KB_A {a}\n#define MM_KB_B {b}"

    return _KB()


CASES = [
    (fmt, sync, (rt, ct, kt), mf, kb)
    for fmt in (InputOutputFormat(F.Float16_b, F.Float16_b), InputOutputFormat(F.Bfp8_b, F.Bfp8_b))
    for sync in (DestSync.Half, DestSync.Full)
    for rt, ct in ((1, 8), (2, 4), (4, 2), (8, 1), (1, 1))
    for kt in (1, 4, 32)
    for mf in (MathFidelity.LoFi, MathFidelity.HiFi4)
    for kb in [(0, 1) if ct >= rt else (1, 0)]
]


@pytest.mark.perf
@pytest.mark.parametrize("fmt,sync,dims,mf,kb", CASES)
def test_perf_matmul_kb(perf_report, fmt, sync, dims, mf, kb):
    rt, ct, kt = dims
    run_mm(perf_report, fmt, DestAccumulation.No, sync, [rt * 32, kt * 32], [kt * 32, ct * 32], mf, extra=[MM_KB(*kb)])
