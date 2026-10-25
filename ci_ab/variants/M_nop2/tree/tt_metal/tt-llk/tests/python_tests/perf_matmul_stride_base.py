# Local sweep extension (not committed): in1 column stride (_llk_unpack_AB_matmul_set_in1_column_stride_) on the main
# matmul path, sources/matmul_stride_test.cpp; the stride keeps the in1 walk inside the 16-tile perf ring.
import pytest
from dataclasses import dataclass
from perf_matmul_ext import run_mm
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync, MathFidelity
from helpers.test_variant_parameters import TemplateParameter

F = DataFormat


def MM_STRIDE(n):
    @dataclass(repr=False)
    class _S(TemplateParameter):
        def __repr__(self) -> str:
            return f"MM_STRIDE({n})"

        def convert_to_cpp(self) -> str:
            return f"#define MM_IN1_STRIDE {n}"

    return _S()


CASES = [
    (fmt, sync, (rt, ct, kt), mf)
    for fmt in (InputOutputFormat(F.Float16_b, F.Float16_b), InputOutputFormat(F.Bfp8_b, F.Bfp8_b))
    for sync in (DestSync.Half, DestSync.Full)
    for rt, ct in ((1, 2), (1, 4), (1, 8), (2, 4))
    for kt in (1, 4, 32)
    if rt == 1 or kt < 16
    for mf in (MathFidelity.LoFi, MathFidelity.HiFi4)
]




def run_stride(perf_report, fmt, sync, dims, mf, s):
    rt, ct, kt = dims
    run_mm(perf_report, fmt, DestAccumulation.No, sync, [rt * 32, kt * 32], [kt * 32, ct * 32], mf,
           extra=[MM_STRIDE(s)], source="sources/matmul_stride_test.cpp")
