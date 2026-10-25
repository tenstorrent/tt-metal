# Local sweep helper (not committed): matmul_block_math_dynamic_throttle emulated in sources/matmul_dyn_test.cpp
# (per call a volatile read of the ARC FW scratch word, MOP re-init on a change, level 5 when odd).
import pytest
from perf_matmul_ext import run_mm
from helpers.format_config import DataFormat, InputOutputFormat
from dataclasses import dataclass
from helpers.llk_params import DestAccumulation, DestSync, MathFidelity
from helpers.test_variant_parameters import TemplateParameter


def DYN(mode):
    # 0: the FW word as read (production), 1: forced odd (throttled branch), 2: toggled every call (re-init per call), 3: forced even (unthrottled branch)
    @dataclass(repr=False)
    class _D(TemplateParameter):
        def __repr__(self) -> str:
            return f"DYN({mode})"

        def convert_to_cpp(self) -> str:
            return f"#define DYN_MODE {mode}"

    return _D()

F = DataFormat
CASES = [
    (fmt, sync, (rt, ct, kt), mf)
    for fmt in (InputOutputFormat(F.Float16_b, F.Float16_b), InputOutputFormat(F.Bfp8_b, F.Bfp8_b))
    for sync in (DestSync.Half, DestSync.Full)
    for rt, ct in ((1, 1), (1, 8), (4, 2), (8, 1))
    for kt in (1, 4, 32)
    for mf in (MathFidelity.LoFi, MathFidelity.HiFi2, MathFidelity.HiFi4)
]




def run_dyn(perf_report, fmt, sync, dims, mf, mode):
    rt, ct, kt = dims
    run_mm(perf_report, fmt, DestAccumulation.No, sync, [rt * 32, kt * 32], [kt * 32, ct * 32], mf,
           extra=[DYN(mode)], source="sources/matmul_dyn_test.cpp")
