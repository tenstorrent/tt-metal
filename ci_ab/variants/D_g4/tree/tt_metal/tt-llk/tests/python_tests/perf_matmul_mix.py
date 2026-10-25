# Local sweep extension (not committed): SrcA/SrcB format mixes beyond the production pairs, every output format.
# input_format = unpacker 0 (SrcA, matmul in1), input_format_B = unpacker 1 (SrcB, matmul in0).
import perf_matmul as pm
import pytest
from perf_matmul_ext import run_mm
from helpers.format_config import DataFormat, InputOutputFormat, is_dest_acc_needed
from helpers.llk_params import DestAccumulation, MathFidelity
from helpers.param_config import parametrize

F = DataFormat
PAIRS = [(F.Float16_b, F.Bfp8_b), (F.Float16_b, F.Bfp4_b), (F.Float32, F.Float16_b), (F.Float16_b, F.Float32),
         (F.Bfp8_b, F.Bfp4_b), (F.Bfp4_b, F.Bfp8_b)]
# Float16 with Bfp8_b is rejected by the harness (data_format_inference.py:410-414, exponent widths differ) and
# Float16 with Bfp8 has no tile size (tile_constants.py:215); outputs Float16_b, Bfp8_b, Float32 (the output only moves
# the pack term; perf_matmul covers all 16 output pairs); LoFi, HiFi2, HiFi4 (HiFi3 sampled out: interpolates the FPU term)
MIX = [InputOutputFormat(a, o, input_format_B=b) for a, b in PAIRS for o in (F.Float16_b, F.Bfp8_b, F.Float32)]


@pytest.mark.perf
@parametrize(
    combos=pm.matmul_combos(
        formats=MIX,
        dest_acc=lambda f: [DestAccumulation.Yes] if is_dest_acc_needed(f) else [DestAccumulation.No, DestAccumulation.Yes],
    ),
    math_fidelity=[MathFidelity.LoFi, MathFidelity.HiFi2, MathFidelity.HiFi4],
)
def test_perf_matmul_mix(perf_report, combos, math_fidelity):
    formats, dest_acc, dest_sync, (a, b) = combos
    run_mm(perf_report, formats, dest_acc, dest_sync, a, b, math_fidelity)
