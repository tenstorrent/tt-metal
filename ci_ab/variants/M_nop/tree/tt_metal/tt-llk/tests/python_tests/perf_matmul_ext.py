# Local sweep extension (not committed): Bfp4_b, mixed SrcA/SrcB formats (production weights in Bfp8_b/Bfp4_b with
# bf16 activations), on perf_matmul's grid. Harness convention: input_format configures unpacker 0 (SrcA, matmul in1),
# input_format_B unpacker 1 (SrcB, matmul in0).
import perf_matmul as pm
import pytest
import helpers.test_variant_parameters as tvp
from helpers.format_config import DataFormat, InputOutputFormat, is_dest_acc_needed
from helpers.llk_params import DestAccumulation, MathFidelity, PerfRunType, Transpose
from helpers.matmul_sweep import PERF_RING_TILES, generate_tile_dims
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM, DEST_SYNC, LOOP_FACTOR, MATH_FIDELITY, NUM_FACES, THROTTLE_LEVEL, TILE_COUNT, UNPACK_TRANS_FACES,
)

F = DataFormat
EXT_FORMATS = [
    InputOutputFormat(F.Bfp4_b, F.Float16_b),
    InputOutputFormat(F.Bfp4_b, F.Bfp8_b),
    InputOutputFormat(F.Bfp4_b, F.Bfp4_b),
    InputOutputFormat(F.Bfp4_b, F.Float32),
    InputOutputFormat(F.Bfp8_b, F.Float16_b, input_format_B=F.Float16_b),
    InputOutputFormat(F.Bfp8_b, F.Bfp8_b, input_format_B=F.Float16_b),
    InputOutputFormat(F.Bfp4_b, F.Float16_b, input_format_B=F.Float16_b),
    InputOutputFormat(F.Bfp4_b, F.Bfp8_b, input_format_B=F.Float16_b),
]
NARROW = getattr(tvp, "MATMUL_STREAM_NARROW", None)


def run_mm(perf_report, formats, dest_acc, dest_sync, matrix_a, matrix_b, math_fidelity, throttle=0, extra=(), source=None, transpose=Transpose.No):
    dims = generate_tile_dims((matrix_a, matrix_b))
    n = dims.rt_dim * dims.ct_dim * dims.kt_dim
    st = min(n, PERF_RING_TILES)
    templates = [MATH_FIDELITY(math_fidelity), DEST_SYNC(dest_sync), THROTTLE_LEVEL(throttle)]
    if NARROW is not None:
        templates += NARROW.for_block(formats, dest_acc, dest_sync, dims.ct_dim, dims.rt_dim)
    templates += list(extra)
    try:
        cfg = PerfConfig(
        source or ("sources/matmul_test.cpp" if not extra else "sources/matmul_kb_test.cpp"),
        formats,
        [PerfRunType.L1_TO_L1, PerfRunType.UNPACK_ISOLATE, PerfRunType.MATH_ISOLATE, PerfRunType.PACK_ISOLATE,
         PerfRunType.L1_CONGESTION],
        templates=templates,
        runtimes=[UNPACK_TRANS_FACES(transpose), NUM_FACES(), LOOP_FACTOR(64), TILE_COUNT(n),
                  CRK_TILE_DIMM(dims.ct_dim, dims.rt_dim, dims.kt_dim)],
        variant_stimuli=StimuliConfig(None, formats.input_format, None, formats.input_format_B, formats.output_format,
                                      tile_count_A=st, tile_count_B=st,
                                      tile_count_res=min(dims.rt_dim * dims.ct_dim, PERF_RING_TILES)),
        dest_acc=dest_acc,
        )
    except ValueError as e:  # format pairs the harness rejects (data_format_inference.py)
        pytest.skip(f"rejected by the harness: {e}")
    cfg.run(perf_report)


@pytest.mark.perf
@parametrize(
    combos=pm.matmul_combos(
        formats=EXT_FORMATS,
        dest_acc=lambda f: [DestAccumulation.Yes] if is_dest_acc_needed(f) else [DestAccumulation.No, DestAccumulation.Yes],
    ),
    math_fidelity=[MathFidelity.LoFi, MathFidelity.HiFi2, MathFidelity.HiFi3, MathFidelity.HiFi4],
)
def test_perf_matmul_ext(perf_report, combos, math_fidelity):
    formats, dest_acc, dest_sync, (a, b) = combos
    run_mm(perf_report, formats, dest_acc, dest_sync, a, b, math_fidelity)
