"""HiFi crossover (ad hoc): the block-float pack forms of llk_pack_block behind matmul math at LoFi, HiFi2, HiFi3 and HiFi4 (matmul_pack_sweep_test.cpp)."""

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TILE_DIM
from helpers.llk_params import DestAccumulation, DestSync, MathFidelity, Transpose
from helpers.matmul_sweep import (
    PERF_RING_TILES,
    generate_tile_dims,
    unpack_matmul_fits_perf_ring,
)
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES, PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    DEST_INDEX,
    DEST_SYNC,
    LOOP_FACTOR,
    MATH_FIDELITY,
    NUM_FACES,
    PACK_SRC_FROM_OUTPUT,
    THROTTLE_LEVEL,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
)

F = DataFormat
OUTS = [F.Bfp8_b, F.Bfp4_b, F.Bfp2_b, F.Bfp8]
FIDS = [MathFidelity.LoFi, MathFidelity.HiFi2, MathFidelity.HiFi3, MathFidelity.HiFi4]


def _cases():
    out = []
    for fmt in OUTS:
        for acc in (DestAccumulation.No, DestAccumulation.Yes):
            for sync in (DestSync.Half, DestSync.Full):
                cap = (
                    4 if (acc == DestAccumulation.Yes and sync == DestSync.Half) else 8
                )
                for rt, ct in [(1, n) for n in range(2, cap + 1)] + [
                    (2, n) for n in (2, 3, 4) if 2 * n <= cap
                ]:
                    for kt in (1, 4):
                        if unpack_matmul_fits_perf_ring(rt, ct, kt):
                            out.append((fmt, acc, sync, rt, ct, kt))
    return out


@pytest.mark.perf
@parametrize(case=_cases(), math_fidelity=FIDS, form=[0, 1, 2])
def test_perf_matmul_pack_sweep(perf_report, case, math_fidelity, form):
    fmt, dest_acc, dest_sync, rt, ct, kt = case
    formats = InputOutputFormat(F.Float16_b, fmt)
    dims = generate_tile_dims(
        ([rt * TILE_DIM, kt * TILE_DIM], [kt * TILE_DIM, ct * TILE_DIM])
    )
    variant_tile_count = dims.rt_dim * dims.ct_dim * dims.kt_dim
    stimuli_tiles = min(variant_tile_count, PERF_RING_TILES)
    PerfConfig(
        "sources/matmul_pack_sweep_test.cpp",
        formats,
        run_types=ALL_PERF_RUN_TYPES,
        templates=[
            MATH_FIDELITY(math_fidelity),
            DEST_SYNC(dest_sync),
            THROTTLE_LEVEL(),
            PACK_SRC_FROM_OUTPUT(),
        ],
        runtimes=[
            UNPACK_TRANS_FACES(Transpose.No),
            NUM_FACES(),
            LOOP_FACTOR(64),
            TILE_COUNT(variant_tile_count),
            CRK_TILE_DIMM(dims.ct_dim, dims.rt_dim, dims.kt_dim),
            DEST_INDEX(form),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=stimuli_tiles,
            tile_count_B=stimuli_tiles,
            tile_count_res=min(dims.rt_dim * dims.ct_dim, PERF_RING_TILES),
        ),
        dest_acc=dest_acc,
    ).run(perf_report)
