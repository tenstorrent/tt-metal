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
from dataclasses import dataclass
from helpers.test_variant_parameters import (
    TemplateParameter,
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
OUTS = [F.Bfp8_b, F.Bfp4_b, F.Bfp2_b, F.Bfp8, F.Bfp4, F.Bfp2, F.Float16_b, F.Float32]
FIDS = [MathFidelity.LoFi, MathFidelity.HiFi2, MathFidelity.HiFi3, MathFidelity.HiFi4]
KTS = (1, 2, 4, 8)


@dataclass
class MM_CONTIG(TemplateParameter):
    """sweep_contig: the block forms of the cell are the plain contiguous ones (not block float)"""

    contig: bool
    f32: bool = False

    def convert_to_cpp(self) -> str:
        return f"constexpr bool sweep_contig = {str(self.contig).lower()};\nconstexpr bool sweep_float32_out = {str(self.f32).lower()};"


def _shapes(cap):
    out = [(1, n) for n in range(2, cap + 1)] + [(2, n) for n in range(2, cap // 2 + 1)]
    if cap >= 8:
        out += [(4, 2), (4, 3), (4, 4)] if cap == 8 else [(4, n) for n in range(2, 5)]
    return [(r, c) for r, c in out if r * c <= cap]


def _cases():
    out = []
    for fmt in OUTS:
        for acc in (DestAccumulation.No, DestAccumulation.Yes):
            for sync in (DestSync.Half, DestSync.Full):
                cap = (
                    (4 if acc == DestAccumulation.Yes else 8)
                    if sync == DestSync.Half
                    else (8 if acc == DestAccumulation.Yes else 16)
                )
                for rt, ct in _shapes(cap):
                    for kt in KTS:
                        if unpack_matmul_fits_perf_ring(rt, ct, kt):
                            out.append((fmt, acc, sync, rt, ct, kt))
    return out


@pytest.mark.perf
@parametrize(
    case=_cases(), math_fidelity=FIDS, form=[0, 1, 2, 3, 4]
)  # pk-limit5: 3 = main per tile, 4 = W under the continuation rule (CB of two blocks)
def test_perf_matmul_pack_sweep9(perf_report, case, math_fidelity, form):
    fmt, dest_acc, dest_sync, rt, ct, kt = case
    formats = InputOutputFormat(F.Float16_b, fmt)
    dims = generate_tile_dims(
        ([rt * TILE_DIM, kt * TILE_DIM], [kt * TILE_DIM, ct * TILE_DIM])
    )
    variant_tile_count = dims.rt_dim * dims.ct_dim * dims.kt_dim
    stimuli_tiles = min(variant_tile_count, PERF_RING_TILES)
    PerfConfig(
        "sources/matmul_pack_sweep9_test.cpp",
        formats,
        run_types=ALL_PERF_RUN_TYPES,
        templates=[
            MATH_FIDELITY(math_fidelity),
            DEST_SYNC(dest_sync),
            THROTTLE_LEVEL(),
            PACK_SRC_FROM_OUTPUT(),
            MM_CONTIG(not fmt.is_block_float(), fmt == F.Float32),
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
