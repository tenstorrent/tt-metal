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
from helpers.pack_sweep import SHIM_CONFIG
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
    PACK_API_SWEEP,
    PACK_SRC_FROM_OUTPUT,
    THROTTLE_LEVEL,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
)

F = DataFormat
OUTS = [F.Bfp8_b, F.Bfp4_b, F.Float16_b, F.Float32]
FIDS = [MathFidelity.LoFi, MathFidelity.HiFi2, MathFidelity.HiFi3, MathFidelity.HiFi4]
KTS = (1, 2, 4, 8)
# form: (arm, CB depth in blocks, context 2) through pack_api_shim.h: 0 per tile, 1 block, 2 W (a CB of the blocks the output ring
# holds), 3 main per tile, 4 and 5 W in a CB of two and one blocks, 6 the form tables; 11, 12 and 16 the block, W and tables behind a pack-thread
# SFPU init (context 2)
FORM_ARM = {
    0: ("T", 0, False),
    1: ("B", 0, False),
    2: ("W", 0, False),
    3: ("M", 0, False),
    4: ("W", 2, False),
    5: ("W", 1, False),
    6: ("0", 0, False),
    11: ("B", 0, True),
    12: ("W", 0, True),
    16: ("0", 0, True),
}


@dataclass
class MM_CONTIG(TemplateParameter):
    """sweep_contig: the block forms of the cell are the plain contiguous ones (not block float)"""

    mm_contig: bool
    mm_f32: bool = False

    def convert_to_cpp(self) -> str:
        return f"constexpr bool sweep_contig = {str(self.mm_contig).lower()};\nconstexpr bool sweep_float32_out = {str(self.mm_f32).lower()};"


def _shapes(cap):
    out = [(1, n) for n in range(2, cap + 1)] + [(2, n) for n in range(2, cap // 2 + 1)]
    if cap >= 8:
        out += [(4, 2), (4, 3), (4, 4)] if cap == 8 else [(4, n) for n in range(2, 5)]
    return [(r, c) for r, c in out if r * c <= cap]


def _cases():
    out = []
    for fmt in OUTS:
        for acc in (DestAccumulation.Yes,):
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
@parametrize(case=_cases(), math_fidelity=FIDS, form=list(FORM_ARM))
def test_perf_matmul_pack_sweep10(perf_report, case, math_fidelity, form):
    fmt, dest_acc, dest_sync, rt, ct, kt = case
    arm, cbd, ctx2 = FORM_ARM[form]
    formats = InputOutputFormat(F.Float32, fmt)
    dims = generate_tile_dims(
        ([rt * TILE_DIM, kt * TILE_DIM], [kt * TILE_DIM, ct * TILE_DIM])
    )
    variant_tile_count = dims.rt_dim * dims.ct_dim * dims.kt_dim
    stimuli_tiles = min(variant_tile_count, PERF_RING_TILES)
    PerfConfig(
        "sources/matmul_pack_sweep10_test.cpp",
        formats,
        run_types=ALL_PERF_RUN_TYPES,
        templates=[
            MATH_FIDELITY(math_fidelity),
            DEST_SYNC(dest_sync),
            THROTTLE_LEVEL(),
            PACK_SRC_FROM_OUTPUT(),
            MM_CONTIG(not fmt.is_block_float(), fmt == F.Float32),
            PACK_API_SWEEP(arm, rt * ct, matmul=True, sfpu=ctx2, cbd=cbd),
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
        **SHIM_CONFIG,
    ).run(perf_report)
