# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import pytest
from helpers.format_config import DataFormat, is_dest_acc_needed
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    StochasticRounding,
)
from helpers.matmul_sweep import (
    MatmulConfig,
)
from helpers.pack_sweep import SHIM_CONFIG
from helpers.param_config import input_output_formats
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    DEST_INDEX,
    DEST_SYNC,
    IN_TILE_DIMS,
    LOOP_FACTOR,
    MATH_FIDELITY,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    PACK_API_SWEEP,
    PARTIAL_FACE,
    THROTTLE_LEVEL,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)

MATMUL_FORMATS = input_output_formats(
    [
        DataFormat.Bfp8_b,
        DataFormat.Float16_b,
        DataFormat.Float32,
    ]
)
DEST_SYNC_MODES = [DestSync.Half, DestSync.Full]
STOCHASTIC_ROUNDING_MODES = [StochasticRounding.No]

# Dest-fill RT x CT from dest capacity, dest occupancy blocks, and half-dest
# mid-fill when they fit. KT 1 and 4 (long-K lives on perf_matmul).
PERF_KT_DIMS = (1, 4)

from dataclasses import dataclass

from conftest import skip_for_wormhole
from helpers.format_config import InputOutputFormat
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    MathFidelity,
    PerfRunType,
    StochasticRounding,
    Transpose,
)
from helpers.matmul_sweep import (
    FaceLayoutConfig,
    MatmulConfig,
    calculate_matmul_output_faces,
)
from helpers.matmul_sweep import generate_tile_dims as _gtd
from helpers.param_config import parametrize
from helpers.test_variant_parameters import TemplateParameter


@dataclass
class TINY_F32(TemplateParameter):
    tiny_f32: bool

    def convert_to_cpp(self) -> str:
        return f"constexpr bool sweep_float32_out = {str(self.tiny_f32).lower()};"


FMTS = [
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Bfp8_b),
    InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
]
FIDS = [MathFidelity.LoFi, MathFidelity.HiFi2, MathFidelity.HiFi3, MathFidelity.HiFi4]
R0 = (16, 8, 4, 1)
KTS = (1, 2, 4, 8)
# form: (arm, CB depth in blocks, context 2) through pack_api_shim.h: 0 per tile, 1 main per tile, 2 block, 3 W (a CB of four blocks),
# 7 and 8 W in a CB of two and one blocks, 9 the form tables (a CB of four blocks); 12, 13 and 19 the block, W and tables behind a pack-thread SFPU
# init (context 2)
FORM_ARM = {
    0: ("T", 1, False),
    1: ("M", 1, False),
    2: ("B", 1, False),
    3: ("W", 4, False),
    7: ("W", 2, False),
    8: ("W", 1, False),
    9: ("0", 4, False),
    12: ("B", 1, True),
    13: ("W", 4, True),
    19: ("0", 4, True),
}


def _cases():
    out = []
    for fmt in FMTS:
        for acc in (DestAccumulation.No, DestAccumulation.Yes):
            if is_dest_acc_needed(fmt) and acc == DestAccumulation.No:
                continue
            for sync in (DestSync.Half, DestSync.Full):
                cap = (
                    (4 if acc == DestAccumulation.Yes else 8)
                    if sync == DestSync.Half
                    else (8 if acc == DestAccumulation.Yes else 16)
                )
                for r0 in R0:
                    for ct in range(2, cap + 1):
                        for kt in KTS:
                            for fid in FIDS:
                                for form in FORM_ARM:
                                    out.append((fmt, acc, sync, r0, ct, kt, fid, form))
    return out


@pytest.mark.perf
@skip_for_wormhole
@parametrize(case=_cases())
def test_perf_matmul_tiny9(case, perf_report):
    case = case[0] if len(case) == 1 else case
    formats, dest_acc, dest_sync, r0, ct, kt, math_fidelity, form = case
    arm, cbd, ctx2 = FORM_ARM[form]
    tile_dims = _gtd(([32, 32 * kt], [32 * kt, 32 * ct]), in0_tile_r_dim=r0)
    face = FaceLayoutConfig(
        num_faces_in0=2,
        num_faces_in1=4,
        num_faces=calculate_matmul_output_faces(
            num_faces_in0=2, num_faces_in1=4, is_in0_horizontal=True
        ),
        unpack_transpose_faces=Transpose.No,
        unpack_transpose_within_face=Transpose.No,
        partial_face_in0=True,
        partial_face_in1=False,
        partial_face_math=r0 < 16,
        partial_face_pack=True,
    )
    matmul_config = MatmulConfig(
        tile_dimensions=tile_dims,
        face_layout_config=face,
        formats=formats,
        stochastic_rnd=StochasticRounding.No,
        dst_index=0,
        dest_sync=dest_sync,
        dest_acc=dest_acc,
    )
    transpose = face.unpack_transpose_faces
    variant_tile_count = tile_dims.rt_dim * tile_dims.ct_dim * tile_dims.kt_dim
    configuration = PerfConfig(
        "sources/math_matmul_tiny9_test.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1],
        templates=[
            MATH_FIDELITY(math_fidelity),
            DEST_SYNC(dest_sync),
            THROTTLE_LEVEL(0),
            TINY_F32(formats.output_format == DataFormat.Float32),
            PACK_API_SWEEP(
                arm,
                tile_dims.rt_dim * tile_dims.ct_dim,
                matmul=True,
                sfpu=ctx2,
                cbd=cbd,
                faces=face.num_faces,
                face_r=min(r0, 16),
            ),
        ],
        runtimes=[
            DEST_INDEX(form << 8),
            UNPACK_TRANS_FACES(transpose),
            UNPACK_TRANS_WITHIN_FACE(transpose),
            TILE_COUNT(variant_tile_count),
            NUM_BLOCKS(1),
            NUM_TILES_IN_BLOCK(tile_dims.rt_dim * tile_dims.ct_dim),
            NUM_FACES(face.num_faces, face.num_faces_in0, face.num_faces_in1),
            PARTIAL_FACE(
                partial_a=face.partial_face_in0,
                partial_face_pack=face.partial_face_pack,
                partial_b=face.partial_face_in1,
                partial_face_math=face.partial_face_math,
            ),
            CRK_TILE_DIMM(tile_dims.ct_dim, tile_dims.rt_dim, tile_dims.kt_dim),
            IN_TILE_DIMS(
                tile_dims.in0_tile_r_dim,
                tile_dims.in0_tile_c_dim,
                tile_dims.in1_tile_r_dim,
                tile_dims.in1_tile_c_dim,
            ),
            LOOP_FACTOR(64),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_dims.tile_cnt_in0,
            tile_count_B=tile_dims.tile_cnt_in1,
            tile_count_res=tile_dims.output_tile_cnt * cbd,
        ),
        dest_acc=dest_acc,
        **SHIM_CONFIG,
    )
    configuration.run(perf_report)
