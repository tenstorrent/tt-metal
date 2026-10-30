# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tiny tile half of matmul grid 2 (see perf_matmul_grid2): in0 height 1 to 16 with in1 width 32, in0 height 1 to 16 and 32 with in1 width 16, over every SrcA x SrcB x output of
Float16_b, Bfp8_b, Bfp4_b and Float32 (Bfp4_b from face height 4), one block per cell and tile shape, on the math matmul test with per operand tile sizes. Its own module because its report columns differ."""

from typing import Callable, List, Union

import pytest
from helpers.format_config import DataFormat, FormatConfig, is_dest_acc_needed
from helpers.golden_generators import TILE_DIM
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    MathFidelity,
    PerfRunType,
    Transpose,
)
from helpers.matmul_sweep import (
    DEST_RT_CT_BLOCKS,
    PERF_RING_TILES,
    generate_tile_dims,
    matmul_pack_form,
    mid_fill_rt_ct_pairs,
    unpack_matmul_fits_perf_ring,
)
from helpers.param_config import (
    DEST_SYNC_TILE_LIMITS,
    input_output_formats,
    parametrize,
)
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    MATMUL_PHASE_ELIDE,
    CRK_TILE_DIMM,
    DEST_SYNC,
    LOOP_FACTOR,
    MATH_FIDELITY,
    MATMUL_FORM_KEYS,
    MATMUL_K_LOOP,
    MATMUL_ROW_END_FOLD,
    MATMUL_PACK_BLOCK,
    MATMUL_ROW_MOP,
    MATMUL_STREAM_NARROW,
    MATMUL_REUSE_DIR,
    MATMUL_STREAM_Z,
    MATMUL_NB,
    MATMUL_OPERAND_SIZES,
    MATMUL_WIDE_WALK,
    IN_TILE_DIMS,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    PARTIAL_FACE,
    DEST_INDEX,
    UNPACK_TRANS_WITHIN_FACE,
    MATMUL_IN1_ALT,
    MATMUL_UNPACK_TTSYNC,
    NUM_FACES,
    THROTTLE_LEVEL,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
)
from helpers.llk_params import format_tile_sizes

from conftest import skip_for_wormhole
from perf_matmul_grid2 import _tiny_cases, tiny_config


@pytest.mark.perf
@skip_for_wormhole
@parametrize(case=_tiny_cases())
def test_perf_matmul_grid2_tiny(case, perf_report):
    case = case[0] if len(case) == 1 else case
    a, b, o, acc, sy, math_fidelity, r0, c1, rt, ct, kt = case
    matmul_config = tiny_config(a, b, o, acc, sy, r0, c1, rt, ct, kt)
    throttle = 0
    num_blocks = 1
    formats = matmul_config.formats
    transpose = matmul_config.face_layout_config.unpack_transpose_faces
    num_faces_in0 = matmul_config.face_layout_config.num_faces_in0
    num_faces_in1 = matmul_config.face_layout_config.num_faces_in1
    num_faces = matmul_config.face_layout_config.num_faces
    run_types = [
        PerfRunType.L1_TO_L1,
        PerfRunType.UNPACK_ISOLATE,
        PerfRunType.MATH_ISOLATE,
        PerfRunType.PACK_ISOLATE,
        PerfRunType.L1_CONGESTION,
    ]

    variant_tile_count = (
        matmul_config.tile_dimensions.rt_dim
        * matmul_config.tile_dimensions.ct_dim
        * matmul_config.tile_dimensions.kt_dim
    )


    configuration = PerfConfig(
        "sources/math_matmul_test.cpp",
        formats,
        run_types,
        templates=[
            MATH_FIDELITY(math_fidelity),
            MATMUL_OPERAND_SIZES(),
            *MATMUL_PHASE_ELIDE.for_block(formats, math_fidelity, matmul_config.dest_acc, throttle, matmul_config.face_layout_config.unpack_transpose_faces, MATMUL_PHASE_ELIDE.keeps_requested(matmul_config.tile_dimensions, matmul_config.dest_sync, math_fidelity)),
            DEST_SYNC(matmul_config.dest_sync),
            THROTTLE_LEVEL(throttle),
            *MATMUL_STREAM_NARROW.for_block(
                formats,
                matmul_config.dest_acc,
                matmul_config.dest_sync,
                matmul_config.tile_dimensions.ct_dim,
                matmul_config.tile_dimensions.rt_dim,
                matmul_config.tile_dimensions.in0_tile_r_dim,
                math_fidelity=math_fidelity,
            ),
            *MATMUL_REUSE_DIR.for_block(
                formats,
                math_fidelity,
                matmul_config.tile_dimensions.ct_dim,
                matmul_config.tile_dimensions.rt_dim,
                matmul_config.dest_sync,
            ),
            *MATMUL_STREAM_Z.for_block(
                formats,
                matmul_config.dest_acc,
                matmul_config.dest_sync,
                matmul_config.tile_dimensions.ct_dim,
                matmul_config.tile_dimensions.rt_dim,
                matmul_config.tile_dimensions.in0_tile_r_dim,
                math_fidelity=math_fidelity,
                in1_faces=num_faces_in1,
                in1_partial=matmul_config.face_layout_config.partial_face_in1,
                kt_dim=matmul_config.tile_dimensions.kt_dim,
                throttle=throttle,
            ),
            *MATMUL_K_LOOP.for_block(
                formats,
                matmul_config.dest_acc,
                math_fidelity,
                matmul_config.tile_dimensions.ct_dim,
                matmul_config.tile_dimensions.rt_dim,
                matmul_config.tile_dimensions.kt_dim,
                matmul_config.tile_dimensions.in0_tile_r_dim,
                matmul_config.tile_dimensions.in0_tile_c_dim,
                matmul_config.tile_dimensions.in1_tile_r_dim,
                matmul_config.tile_dimensions.in1_tile_c_dim,
                dest_sync=matmul_config.dest_sync,
                throttle=throttle,
            ),
            *MATMUL_ROW_END_FOLD.for_block(matmul_config.tile_dimensions.ct_dim, matmul_config.tile_dimensions.rt_dim),
            *(
                MATMUL_ROW_MOP.for_block(
                    formats,
                    matmul_config.dest_acc,
                    math_fidelity,
                    matmul_config.dest_sync,
                    matmul_config.tile_dimensions.ct_dim,
                    matmul_config.tile_dimensions.rt_dim,
                    matmul_config.tile_dimensions.kt_dim,
                    num_blocks,
                )
                if math_fidelity == MathFidelity.LoFi and matmul_config.dest_sync == DestSync.Full
                else []
            ),
            *MATMUL_FORM_KEYS.for_block(
                formats,
                matmul_config.tile_dimensions.kt_dim,
                matmul_config.tile_dimensions.ct_dim,
                matmul_config.tile_dimensions.rt_dim,
                math_fidelity=math_fidelity,
                dest_sync=matmul_config.dest_sync,
                in0_r_dim=matmul_config.tile_dimensions.in0_tile_r_dim,
                unfuse_cells=True,
                dest_acc=matmul_config.dest_acc,
                throttle=throttle,
                sections=num_blocks,
            ),
            *MATMUL_IN1_ALT.for_tiles(
                formats,
                matmul_config.dest_acc,
                math_fidelity,
                matmul_config.tile_dimensions.in0_tile_r_dim,
                matmul_config.tile_dimensions.in0_tile_c_dim,
                matmul_config.tile_dimensions.in1_tile_r_dim,
                matmul_config.tile_dimensions.in1_tile_c_dim,
            ),
            MATMUL_PACK_BLOCK(
                matmul_pack_form(
                    formats,
                    matmul_config.dest_sync,
                    math_fidelity,
                    matmul_config.tile_dimensions.rt_dim,
                    matmul_config.tile_dimensions.ct_dim,
                    matmul_config.tile_dimensions.kt_dim,
                    matmul_config.tile_dimensions.in0_tile_r_dim,
                    matmul_config.tile_dimensions.in1_tile_c_dim,
                    throttle,
                    dense_pages=True,
                )
            ),
        ],
        runtimes=[
            DEST_INDEX(matmul_config.dst_index),
            UNPACK_TRANS_FACES(transpose),
            UNPACK_TRANS_WITHIN_FACE(transpose),
            TILE_COUNT(variant_tile_count * num_blocks),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(
                matmul_config.tile_dimensions.rt_dim
                * matmul_config.tile_dimensions.ct_dim
            ),
            NUM_FACES(
                num_faces, num_faces_in0, num_faces_in1
            ),  # In0 -> Input A, In1 -> Input B
            PARTIAL_FACE(  # In0 -> Input A, In1 -> Input B
                partial_a=matmul_config.face_layout_config.partial_face_in0,
                partial_face_pack=matmul_config.face_layout_config.partial_face_pack,
                partial_b=matmul_config.face_layout_config.partial_face_in1,
                partial_face_math=matmul_config.face_layout_config.partial_face_math,
            ),
            CRK_TILE_DIMM(
                matmul_config.tile_dimensions.ct_dim,
                matmul_config.tile_dimensions.rt_dim,
                matmul_config.tile_dimensions.kt_dim,
            ),
            IN_TILE_DIMS(
                matmul_config.tile_dimensions.in0_tile_r_dim,
                matmul_config.tile_dimensions.in0_tile_c_dim,
                matmul_config.tile_dimensions.in1_tile_r_dim,
                matmul_config.tile_dimensions.in1_tile_c_dim,
            ),
            LOOP_FACTOR(1024),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format_B,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=matmul_config.tile_dimensions.tile_cnt_in0,
            tile_count_B=matmul_config.tile_dimensions.tile_cnt_in1,
            tile_count_res=matmul_config.tile_dimensions.output_tile_cnt * num_blocks,
            num_faces_A=num_faces_in0,
            face_r_dim_A=min(matmul_config.tile_dimensions.in0_tile_r_dim, 16),
            tile_dimensions_A=[matmul_config.tile_dimensions.in0_tile_r_dim, matmul_config.tile_dimensions.in0_tile_c_dim],
            num_faces_B=num_faces_in1,
            tile_dimensions_B=[matmul_config.tile_dimensions.in1_tile_r_dim, matmul_config.tile_dimensions.in1_tile_c_dim],
        ),
        dest_acc=matmul_config.dest_acc,
    )

    configuration.run(perf_report)
