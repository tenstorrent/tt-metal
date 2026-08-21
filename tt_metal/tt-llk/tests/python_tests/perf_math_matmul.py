# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from itertools import product

import pytest
from helpers.format_config import DataFormat, is_dest_acc_needed
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    MathFidelity,
    PerfRunType,
    StochasticRounding,
    Transpose,
)
from helpers.matmul_sweep import FaceLayoutConfig, MatmulConfig, TileDimensions
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
    NUM_FACES,
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
        DataFormat.Float16,
        DataFormat.Float32,
    ]
)
DEST_ACC_MODES = [DestAccumulation.No, DestAccumulation.Yes]
DEST_SYNC_MODES = [DestSync.Half, DestSync.Full]
MATH_FIDELITIES = [
    MathFidelity.LoFi,
    MathFidelity.HiFi2,
    MathFidelity.HiFi3,
    MathFidelity.HiFi4,
]
RT_DIMS = [1, 2, 4, 8]
CT_DIMS = [1, 2, 4, 8, 16]
KT_DIMS = [1, 2, 4]
IN0_TILE_DIMENSIONS = [(1, 32), (2, 32), (4, 32), (8, 32), (16, 32), (32, 32)]
UNPACK_TRANSPOSE_MODES = [Transpose.No, Transpose.Yes]


def generate_experiment_combinations():
    """Generate a focused tile-count scaling experiment.

    RT, CT, and KT are swept independently. The kernel splits RT x CT grids
    larger than destination capacity into destination-sized blocks.
    Unpack transpose Yes is limited to full 32x32 tiles, matching functional.
    """
    combinations = []
    for (
        formats,
        dest_acc,
        dest_sync,
        (in0_tile_rows, in0_tile_cols),
        rt_dim,
        ct_dim,
        kt_dim,
        transpose,
    ) in product(
        MATMUL_FORMATS,
        DEST_ACC_MODES,
        DEST_SYNC_MODES,
        IN0_TILE_DIMENSIONS,
        RT_DIMS,
        CT_DIMS,
        KT_DIMS,
        UNPACK_TRANSPOSE_MODES,
    ):
        if is_dest_acc_needed(formats) and dest_acc == DestAccumulation.No:
            continue

        is_tiny_tile = in0_tile_rows < 32
        if is_tiny_tile and transpose == Transpose.Yes:
            continue

        output_tile_cnt = rt_dim * ct_dim
        num_faces_in0 = 2 if is_tiny_tile else 4
        output_num_faces = 2 if is_tiny_tile else 4

        combinations.append(
            MatmulConfig(
                tile_dimensions=TileDimensions(
                    in0_dimensions=(rt_dim * in0_tile_rows, kt_dim * 32),
                    in1_dimensions=(kt_dim * 32, ct_dim * 32),
                    output_dimensions=(rt_dim * in0_tile_rows, ct_dim * 32),
                    rt_dim=rt_dim,
                    ct_dim=ct_dim,
                    kt_dim=kt_dim,
                    tile_cnt=output_tile_cnt,
                    tile_cnt_in0=rt_dim * kt_dim,
                    tile_cnt_in1=kt_dim * ct_dim,
                    output_tile_cnt=output_tile_cnt,
                    in0_tile_r_dim=in0_tile_rows,
                    in0_tile_c_dim=in0_tile_cols,
                    in1_tile_r_dim=32,
                    in1_tile_c_dim=32,
                ),
                face_layout_config=FaceLayoutConfig(
                    unpack_transpose_faces=transpose,
                    unpack_transpose_within_face=transpose,
                    num_faces_in0=num_faces_in0,
                    num_faces_in1=4,
                    num_faces=output_num_faces,
                    partial_face_in0=is_tiny_tile,
                    partial_face_in1=False,
                    partial_face_math=in0_tile_rows < 16,
                    partial_face_pack=is_tiny_tile,
                ),
                formats=formats,
                stochastic_rnd=StochasticRounding.No,
                dst_index=0,
                dest_sync=dest_sync,
                dest_acc=dest_acc,
            )
        )
    return combinations


MATMUL_COMBINATIONS = generate_experiment_combinations()
ALL_TEST_PARAMS = [
    (fidelity, combination, 0)
    for fidelity, combination in product(MATH_FIDELITIES, MATMUL_COMBINATIONS)
]


@pytest.mark.perf
@pytest.mark.parametrize("math_fidelity,matmul_config,throttle", ALL_TEST_PARAMS)
def test_perf_math_matmul(
    math_fidelity,
    matmul_config,
    throttle,
    perf_report,
):
    """
    Matmul performance scaling experiment for 1/2/4/8/16x32 and 32x32 input-0 tiles.

    RT is 1, 2, 4, or 8; CT is 1, 2, 4, 8, or 16; KT is 1, 2, or 4.
    Full 32x32 tiles also sweep unpack transpose Yes.
    """
    formats = matmul_config.formats
    in0_dimensions = matmul_config.tile_dimensions.in0_dimensions
    in1_dimensions = matmul_config.tile_dimensions.in1_dimensions
    transpose = matmul_config.face_layout_config.unpack_transpose_faces
    num_faces_in0 = matmul_config.face_layout_config.num_faces_in0
    num_faces_in1 = matmul_config.face_layout_config.num_faces_in1
    num_faces = matmul_config.face_layout_config.num_faces

    if is_dest_acc_needed(formats) and matmul_config.dest_acc == DestAccumulation.No:
        pytest.skip("Dest accumulation must be enabled for this format")

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
        "sources/math_matmul_perf.cpp",
        formats,
        run_types,
        templates=[
            MATH_FIDELITY(math_fidelity),
            DEST_SYNC(matmul_config.dest_sync),
            THROTTLE_LEVEL(throttle),
        ],
        runtimes=[
            DEST_INDEX(matmul_config.dst_index),
            UNPACK_TRANS_FACES(transpose),
            UNPACK_TRANS_WITHIN_FACE(transpose),
            TILE_COUNT(variant_tile_count),
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
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=matmul_config.tile_dimensions.tile_cnt_in0,
            tile_count_B=matmul_config.tile_dimensions.tile_cnt_in1,
            tile_count_res=matmul_config.tile_dimensions.output_tile_cnt,
        ),
        dest_acc=matmul_config.dest_acc,
    )

    configuration.run(perf_report)
