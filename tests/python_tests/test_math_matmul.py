# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from itertools import chain, product

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    MatmulGolden,
    TransposeGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    MathFidelity,
    StochasticRounding,
    Transpose,
    format_dict,
)
from helpers.matmul_sweep import sweep_matmul, sweep_tiny_tiles_matmul
from helpers.param_config import input_output_formats
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_face_matmul_data
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    DEST_INDEX,
    DEST_SYNC,
    IN_TILE_DIMS,
    MATH_FIDELITY,
    NUM_FACES,
    PARTIAL_FACE,
    STOCHASTIC_ROUNDING,
    THROTTLE_LEVEL,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
    generate_input_dim,
)
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test

MATMUL_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float16,
        DataFormat.Float32,
    ]
)
DEST_ACC_MODES = [DestAccumulation.No, DestAccumulation.Yes]
DEST_SYNC_MODES = [DestSync.Half, DestSync.Full]
STOCHASTIC_ROUNDING_MODES = [StochasticRounding.No]
MATH_FIDELITIES = [
    MathFidelity.LoFi,
    MathFidelity.HiFi2,
    MathFidelity.HiFi3,
    MathFidelity.HiFi4,
]
MATMUL_COMBINATIONS = sweep_matmul(
    MATMUL_FORMATS,
    DEST_ACC_MODES,
    STOCHASTIC_ROUNDING_MODES,
    DEST_SYNC_MODES,
    math_matmul=True,
)
TINY_TILES_MATMUL_COMBINATIONS = sweep_tiny_tiles_matmul(
    MATMUL_FORMATS,
    DEST_ACC_MODES,
    STOCHASTIC_ROUNDING_MODES,
    DEST_SYNC_MODES,
    math_matmul=True,
)


ALL_TEST_PARAMS = list(
    chain(
        # Regular matmul with all throttle levels
        (
            (fidelity, combinations, throttle)
            for fidelity, combinations, throttle in product(
                MATH_FIDELITIES, MATMUL_COMBINATIONS, [1, 2, 3, 4, 5]
            )
        ),
        # Tiny tiles matmul with throttle level 1 only
        (
            (fidelity, combinations, 0)
            for fidelity, combinations in product(
                MATH_FIDELITIES, TINY_TILES_MATMUL_COMBINATIONS
            )
        ),
    )
)


@pytest.mark.nightly
@pytest.mark.parametrize("math_fidelity,matmul_config,throttle", ALL_TEST_PARAMS)
def test_math_matmul(
    math_fidelity, matmul_config, throttle, workers_tensix_coordinates
):
    formats = matmul_config.formats
    input_A_dimensions = matmul_config.tile_dimensions.input_A_dimensions
    input_B_dimensions = matmul_config.tile_dimensions.input_B_dimensions
    transpose = matmul_config.face_layout_config.unpack_transpose_faces
    tile_cnt_B = matmul_config.tile_dimensions.tile_cnt_B
    num_faces_A = matmul_config.face_layout_config.num_faces_A
    num_faces_B = matmul_config.face_layout_config.num_faces_B
    num_faces = matmul_config.face_layout_config.num_faces

    torch_format = format_dict[formats.output_format]

    src_A = generate_face_matmul_data(
        num_faces=num_faces_A,
        stimuli_format=formats.input_format,
        input_dimensions=input_A_dimensions,  # This will generate the right number of tiles
        is_matrix_A=True,  # Matrix A (SrcB) uses f0,f2 for 2-face mode
    )

    src_B = generate_face_matmul_data(
        num_faces=num_faces_B,
        stimuli_format=formats.input_format,
        input_dimensions=input_B_dimensions,  # This will generate the right number of tiles
        is_matrix_A=False,  # Matrix B (SrcA) uses f0,f1 for 2-face mode
    )

    src_B_golden = src_B
    if transpose == Transpose.Yes:
        t_matrix = get_golden_generator(TransposeGolden)

        src_B_golden = t_matrix.transpose_faces_multi_tile(
            src_B,
            formats.input_format,
            num_tiles=tile_cnt_B,
            tilize=True,
            input_dimensions=input_B_dimensions,
        )
        src_B_golden = t_matrix.transpose_within_faces_multi_tile(
            src_B_golden,
            formats.input_format,
            num_tiles=tile_cnt_B,
            untilize=True,
            input_dimensions=input_B_dimensions,
        )

    generate_golden = get_golden_generator(MatmulGolden)
    golden_tensor = generate_golden(
        src_A,
        src_B_golden,
        formats.output_format,
        math_fidelity,
        input_A_dimensions=input_A_dimensions,
        input_B_dimensions=input_B_dimensions,
        tilize=True,  # Golden cannot model FPU strided for tilized data computation, so we tilize output after computation
    )

    tilized_A = tilize_block(
        src_A, dimensions=input_A_dimensions, stimuli_format=formats.input_format
    )
    tilized_B = tilize_block(
        src_B, dimensions=input_B_dimensions, stimuli_format=formats.input_format
    )

    configuration = TestConfig(
        "sources/math_matmul_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_A_dimensions, input_B_dimensions),
            STOCHASTIC_ROUNDING(matmul_config.stochastic_rnd),
            MATH_FIDELITY(math_fidelity),
            THROTTLE_LEVEL(throttle),
            DEST_SYNC(matmul_config.dest_sync),
        ],
        runtimes=[
            TILE_COUNT(matmul_config.tile_dimensions.tile_cnt),
            NUM_FACES(num_faces, num_faces_A, num_faces_B),
            UNPACK_TRANS_FACES(transpose),
            UNPACK_TRANS_WITHIN_FACE(transpose),
            PARTIAL_FACE(
                partial_a=matmul_config.face_layout_config.partial_face_A,
                partial_face_pack=matmul_config.face_layout_config.partial_face_A,
                partial_b=matmul_config.face_layout_config.partial_face_B,
                partial_face_math=matmul_config.face_layout_config.partial_face_B,
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
            DEST_INDEX(matmul_config.dst_index),
        ],
        variant_stimuli=StimuliConfig(
            tilized_A.flatten(),
            formats.input_format,
            tilized_B.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=matmul_config.tile_dimensions.tile_cnt_A,
            tile_count_B=matmul_config.tile_dimensions.tile_cnt_B,
            tile_count_res=matmul_config.tile_dimensions.tile_cnt,
        ),
        dest_acc=matmul_config.dest_acc,
    )
    res_from_L1 = configuration.run(workers_tensix_coordinates)

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)
    if num_faces < 4:
        # Only compare the active faces in each tile since that's what the hardware processes
        num_elements_per_tile = num_faces * 256  # Each face is 16x16 = 256 elements
        tile_cnt = matmul_config.tile_dimensions.output_tile_cnt

        # Compare each tile separately
        for i in range(tile_cnt):
            start = i * 1024  # Each tile is 1024 elements
            assert passed_test(
                golden_tensor[
                    start : start + num_elements_per_tile
                ],  # Only compare active faces in this tile
                res_tensor[
                    start : start + num_elements_per_tile
                ],  # Only compare active faces in this tile
                formats.output_format,
            ), f"Assert on tile {i}/{tile_cnt} against golden failed"
    else:
        assert passed_test(
            golden_tensor, res_tensor, formats.output_format
        ), "Assert against golden failed"
