# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

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
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_face_matmul_data
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    IN_TILE_DIMS,
    INPUT_DIMENSIONS,
    MATH_FIDELITY,
    NUM_FACES,
    PARTIAL_FACE,
    STOCHASTIC_ROUNDING,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test

# Generate format-aware combinations
MATMUL_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float16,
        DataFormat.Float32,
    ]
)
DEST_ACC_MODES = [DestAccumulation.No, DestAccumulation.Yes]
STOCHASTIC_ROUNDING_MODES = [
    StochasticRounding.No,
    StochasticRounding.Fpu,
    StochasticRounding.Pack,
    StochasticRounding.All,
]

FACE_MODES = [1, 2, 4]
TRANSPOSE_MODES = [Transpose.No, Transpose.Yes]
DEST_SYNC_MODES = [DestSync.Half]

MATMUL_COMBINATIONS = sweep_matmul(
    MATMUL_FORMATS,
    DEST_ACC_MODES,
    STOCHASTIC_ROUNDING_MODES,
    DEST_SYNC_MODES,
)

TINY_TILES_MATMUL_COMBINATIONS = sweep_tiny_tiles_matmul(
    MATMUL_FORMATS,
    DEST_ACC_MODES,
    STOCHASTIC_ROUNDING_MODES,
    DEST_SYNC_MODES,
)


@pytest.mark.nightly
@parametrize(
    math_fidelity=[
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
    matmul_config=MATMUL_COMBINATIONS + TINY_TILES_MATMUL_COMBINATIONS,
)
def test_unpack_matmul(math_fidelity, matmul_config, workers_tensix_coordinates):

    formats = matmul_config.formats
    dest_acc = matmul_config.dest_acc
    input_A_dimensions = matmul_config.tile_dimensions.input_A_dimensions
    input_B_dimensions = matmul_config.tile_dimensions.input_B_dimensions
    num_faces_A = matmul_config.face_layout_config.num_faces_A
    num_faces_B = matmul_config.face_layout_config.num_faces_B
    num_faces = matmul_config.face_layout_config.num_faces
    partial_face_A = matmul_config.face_layout_config.partial_face_A
    partial_face_B = matmul_config.face_layout_config.partial_face_B
    transpose = matmul_config.face_layout_config.unpack_transpose_faces

    # Generate test data for all tiles with the right faces zeroed out
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
            num_tiles=matmul_config.tile_dimensions.tile_cnt_B,
            tilize=True,
            input_dimensions=input_B_dimensions,
        )
        src_B_golden = t_matrix.transpose_within_faces_multi_tile(
            src_B_golden,
            formats.input_format,
            num_tiles=matmul_config.tile_dimensions.tile_cnt_B,
            untilize=True,
            input_dimensions=input_B_dimensions,
        )

    generate_golden = get_golden_generator(MatmulGolden)

    # Generate standard golden reference (PCC validation will handle stochastic tolerance)
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
        "sources/unpack_matmul_test.cpp",
        formats,
        templates=[
            INPUT_DIMENSIONS(input_A_dimensions, input_B_dimensions),
            STOCHASTIC_ROUNDING(matmul_config.stochastic_rnd),
            MATH_FIDELITY(math_fidelity),
        ],
        runtimes=[
            TILE_COUNT(matmul_config.tile_dimensions.tile_cnt),
            NUM_FACES(num_faces, num_faces_A, num_faces_B),
            UNPACK_TRANS_FACES(transpose),
            UNPACK_TRANS_WITHIN_FACE(transpose),
            PARTIAL_FACE(
                partial_a=partial_face_A,
                partial_face_pack=partial_face_A,
                partial_b=partial_face_B,
                partial_face_math=partial_face_B,
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
        dest_acc=dest_acc,
    )
    res_from_L1 = configuration.run(workers_tensix_coordinates)

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golder tensor are not of the same length"

    # Only compare the active faces in each tile since that's what the hardware processes
    num_elements_per_tile = num_faces * 256  # Each face is 16x16 = 256 elements
    tile_cnt = matmul_config.tile_dimensions.output_tile_cnt

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

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
