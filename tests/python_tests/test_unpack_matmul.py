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
from helpers.stimuli_generator import convert_to_l1_view, generate_face_matmul_data
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
    in0_dimensions = matmul_config.tile_dimensions.in0_dimensions
    in1_dimensions = matmul_config.tile_dimensions.in1_dimensions
    in0_tile_r_dim = matmul_config.tile_dimensions.in0_tile_r_dim
    in0_tile_c_dim = matmul_config.tile_dimensions.in0_tile_c_dim
    in1_tile_r_dim = matmul_config.tile_dimensions.in1_tile_r_dim
    in1_tile_c_dim = matmul_config.tile_dimensions.in1_tile_c_dim
    num_faces_in0 = matmul_config.face_layout_config.num_faces_in0
    num_faces_in1 = matmul_config.face_layout_config.num_faces_in1
    num_faces = matmul_config.face_layout_config.num_faces
    partial_face_in0 = matmul_config.face_layout_config.partial_face_in0
    partial_face_in1 = matmul_config.face_layout_config.partial_face_in1
    partial_face_pack = matmul_config.face_layout_config.partial_face_pack
    partial_face_math = matmul_config.face_layout_config.partial_face_math
    transpose = matmul_config.face_layout_config.unpack_transpose_faces

    # Generate test data for all tiles with the right faces zeroed out
    in0 = generate_face_matmul_data(
        num_faces=num_faces_in0,
        stimuli_format=formats.input_format,
        input_dimensions=in0_dimensions,  # This will generate the right number of tiles
        is_matrix_A=True,  # input 0 (SrcB) uses f0,f1 for 2-face mode
        face_r_dim=(in0_tile_r_dim if in0_tile_r_dim < 16 else 16),
    )

    in1 = generate_face_matmul_data(
        num_faces=num_faces_in1,
        stimuli_format=formats.input_format,
        input_dimensions=in1_dimensions,  # This will generate the right number of tiles
        is_matrix_A=False,  # input 1 (SrcA) uses f0,f2 for 2-face mode
    )

    in1_golden = in1
    if transpose == Transpose.Yes:
        t_matrix = get_golden_generator(TransposeGolden)
        in1_golden = t_matrix.transpose_faces_multi_tile(
            in1,
            formats.input_format,
            num_tiles=matmul_config.tile_dimensions.tile_cnt_in1,
            tilize=True,
            input_dimensions=in1_dimensions,
        )
        in1_golden = t_matrix.transpose_within_faces_multi_tile(
            in1_golden,
            formats.input_format,
            num_tiles=matmul_config.tile_dimensions.tile_cnt_in1,
            untilize=True,
            input_dimensions=in1_dimensions,
        )

    generate_golden = get_golden_generator(MatmulGolden)

    # Generate standard golden reference (PCC validation will handle stochastic tolerance)
    golden_tensor = generate_golden(
        in0,
        in1_golden,
        formats.output_format,
        math_fidelity,
        input_A_dimensions=in0_dimensions,
        input_B_dimensions=in1_dimensions,
        tilize=True,  # Golden cannot model FPU strided for tilized data computation, so we tilize output after computation
    )

    tilized_in0 = tilize_block(
        in0, dimensions=in0_dimensions, stimuli_format=formats.input_format
    )
    tilized_in1 = tilize_block(
        in1, dimensions=in1_dimensions, stimuli_format=formats.input_format
    )
    tilized_in0_l1_view = convert_to_l1_view(
        tilized_in0, in0_dimensions, tile_dimensions=[in0_tile_r_dim, in0_tile_c_dim]
    )
    tilized_in1_l1_view = convert_to_l1_view(
        tilized_in1, in1_dimensions, tile_dimensions=[in1_tile_r_dim, in1_tile_c_dim]
    )

    configuration = TestConfig(
        "sources/unpack_matmul_test.cpp",
        formats,
        templates=[
            STOCHASTIC_ROUNDING(matmul_config.stochastic_rnd),
            MATH_FIDELITY(math_fidelity),
            THROTTLE_LEVEL(0),
            DEST_SYNC(matmul_config.dest_sync),
        ],
        runtimes=[
            TILE_COUNT(matmul_config.tile_dimensions.tile_cnt),
            NUM_FACES(num_faces, num_faces_in0, num_faces_in1),
            UNPACK_TRANS_FACES(transpose),
            UNPACK_TRANS_WITHIN_FACE(transpose),
            PARTIAL_FACE(
                partial_a=partial_face_in0,
                partial_face_pack=partial_face_pack,
                partial_b=partial_face_in1,
                partial_face_math=partial_face_math,
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
            tilized_in0_l1_view.flatten(),
            formats.input_format,
            tilized_in1_l1_view.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=matmul_config.tile_dimensions.tile_cnt_in0,
            tile_count_B=matmul_config.tile_dimensions.tile_cnt_in1,
            tile_count_res=matmul_config.tile_dimensions.tile_cnt,
        ),
        dest_acc=dest_acc,
    )
    res_from_L1 = configuration.run(workers_tensix_coordinates).result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    # Convert golden tensor to L1 memory view for comparison
    golden_tensor = convert_to_l1_view(
        golden_tensor,
        (in0_dimensions[0], in1_dimensions[1]),
        tile_dimensions=[in0_tile_r_dim, in1_tile_c_dim],
    )

    if num_faces < 4:
        num_elements_per_tile = in0_tile_r_dim * in1_tile_c_dim
        tile_cnt = matmul_config.tile_dimensions.output_tile_cnt

        # Compare each tile separately
        TILE_R_DIM, TILE_C_DIM = 32, 32
        for i in range(tile_cnt):
            start = i * (TILE_R_DIM * TILE_C_DIM)
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
