# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from itertools import chain, product

import pytest
import torch
from helpers.device import collect_results, write_stimuli_to_l1
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
from helpers.stimuli_generator import generate_face_matmul_data
from helpers.test_config import run_test
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
            ("math_matmul_test", fidelity, combinations, throttle)
            for fidelity, combinations, throttle in product(
                MATH_FIDELITIES, MATMUL_COMBINATIONS, [1, 2, 3, 4, 5]
            )
        ),
        # Tiny tiles matmul with throttle level 1 only
        (
            ("math_matmul_test", fidelity, combinations, 0)
            for fidelity, combinations in product(
                MATH_FIDELITIES, TINY_TILES_MATMUL_COMBINATIONS
            )
        ),
    )
)


@pytest.mark.nightly
@pytest.mark.parametrize(
    "test_name,math_fidelity,matmul_config,throttle", ALL_TEST_PARAMS
)
def test_math_matmul(test_name, math_fidelity, matmul_config, throttle):

    formats = matmul_config.formats
    dest_acc = matmul_config.dest_acc
    input_A_dimensions = matmul_config.tile_dimensions.input_A_dimensions
    input_B_dimensions = matmul_config.tile_dimensions.input_B_dimensions
    transpose = matmul_config.face_layout_config.unpack_transpose_faces
    stochastic_rnd = matmul_config.stochastic_rnd
    tile_cnt_A = matmul_config.tile_dimensions.tile_cnt_A
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

    test_config = {
        "formats": formats,
        "testname": test_name,
        "dest_acc": dest_acc,
        "math_fidelity": math_fidelity,
        "tile_cnt": matmul_config.tile_dimensions.tile_cnt,
        "input_A_dimensions": input_A_dimensions,
        "input_B_dimensions": input_B_dimensions,
        "output_dimensions": matmul_config.tile_dimensions.output_dimensions,
        "rt_dim": matmul_config.tile_dimensions.rt_dim,
        "ct_dim": matmul_config.tile_dimensions.ct_dim,
        "kt_dim": matmul_config.tile_dimensions.kt_dim,
        "in0_tile_r_dim": matmul_config.tile_dimensions.in0_tile_r_dim,
        "in0_tile_c_dim": matmul_config.tile_dimensions.in0_tile_c_dim,
        "in1_tile_r_dim": matmul_config.tile_dimensions.in1_tile_r_dim,
        "in1_tile_c_dim": matmul_config.tile_dimensions.in1_tile_c_dim,
        "stochastic_rnd": stochastic_rnd,
        "unpack_transpose_faces": transpose,
        "unpack_transpose_within_face": transpose,  # matmul transposes both faces and within faces, there is no option for one or the other
        "num_faces_A": num_faces_A,  # Number of active faces for matrix A
        "num_faces_B": num_faces_B,  # Number of active faces for matrix B
        "num_faces": num_faces,  # Number of active faces for result matrix
        "partial_face_A": matmul_config.face_layout_config.partial_face_A,  # Partial face setting for matrix A
        "partial_face_B": matmul_config.face_layout_config.partial_face_B,  # Partial face setting for matrix B
        "tiny_tiles": matmul_config.tile_dimensions.in0_tile_r_dim <= 16,
        "throttle": throttle,
        "dst_index": matmul_config.dst_index,
        "dest_sync": matmul_config.dest_sync,
    }

    res_address = write_stimuli_to_l1(
        test_config,
        tilized_A.flatten(),
        tilized_B.flatten(),
        formats.input_format,
        formats.input_format,
        tile_count_A=tile_cnt_A,
        tile_count_B=tile_cnt_B,
    )

    run_test(test_config)

    res_from_L1 = collect_results(
        formats, tile_count=matmul_config.tile_dimensions.tile_cnt, address=res_address
    )
    assert len(res_from_L1) == len(golden_tensor)

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
            )
    else:
        assert passed_test(golden_tensor, res_tensor, formats.output_format)
