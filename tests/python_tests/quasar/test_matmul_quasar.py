# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    MAX_TILES_16_BIT_DEST,
    MAX_TILES_32_BIT_DEST,
    TILE_DIM,
    MatmulGolden,
    TransposeGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathFidelity,
    Transpose,
    format_dict,
)
from helpers.matmul_sweep import (
    generate_tile_dims,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import BootMode, TestConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    MATH_FIDELITY,
    NUM_FACES,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    generate_input_dim,
)
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test

kt_dims = [1, 2, 4]
matmul_dimensions_32_bit_dest = [
    (
        [mt_dim * TILE_DIM, kt_dim * TILE_DIM],
        [kt_dim * TILE_DIM, nt_dim * TILE_DIM],
        DestAccumulation.Yes,
    )
    for mt_dim in range(1, MAX_TILES_32_BIT_DEST + 1)
    for nt_dim in range(1, MAX_TILES_32_BIT_DEST // mt_dim + 1)
    for kt_dim in kt_dims
]
matmul_dimensions_16_bit_dest = [
    (
        [mt_dim * TILE_DIM, kt_dim * TILE_DIM],
        [kt_dim * TILE_DIM, nt_dim * TILE_DIM],
        DestAccumulation.No,
    )
    for mt_dim in range(1, MAX_TILES_16_BIT_DEST + 1)
    for nt_dim in range(1, MAX_TILES_16_BIT_DEST // mt_dim + 1)
    for kt_dim in kt_dims
]

# Generate format-aware combinations
MATMUL_FORMAT = input_output_formats(
    [
        DataFormat.Float16,
        DataFormat.Float16_b,
    ],
)


@pytest.mark.quasar
@parametrize(
    implied_math_format=[ImpliedMathFormat.No, ImpliedMathFormat.Yes],
    math_fidelity=[
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
    dimensions_dest_acc=matmul_dimensions_32_bit_dest + matmul_dimensions_16_bit_dest,
    format=MATMUL_FORMAT,
    dest_sync_mode=[DestSync.Half, DestSync.Full],
    transpose=[Transpose.No],
)
# Note: this test is used to test boot modes, that is why it has them piped as default arguments to the test itself
def test_matmul(
    math_fidelity,
    dimensions_dest_acc,
    format,
    implied_math_format,
    dest_sync_mode,
    transpose,
):
    input_A_dimensions, input_B_dimensions, dest_acc = dimensions_dest_acc

    torch_format = format_dict[format.output_format]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=format.input_format,
        input_dimensions_A=input_A_dimensions,
        stimuli_format_B=format.input_format,
        input_dimensions_B=input_B_dimensions,
        sfpu=False,
    )

    src_B_golden = src_B
    if transpose == Transpose.Yes:
        t_matrix = get_golden_generator(TransposeGolden)

        src_B_golden = t_matrix.transpose_faces_multi_tile(
            src_B,
            format.input_format,
            num_tiles=tile_cnt_B,
            tilize=True,
            input_dimensions=input_B_dimensions,
        )
        src_B_golden = t_matrix.transpose_within_faces_multi_tile(
            src_B_golden,
            format.input_format,
            num_tiles=tile_cnt_B,
            untilize=True,
            input_dimensions=input_B_dimensions,
        )

    # Calculate all matmul dimensions using helper function
    matmul_dims = generate_tile_dims((input_A_dimensions, input_B_dimensions))

    generate_golden = get_golden_generator(MatmulGolden)
    golden_tensor = generate_golden(
        src_A,
        src_B_golden,
        format.output_format,
        math_fidelity,
        input_A_dimensions=input_A_dimensions,
        input_B_dimensions=input_B_dimensions,
        tilize=True,  # Golden cannot model FPU strided for tilized data computation, so we tilize output after computation
    )

    tilized_A = tilize_block(
        src_A, dimensions=input_A_dimensions, stimuli_format=format.input_format
    )
    tilized_B = tilize_block(
        src_B, dimensions=input_B_dimensions, stimuli_format=format.input_format
    )

    num_faces = 4

    configuration = TestConfig(
        "sources/quasar/matmul_quasar_test.cpp",
        format,
        templates=[
            MATH_FIDELITY(math_fidelity),
            generate_input_dim(input_A_dimensions, input_B_dimensions),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DEST_SYNC(dest_sync_mode),
            UNPACK_TRANS_FACES(transpose),
            CRK_TILE_DIMM(matmul_dims.ct_dim, matmul_dims.rt_dim, matmul_dims.kt_dim),
            TILE_COUNT(matmul_dims.output_tile_cnt),
            NUM_FACES(num_faces, num_faces, num_faces),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            tilized_A.flatten(),
            format.input_format,
            tilized_B.flatten(),
            format.input_format,
            format.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=matmul_dims.output_tile_cnt,
            num_faces=num_faces,
        ),
        unpack_to_dest=False,
        dest_acc=dest_acc,
        boot_mode=BootMode.TRISC,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, format.output_format
    ), "Assert against golden failed"
