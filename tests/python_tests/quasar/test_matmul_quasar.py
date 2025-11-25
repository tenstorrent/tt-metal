# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from helpers.device import BootMode, collect_results, write_stimuli_to_l1
from helpers.format_config import DataFormat
from helpers.golden_generators import (
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
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import run_test
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test

TILE_DIM = 32  # Standard tile dimension for row and column
MAX_TILES_32_BIT_DEST = 4
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
MAX_TILES_16_BIT_DEST = 8
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
DEST_ACC_MODES = [DestAccumulation.No, DestAccumulation.Yes]
IMPLIED_MATH_FORMAT = [ImpliedMathFormat.No, ImpliedMathFormat.Yes]
DEST_SYNC_MODES = [DestSync.Half, DestSync.Full]
TRANSPOSE_MODES = [Transpose.No]


@pytest.mark.quasar
@parametrize(
    test_name="matmul_quasar_test",
    implied_math_format=IMPLIED_MATH_FORMAT,
    math_fidelity=[
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
    dimensions_dest_acc=matmul_dimensions_32_bit_dest + matmul_dimensions_16_bit_dest,
    format=MATMUL_FORMAT,
    dest_sync_mode=DEST_SYNC_MODES,
    transpose=TRANSPOSE_MODES,
)
# Note: this test is used to test boot modes, that is why it has them piped as default arguments to the test itself
def test_matmul(
    test_name,
    math_fidelity,
    dimensions_dest_acc,
    format,
    implied_math_format,
    dest_sync_mode,
    transpose,
):
    input_A_dimensions, input_B_dimensions, dest_acc = dimensions_dest_acc

    torch_format = format_dict[format.output_format]

    src_A, _, tile_cnt_A = generate_stimuli(
        format.input_format,
        format.input_format,
        input_dimensions=input_A_dimensions,
        sfpu=False,
    )
    src_B, _, tile_cnt_B = generate_stimuli(
        format.input_format,
        format.input_format,
        input_dimensions=input_B_dimensions,
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

    test_config = {
        "formats": format,
        "testname": test_name,
        "dest_acc": dest_acc,
        "math_fidelity": math_fidelity,
        "tile_cnt": matmul_dims.output_tile_cnt,
        "input_A_dimensions": input_A_dimensions,
        "input_B_dimensions": input_B_dimensions,
        "output_dimensions": matmul_dims.output_dimensions,
        "rt_dim": matmul_dims.rt_dim,
        "ct_dim": matmul_dims.ct_dim,
        "kt_dim": matmul_dims.kt_dim,
        "implied_math_format": implied_math_format,
        "dest_sync_mode": dest_sync_mode,
        "unpack_transpose_faces": transpose,
    }

    # Use the new helper function for writing stimuli
    res_address = write_stimuli_to_l1(
        test_config,
        tilized_A.flatten(),
        tilized_B.flatten(),
        format.input_format,
        format.input_format,
        tile_cnt_A,
        tile_cnt_B,
    )

    run_test(test_config, BootMode.TRISC)

    res_from_L1 = collect_results(
        format, tile_count=matmul_dims.output_tile_cnt, address=res_address
    )
    assert len(res_from_L1) == len(golden_tensor)

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(golden_tensor, res_tensor, format.output_format)
