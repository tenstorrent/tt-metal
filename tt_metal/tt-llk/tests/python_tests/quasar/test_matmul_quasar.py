# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from helpers.data_format_inference import data_formats
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    TILE_DIM,
    MatmulGolden,
    TransposeGolden,
    get_golden_generator,
    quantize_mx_tensor_chunked,
)
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathFidelity,
    Transpose,
    format_dict,
)
from helpers.matmul_sweep import generate_tile_dims
from helpers.param_config import (
    DEST_SYNC_TILE_LIMITS,
    input_output_formats,
    parametrize,
)
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
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

kt_dims = [1, 2, 4]
matmul_dimensions_dest_sync = [
    (
        [mt_dim * TILE_DIM, kt_dim * TILE_DIM],
        [kt_dim * TILE_DIM, nt_dim * TILE_DIM],
        dest_acc,
        dest_sync,
    )
    for dest_sync in (DestSync.Half, DestSync.Full)
    for dest_acc in (DestAccumulation.Yes, DestAccumulation.No)
    for max_tiles in (
        DEST_SYNC_TILE_LIMITS[dest_sync]
        // (2 if dest_acc == DestAccumulation.Yes else 1),
    )
    for mt_dim in range(1, max_tiles + 1)
    for nt_dim in range(1, max_tiles // mt_dim + 1)
    for kt_dim in kt_dims
]

# Generate format-aware combinations
MATMUL_FORMAT = input_output_formats(
    [
        DataFormat.Float16,
        DataFormat.Float16_b,
        DataFormat.MxFp4,
    ],
)


@pytest.mark.quasar
@parametrize(
    math_fidelity=[
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
    dimensions_dest_acc_dest_sync=matmul_dimensions_dest_sync,
    format=MATMUL_FORMAT,
    implied_math_format=lambda format: (
        [ImpliedMathFormat.Yes]
        if format.input_format.is_mx_format()
        else [ImpliedMathFormat.No, ImpliedMathFormat.Yes]
    ),
    transpose=[Transpose.No],
)
# Note: this test is used to test boot modes, that is why it has them piped as default arguments to the test itself
def test_matmul(
    math_fidelity,
    dimensions_dest_acc_dest_sync,
    format,
    implied_math_format,
    transpose,
):

    input_A_dimensions, input_B_dimensions, dest_acc, dest_sync_mode = (
        dimensions_dest_acc_dest_sync
    )

    if format.output_format.is_mx_format() and dest_acc == DestAccumulation.No:
        pytest.skip(
            "Mx output format without destination accumulation produces flaky results"
        )

    torch_format = format_dict[format.output_format]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=format.input_format,
        input_dimensions_A=input_A_dimensions,
        stimuli_format_B=format.input_format,
        input_dimensions_B=input_B_dimensions,
        sfpu=False,
        output_format=format.output_format,
    )

    tilized_A = tilize_block(
        src_A, dimensions=input_A_dimensions, stimuli_format=format.input_format
    )
    tilized_B = tilize_block(
        src_B, dimensions=input_B_dimensions, stimuli_format=format.input_format
    )

    src_A_golden = src_A
    src_B_golden = src_B
    if format.input_format.is_mx_format():
        tilized_A_golden = quantize_mx_tensor_chunked(
            tilized_A.flatten().to(torch.bfloat16), format.input_format
        ).reshape(tilized_A.shape)
        tilized_B_golden = quantize_mx_tensor_chunked(
            tilized_B.flatten().to(torch.bfloat16), format.input_format
        ).reshape(tilized_B.shape)
        src_A_golden = untilize_block(
            tilized_A_golden,
            stimuli_format=format.input_format,
            dimensions=input_A_dimensions,
        )
        src_B_golden = untilize_block(
            tilized_B_golden,
            stimuli_format=format.input_format,
            dimensions=input_B_dimensions,
        )

    if transpose == Transpose.Yes:
        t_matrix = get_golden_generator(TransposeGolden)

        src_B_golden = t_matrix.transpose_faces_multi_tile(
            src_B_golden,
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

    formats_config = data_formats(
        input_format=format.input_format,
        input_format_B=format.input_format_B,
        output_format=format.output_format,
        is_fp32_dest_acc_en=dest_acc,
        num_iterations=1,
        unpacking_to_dest=False,
        disable_format_inference=format.input_format.is_mx_format(),
    )[0]
    pack_src_format = formats_config.pack_src

    generate_golden = get_golden_generator(MatmulGolden)
    golden_tensor = generate_golden(
        src_A_golden,
        src_B_golden,
        format.output_format,
        math_fidelity,
        input_A_dimensions=input_A_dimensions,
        input_B_dimensions=input_B_dimensions,
        tilize=True,  # Golden cannot model FPU strided for tilized data computation, so we tilize output after computation
        input_A_format=format.input_format,
        input_B_format=format.input_format,
        math_format=pack_src_format,  # For accumulation of results in matmul we require to calculate in pack_src_format.
        dest_acc=dest_acc,
    )

    num_faces = 4

    configuration = TestConfig(
        "sources/quasar/matmul_quasar_test.cpp",
        format,
        templates=[
            MATH_FIDELITY(math_fidelity),
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
        disable_format_inference=format.input_format.is_mx_format(),
    )

    res_from_L1 = configuration.run().result
    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    if format.output_format.is_mx_format():
        golden_tensor = quantize_mx_tensor_chunked(
            golden_tensor.to(format_dict[pack_src_format]), format.output_format
        ).to(torch_format)

    test_passed = passed_test(golden_tensor, res_tensor, format.output_format)

    assert test_passed, "Assert against golden failed"
