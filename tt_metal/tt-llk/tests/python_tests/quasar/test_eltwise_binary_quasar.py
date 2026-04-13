# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    EltwiseBinaryGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathFidelity,
    MathOperation,
    format_dict,
)
from helpers.param_config import (
    generate_unary_input_dimensions,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import BootMode, TestConfig
from helpers.test_variant_parameters import (
    ACC_TO_DEST,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    INPUT_TILE_CNT,
    MATH_FIDELITY,
    MATH_OP,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    OUTPUT_TILE_CNT,
    TEST_FACE_DIMS,
)
from helpers.utils import passed_test

ELTWISE_DIMENSIONS = [
    (dest_sync, dims, DestAccumulation.No)
    for dest_sync in (DestSync.Half, DestSync.Full)
    for dims in generate_unary_input_dimensions(DestAccumulation.No, dest_sync)
]
from helpers.tile_shape import construct_tile_shape


# For acc_to_dest setting, accumulate two result tiles into dest. Can be extended.
def get_num_tiles_per_accumulation(acc_to_dest: bool) -> int:
    return 2 if acc_to_dest else 1


@pytest.mark.quasar
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.MxFp8R,
            DataFormat.MxFp8P,
            # DataFormat.Float16_b,
            DataFormat.Float16,
        ],
    ),
    mathop=[
        MathOperation.Elwadd,
        MathOperation.Elwsub,
        MathOperation.Elwmul,
    ],
    math_fidelity=[
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
    implied_math_format=[
        ImpliedMathFormat.No,
        ImpliedMathFormat.Yes,
    ],
    dest_sync_dims_dest_acc=ELTWISE_DIMENSIONS,
    acc_to_dest=[False, True],
    num_faces=[4],
)
def test_eltwise_binary(
    formats,
    mathop,
    math_fidelity,
    implied_math_format,
    dest_sync_dims_dest_acc,
    acc_to_dest,
    num_faces,
    boot_mode=BootMode.DEFAULT,
):
    dest_sync_mode, input_dimensions, dest_acc = dest_sync_dims_dest_acc
    tile_shape = construct_tile_shape()

    # Math fidelity only affects multiplication operations
    if (
        mathop in [MathOperation.Elwadd, MathOperation.Elwsub]
        and math_fidelity != MathFidelity.LoFi
    ):
        pytest.skip("Math fidelity only affects multiplication operations")

    # MX formats REQUIRE implied_math_format=Yes on Quasar (bypass format inference pipeline)
    if (
        formats.input_format.is_mx_format()
        and implied_math_format == ImpliedMathFormat.No
    ):
        pytest.skip("MX formats require implied_math_format=Yes on Quasar")

    num_tiles_per_accumulation = get_num_tiles_per_accumulation(acc_to_dest)
    total_tiles = (
        input_dimensions[0] * input_dimensions[1]
    ) // tile_shape.total_tile_size()

    if (
        acc_to_dest and total_tiles < num_tiles_per_accumulation
    ) or total_tiles % num_tiles_per_accumulation != 0:
        pytest.skip("Not enough tiles for dest accumulation")

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        output_format=formats.output_format,
    )

    tile_cnt_res = src_A.numel() // (
        tile_shape.total_tile_size() * num_tiles_per_accumulation
    )

    generate_golden = get_golden_generator(EltwiseBinaryGolden)
    golden_tensor = generate_golden(
        mathop,
        src_A,
        src_B,
        formats.output_format,
        math_fidelity,
        input_format=formats.input_format,
        acc_to_dest=acc_to_dest,
        tile_shape=tile_shape,
        num_tiles_per_accumulation=num_tiles_per_accumulation,
    )

    configuration = TestConfig(
        "sources/quasar/eltwise_binary_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            MATH_OP(mathop=mathop),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DEST_SYNC(dest_sync_mode),
            ACC_TO_DEST(acc_to_dest),
        ],
        runtimes=[
            INPUT_TILE_CNT(tile_cnt_A),
            OUTPUT_TILE_CNT(tile_cnt_res),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            NUM_TILES_IN_BLOCK(num_tiles_per_accumulation),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_res,
            num_faces=num_faces,
        ),
        # Determine unpack_to_dest based on format and accumulation mode
        unpack_to_dest=(
            formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
        ),
        dest_acc=dest_acc,
        boot_mode=boot_mode,
        # MX formats require disable_format_inference to match C++ IMPLIED_MATH_FORMAT setting
        # This ensures Python-side format inference uses Float16_b for MX internal math
        disable_format_inference=(implied_math_format == ImpliedMathFormat.Yes),
    )

    res_from_L1 = configuration.run().result

    # Verify results match golden
    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
