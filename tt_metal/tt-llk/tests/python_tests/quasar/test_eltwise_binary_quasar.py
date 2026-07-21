# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
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
    runtime,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
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
from helpers.tile_shape import construct_tile_shape
from helpers.utils import passed_test


def _eltwise_dest_acc_sync(dest_acc_modes):
    return [
        (dest_sync, dest_acc)
        for dest_sync in (DestSync.Half, DestSync.Full)
        for dest_acc in dest_acc_modes
    ]


# For acc_to_dest setting, accumulate two result tiles into dest. Can be extended.
def get_num_tiles_per_accumulation(acc_to_dest: bool) -> int:
    return 2 if acc_to_dest else 1


_TILE_SHAPE = construct_tile_shape()


def valid_acc_to_dest(input_dimensions) -> list:
    """Pick the acc_to_dest modes worth running for a given input size.

    acc_to_dest=True accumulates `get_num_tiles_per_accumulation(True)` result tiles into
    dest, so it only makes sense when the tile count is a non-zero multiple of that.
    """
    total_tiles = (
        input_dimensions[0] * input_dimensions[1]
    ) // _TILE_SHAPE.total_tile_size()

    per_acc = get_num_tiles_per_accumulation(True)
    if total_tiles >= per_acc and total_tiles % per_acc == 0:
        return [False, True]
    return [False]


ELTWISE_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float16,
    ],
) + [InputOutputFormat(DataFormat.Int8, DataFormat.Int32)]


@pytest.mark.quasar
@parametrize(
    formats=ELTWISE_FORMATS,
    mathop=[
        MathOperation.Elwadd,
        MathOperation.Elwsub,
        MathOperation.Elwmul,
    ],
    # Math fidelity only affects multiplication; for add/sub and Int8 only LoFi is meaningful.
    math_fidelity=lambda mathop, formats: (
        [MathFidelity.LoFi]
        if mathop in [MathOperation.Elwadd, MathOperation.Elwsub]
        or formats.input_format == DataFormat.Int8
        else [
            MathFidelity.LoFi,
            MathFidelity.HiFi2,
            MathFidelity.HiFi3,
            MathFidelity.HiFi4,
        ]
    ),
    implied_math_format=[ImpliedMathFormat.No, ImpliedMathFormat.Yes],
    dest_sync_dest_acc=lambda formats: (
        _eltwise_dest_acc_sync((DestAccumulation.Yes,))
        if formats.input_format == DataFormat.Int8
        else _eltwise_dest_acc_sync((DestAccumulation.No,))
    ),
    input_dimensions=runtime(
        lambda dest_sync_dest_acc: generate_unary_input_dimensions(
            dest_sync_dest_acc[1], dest_sync_dest_acc[0]
        )
    ),
    acc_to_dest=valid_acc_to_dest,
    num_faces=[4],
)
def test_eltwise_binary(
    formats,
    mathop,
    math_fidelity,
    implied_math_format,
    dest_sync_dest_acc,
    input_dimensions,
    acc_to_dest,
    num_faces,
    boot_mode=BootMode.DEFAULT,
):
    dest_sync_mode, dest_acc = dest_sync_dest_acc

    num_tiles_per_accumulation = get_num_tiles_per_accumulation(acc_to_dest)

    if formats.input_format == DataFormat.Int8:
        stimuli_spec = StimuliSpec.uniform(low=-127.0, high=127.0)
    else:
        stimuli_spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=stimuli_spec,
        spec_B=stimuli_spec,
        output_format=formats.output_format,
    )

    tile_cnt_res = src_A.numel() // (
        _TILE_SHAPE.total_tile_size() * num_tiles_per_accumulation
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
        tile_shape=_TILE_SHAPE,
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
    )

    res_from_L1 = configuration.run().result

    # Verify results match golden
    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
    ), "Assert against golden failed"
