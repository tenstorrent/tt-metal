# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SFPU square on TRISC3 (isolated). MATH does datacopy when !unpack_to_dest.

Same structure and parameter coverage as test_sfpu_square_quasar but SFPU runs on TRISC3.
"""

import pytest
import torch
from helpers.golden_generators import UnarySFPUGolden, get_golden_generator
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathOperation,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import (
    is_invalid_quasar_sfpu_format_combination,
    parametrize,
    runtime,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    MATH_OP,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
)
from helpers.tile_constants import MAX_NUM_FACES
from helpers.utils import passed_test

# Reuse the square input-prep and format list from the consolidated unary-SFPU
# test (the standalone test_sfpu_square_quasar.py was folded into it).
from test_eltwise_unary_sfpu_quasar import SFPU_UNARY_FORMATS as SFPU_SQUARE_FORMATS
from test_eltwise_unary_sfpu_quasar import (
    prepare_square_inputs,
)


def generate_sfpu_square_combinations(formats_list):
    """
    Square-only sweep for the TRISC3 variant: (fmt, dest_acc, dest_sync,
    implied_math_format, input_dimensions) tuples. Uniform dims [32,32]/[64,64]
    x {Half,Full} sync (the redundant [32,64] dim is dropped, matching the
    consolidated unary test).
    """
    combinations = []
    dest_sync_modes = (DestSync.Half, DestSync.Full)
    implied_math_modes = (ImpliedMathFormat.No, ImpliedMathFormat.Yes)
    input_dimension_options = ([32, 32], [64, 64])
    for fmt in formats_list:
        in_fmt = fmt.input_format
        dest_acc_modes = (
            (DestAccumulation.Yes,)
            if in_fmt.is_32_bit()
            else (DestAccumulation.No, DestAccumulation.Yes)
        )
        for dest_acc in dest_acc_modes:
            if is_invalid_quasar_sfpu_format_combination(fmt, dest_acc):
                continue
            for dest_sync in dest_sync_modes:
                for implied_math_format in implied_math_modes:
                    for input_dimensions in input_dimension_options:
                        combinations.append(
                            (
                                fmt,
                                dest_acc,
                                dest_sync,
                                implied_math_format,
                                runtime(input_dimensions),
                            )
                        )
    return combinations


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_sync_implied_math_dims=generate_sfpu_square_combinations(
        SFPU_SQUARE_FORMATS
    ),
)
def test_sfpu_square_trisc3_quasar(
    formats_dest_acc_sync_implied_math_dims,
):
    """
    Test square operation on Quasar with SFPU on TRISC3.

    Same parameter coverage as test_sfpu_square_quasar.
    """
    formats, dest_acc, dest_sync_mode, implied_math_format, input_dimensions = (
        formats_dest_acc_sync_implied_math_dims[0]
    )

    torch.manual_seed(42)

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    src_A = prepare_square_inputs(
        src_A, src_B, formats.input_format, formats.output_format
    )

    num_faces = MAX_NUM_FACES

    generate_golden = get_golden_generator(UnarySFPUGolden)
    golden_tensor = generate_golden(
        MathOperation.Square,
        src_A,
        formats.output_format,
        dest_acc,
        formats.input_format,
        input_dimensions,
    )

    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )
    configuration = TestConfig(
        "sources/quasar/sfpu_square_trisc3_quasar_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=MathOperation.Square),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA
            ),
            DEST_SYNC(dest_sync_mode),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_A,
            num_faces=num_faces,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(golden_tensor)
    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)
    assert passed_test(golden_tensor, res_tensor, formats.output_format)
