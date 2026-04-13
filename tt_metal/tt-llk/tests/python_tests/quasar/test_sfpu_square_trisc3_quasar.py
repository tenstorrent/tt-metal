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
    MathOperation,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import parametrize
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
from helpers.utils import passed_test
from test_sfpu_square_quasar import (
    SFPU_SQUARE_FORMATS,
    generate_sfpu_square_combinations,
    prepare_square_inputs,
)


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
    (formats, dest_acc, dest_sync_mode, implied_math_format, input_dimensions) = (
        formats_dest_acc_sync_implied_math_dims[0]
    )

    torch.manual_seed(42)

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        sfpu=True,
    )

    src_A = prepare_square_inputs(
        src_A, src_B, formats.input_format, formats.output_format
    )

    num_faces = 4

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
