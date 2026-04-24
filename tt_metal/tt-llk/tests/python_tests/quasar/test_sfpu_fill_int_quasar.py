# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# AI-generated — run_id: 2026-04-23_fill_quasar_e9608a59

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    ImpliedMathFormat,
    MathOperation,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
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

# Must match FILL_INT_VALUE in sfpu_fill_int_quasar_test.cpp
FILL_INT_VALUE = 5


# Int32 is the only format exercised here. _calculate_fill_int_ also supports UInt16,
# but one mode is sufficient to cover the explicit-store path; UInt16 would need a
# separate dispatcher since the SFPMEM_MODE is a compile-time template parameter.
SFPU_FILL_INT_FORMATS = input_output_formats([DataFormat.Int32])


@pytest.mark.quasar
@parametrize(
    input_dimensions=[[32, 32], [64, 64]],
    formats=SFPU_FILL_INT_FORMATS,
)
def test_sfpu_fill_int_quasar(input_dimensions, formats):
    """
    Test _calculate_fill_int_<DataFormat::Int32> on Quasar.

    The kernel writes FILL_INT_VALUE to every Int32 element of Dest via
    SFPLOADI + SFPSTORE(sfpmem::INT32); the golden is a tensor of the same
    shape filled with FILL_INT_VALUE. Stimuli content is irrelevant — fill
    ignores inputs — but Int32 stimuli are still generated so the unpack
    path sees a valid Int32 buffer.
    """
    # Int32 inputs are 32-bit, so dest_acc must be Yes (bit-width match rule).
    dest_acc = DestAccumulation.Yes
    implied_math_format = ImpliedMathFormat.No

    torch.manual_seed(42)

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        sfpu=True,
    )

    num_faces = 4

    num_elements = src_A.numel()
    golden_tensor = torch.full(
        (num_elements,), FILL_INT_VALUE, dtype=format_dict[formats.output_format]
    )

    configuration = TestConfig(
        "sources/quasar/sfpu_fill_int_quasar_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=MathOperation.Fill),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(UnpackerEngine.UnpDest),
            DEST_SYNC(),
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
        unpack_to_dest=True,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
