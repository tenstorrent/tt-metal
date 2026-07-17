# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# RV_PACR untilize DEMO test.
#
# Proof-of-life for the RISC-V-descriptor pack path (RV_PACR). It reuses the
# UntilizeGolden of the normal pack-untilize test: for a full 32x32 tile the
# RV_PACR untilize must produce byte-identical output to the MOP/PACR_UNTILIZE
# path. Scope is intentionally minimal: a single 32x32 tile, DestSync.Half,
# DestAcc.No, simple 16-bit formats.

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import UntilizeGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
    generate_input_dim,
)
from helpers.utils import passed_test

# Single 32x32 tile.
SINGLE_TILE_DIMS = [32, 32]

RV_PACK_UNTILIZE_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float16,
    ],
)


@pytest.mark.quasar
@parametrize(formats=RV_PACK_UNTILIZE_FORMATS)
def test_pack_untilize_rv_quasar(formats):
    # @parametrize wraps each value in a 1-tuple (matches the sibling test's `[0]`).
    (formats,) = formats
    dest_acc = DestAccumulation.No
    dest_sync_mode = DestSync.Half
    input_dimensions = SINGLE_TILE_DIMS

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    generate_golden = get_golden_generator(UntilizeGolden)
    golden_tensor = generate_golden(
        src_A,
        formats.output_format,
        input_dimensions,
        input_format=formats.input_format,
    )

    num_faces = 4
    configuration = TestConfig(
        "sources/quasar/pack_untilize_rv_quasar_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.Yes),
            DEST_SYNC(dest_sync_mode),
            UNPACKER_ENGINE_SEL(),
            TEST_FACE_DIMS(),
            NUM_FACES(num_faces),
            TILE_COUNT(tile_cnt_A),
        ],
        runtimes=[],
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
        unpack_to_dest=False,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format, print_errors=False
    ), "Assert against golden failed"
