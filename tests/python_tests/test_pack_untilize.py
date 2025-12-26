# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import UntilizeGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, DestSync, format_dict
from helpers.param_config import (
    generate_unary_input_dimensions,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    INPUT_DIMENSIONS,
    TILE_COUNT,
)
from helpers.utils import passed_test


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,  # Test Float32 with both 32bit mode dest (full precision) and 16bit mode dest (precision loss)
            DataFormat.Int32,
            DataFormat.Bfp8_b,
        ]  # Pack Untilize doesn't work for block float formats (Bfp8_b); we only include as input format in our test
    ),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    input_dimensions=[[32, 128], [128, 32], [64, 64], [32, 64], [64, 32]],
    dest_sync=[DestSync.Half, DestSync.Full],
)
def test_pack_untilize(
    formats, dest_acc, input_dimensions, dest_sync, workers_tensix_coordinates
):
    if formats.output_format == DataFormat.Bfp8_b:
        pytest.skip("Pack Untilize does not support Bfp8_b format")

    if (formats.input_format == DataFormat.Int32) ^ (
        formats.output_format == DataFormat.Int32
    ):
        pytest.skip("Pack Untilize does not support mixing Int32 with other formats")

    if (
        formats.input_format == DataFormat.Int32
        and formats.output_format == DataFormat.Int32
        and dest_acc == DestAccumulation.No
    ):
        pytest.skip("Dest must be in 32bit mode when input and output are Int32")

    if input_dimensions not in generate_unary_input_dimensions(
        dest_acc, DestSync.Full if dest_sync == DestSync.Full else DestSync.Half
    ):
        pytest.skip(
            "Input dimensions not supported for the given dest_acc and dest_sync configuration"
        )

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        sfpu=False,
    )

    generate_golden = get_golden_generator(UntilizeGolden)
    golden_tensor = generate_golden(src_A, formats.output_format, input_dimensions)

    configuration = TestConfig(
        "sources/pack_untilize_test.cpp",
        formats,
        templates=[
            INPUT_DIMENSIONS(input_dimensions, input_dimensions),
            DEST_SYNC(dest_sync),
        ],
        runtimes=[TILE_COUNT(tile_cnt_A)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            sfpu=False,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=formats.input_format.is_32_bit()
        and dest_acc == DestAccumulation.Yes,
    )

    res_from_L1 = configuration.run(workers_tensix_coordinates)

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golder tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
