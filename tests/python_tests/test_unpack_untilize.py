# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import UntilizeGolden, get_golden_generator
from helpers.llk_params import format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    INPUT_DIMENSIONS,
    TILE_COUNT,
)
from helpers.utils import passed_test


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,
            DataFormat.Bfp8_b,
        ]
    ),
)
def test_unpack_untilize(formats, workers_tensix_coordinates):
    formats = formats[0]
    if formats.input_format == DataFormat.Bfp8_b:
        pytest.skip(
            "BFP8 format is not supported for unpack_untilize operation for multiple tiles"
        )

    input_dimensions = [32, 128]
    if formats.input_format == DataFormat.Float32:
        input_dimensions = [
            32,
            32,
        ]  # Multiple tile unpack untilize not supported for Float32, @ldjurovic can take a look to extend this

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    generate_golden = get_golden_generator(UntilizeGolden)
    golden_tensor = generate_golden(
        src_A, formats.output_format, dimensions=input_dimensions
    )

    configuration = TestConfig(
        "sources/unpack_untilize_test.cpp",
        formats,
        templates=[
            INPUT_DIMENSIONS(input_dimensions, input_dimensions),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
    )

    res_from_L1 = configuration.run(workers_tensix_coordinates)

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golder tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
