# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Minimal dummy test to verify the SFPU TRISC is compiled and brought out of reset.

Uses sfpu_dummy_test.cpp with empty run_kernels for all 4 TRISCs.
"""

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    NUM_FACES,
    TILE_COUNT,
    generate_input_dim,
)


@pytest.mark.quasar
def test_sfpu_dummy_quasar():
    """Verifies SFPU TRISC is compiled and brought out of reset."""
    formats = InputOutputFormat(
        input_format=DataFormat.Float16_b,
        output_format=DataFormat.Float16_b,
    )
    input_dimensions = [32, 32]

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        sfpu=False,
    )

    configuration = TestConfig(
        "sources/quasar/sfpu_dummy_test.cpp",
        formats,
        templates=[generate_input_dim(input_dimensions, input_dimensions)],
        runtimes=[TILE_COUNT(tile_cnt_A), NUM_FACES(4)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_A,
        ),
    )

    configuration.run()  # Completes if all 4 TRISCs boot and signal
