# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_blackhole
from helpers.format_config import DataFormat
from helpers.llk_params import PerfRunType
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    NUM_FACES,
    TILE_COUNT,
    generate_input_dim,
)
from test_fast_tilize import FAST_TILIZE_SWEEP


@skip_for_blackhole
@pytest.mark.perf
@parametrize(
    **FAST_TILIZE_SWEEP,
)
def test_fast_tilize_perf(
    perf_report,
    formats,
    dest_acc,
    dimensions,
):
    input_height, input_width = dimensions
    tile_count = input_height * input_width
    input_dimensions = (input_height * 32, input_width * 32)

    if formats.input_format == DataFormat.Bfp8_b:
        pytest.skip("Bfp8_b input format is not supported for fast tilize")

    configuration = PerfConfig(
        "sources/fast_tilize_test.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1],
        templates=[],
        runtimes=[
            generate_input_dim(input_dimensions, input_dimensions),
            TILE_COUNT(tile_count),
            LOOP_FACTOR(1024),
            NUM_FACES(4),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    configuration.run(perf_report, run_count=2)
