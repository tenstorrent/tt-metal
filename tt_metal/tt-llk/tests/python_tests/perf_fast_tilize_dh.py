# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shadow (not checked in): the fast tilize perf kernel at the production formats and tiles per row of the Blackhole
fast tilize callers on main (ops/blackhole/tilize/scripts/caller_survey.md section 4.1), one tile row, loop factor 32."""

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, PerfRunType
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    NUM_FACES,
    TILE_COUNT,
    generate_input_dim,
)

BF, B8, B4, F32 = DataFormat.Float16_b, DataFormat.Bfp8_b, DataFormat.Bfp4_b, DataFormat.Float32
# widths 90 and up (ttnn.tilize rows of 2880 to 7168, post_combine_reduce, conv3d K_t 108) hang this perf kernel on the
# base as well (hold fx5, 19:2x UTC); width 64 runs last so a hang there cannot void the other rows
W = list(range(2, 34)) + [40, 48, 56, 64]
CASES = [(InputOutputFormat(BF, BF), w) for w in W] + [(InputOutputFormat(BF, B8), w) for w in W]
CASES_ACC = []


@pytest.mark.perf
@skip_for_wormhole
@skip_for_quasar
@parametrize(case=list(range(len(CASES) + len(CASES_ACC))))
def test_perf_fast_tilize_dh(perf_report, case):
    if isinstance(case, tuple):
        case = case[0]
    acc = case >= len(CASES)
    formats, ct_dim = CASES_ACC[case - len(CASES)] if acc else CASES[case]
    dimensions = (32, ct_dim * 32)
    configuration = PerfConfig(
        "sources/fast_tilize_bh_test.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1],
        templates=[],
        runtimes=[
            generate_input_dim(dimensions, dimensions),
            TILE_COUNT(ct_dim),
            LOOP_FACTOR(32),
            NUM_FACES(4),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=ct_dim,
            tile_count_B=1,
            tile_count_res=ct_dim,
        ),
        compile_time_formats=True,
        dest_acc=DestAccumulation.Yes if acc else DestAccumulation.No,
    )
    configuration.run(perf_report)
