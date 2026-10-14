# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Scratch (not for merge): the fast tilize perf kernel over production row widths, one tile row, loop factor 32."""

from dataclasses import dataclass

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, PerfRunType
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import LOOP_FACTOR, NUM_FACES, TILE_COUNT, TemplateParameter, generate_input_dim

try:
    from helpers.test_variant_parameters import FAST_TILIZE_W_STEP
except ImportError:

    @dataclass
    class FAST_TILIZE_W_STEP(TemplateParameter):
        ct_dim: int = 0

        def convert_to_cpp(self) -> str:
            return "constexpr bool FAST_TILIZE_W_STEP = false;"


BF, B8, B4, F32 = DataFormat.Float16_b, DataFormat.Bfp8_b, DataFormat.Bfp4_b, DataFormat.Float32
W = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 18, 24, 25, 32, 64]
CASES = (
    [(InputOutputFormat(BF, BF), w, False) for w in W]
    + [(InputOutputFormat(BF, B8), w, False) for w in W]
    + [(InputOutputFormat(BF, B4), w, False) for w in W]
    + [(InputOutputFormat(F32, BF), w, False) for w in (4, 8, 16)]
    + [(InputOutputFormat(BF, BF), w, True) for w in (4, 8, 16)]
)


@pytest.mark.perf
@skip_for_wormhole
@skip_for_quasar
@parametrize(case=list(range(len(CASES))))
def test_perf_pl_tilize(perf_report, case):
    if isinstance(case, tuple):
        case = case[0]
    formats, ct_dim, acc = CASES[case]
    dimensions = (32, ct_dim * 32)
    configuration = PerfConfig(
        "sources/fast_tilize_bh_test.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ],
        templates=[FAST_TILIZE_W_STEP(ct_dim)],
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
