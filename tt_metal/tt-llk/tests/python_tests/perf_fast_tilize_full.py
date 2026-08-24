# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Full pipeline performance test for BH fast-tilize (unpack + math + pack).

Test matrix mirrors perf_unpack_tilize.py so that regular-tilize and
fast-tilize numbers are directly comparable in the nightly perf dashboard.
"""

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import PerfRunType
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    NUM_FACES,
    TILE_COUNT,
    generate_input_dim,
)

# ---------------------------------------------------------------------------
# Same-format: mirrors perf_unpack_tilize.py float matrix (1×1 … 8×8)
# ---------------------------------------------------------------------------
_SAME_FORMATS = input_output_formats(
    [DataFormat.Float16_b, DataFormat.Float32], same=True
)
_CROSS_FORMATS = [
    InputOutputFormat(DataFormat.Float32, DataFormat.Float16_b),
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Bfp8_b),
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Bfp4_b),
    InputOutputFormat(DataFormat.Float32, DataFormat.Bfp8_b),
    InputOutputFormat(DataFormat.Float32, DataFormat.Bfp4_b),
]
_FAST_TILIZE_FULL_CASES = [(fmt, 1, ct) for fmt in _SAME_FORMATS for ct in [2, 8]] + [
    (fmt, 1, ct) for fmt in _CROSS_FORMATS for ct in [2, 8]
]


@pytest.mark.perf
@skip_for_wormhole
@skip_for_quasar
@parametrize(formats_rt_ct=_FAST_TILIZE_FULL_CASES)
def test_perf_fast_tilize_full(perf_report, formats_rt_ct):
    formats, rt_dim, ct_dim = formats_rt_ct
    _run_fast_tilize_perf(perf_report, formats, rt_dim, ct_dim)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------
def _run_fast_tilize_perf(perf_report, formats, rt_dim, ct_dim):
    tile_count = rt_dim * ct_dim
    dimensions = (rt_dim * 32, ct_dim * 32)

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
        templates=[],
        runtimes=[
            generate_input_dim(dimensions, dimensions),
            TILE_COUNT(tile_count),
            LOOP_FACTOR(32),
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
        compile_time_formats=True,
    )

    configuration.run(perf_report)
