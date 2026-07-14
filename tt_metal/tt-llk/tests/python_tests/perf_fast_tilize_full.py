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
from helpers.perf import PerfConfig
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
@pytest.mark.perf
@skip_for_wormhole
@skip_for_quasar
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True),
    rt_dim=[1],
    ct_dim=[1, 2, 3, 4, 5, 6, 7, 8],
)
def test_perf_fast_tilize(perf_report, formats, rt_dim, ct_dim):
    # Width 1 uses standard tilize fallback — not representative of fast path
    if ct_dim < 2:
        pytest.skip("ct_dim < 2 uses standard tilize fallback")

    _run_fast_tilize_perf(perf_report, formats, rt_dim, ct_dim)


# ---------------------------------------------------------------------------
# Cross-format output: mirrors test_fast_tilize_full.py format matrix
# ---------------------------------------------------------------------------
@pytest.mark.perf
@skip_for_wormhole
@skip_for_quasar
@parametrize(
    formats=[
        InputOutputFormat(DataFormat.Float32, DataFormat.Float16_b),
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Bfp8_b),
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Bfp4_b),
        InputOutputFormat(DataFormat.Float32, DataFormat.Bfp8_b),
        InputOutputFormat(DataFormat.Float32, DataFormat.Bfp4_b),
    ],
    rt_dim=[1],
    ct_dim=[2, 4, 8],
)
def test_perf_fast_tilize_bfp(perf_report, formats, rt_dim, ct_dim):
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
            LOOP_FACTOR(4),
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
