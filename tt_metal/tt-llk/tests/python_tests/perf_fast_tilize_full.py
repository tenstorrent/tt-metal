# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Full pipeline performance test for BH fast-tilize (unpack + math + pack).

Test matrix mirrors perf_unpack_tilize.py so that regular-tilize and
fast-tilize numbers are directly comparable in the nightly perf dashboard.
"""

import pytest
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
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


def _skip_non_bh():
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("BH only")


# ---------------------------------------------------------------------------
# Same-format: mirrors perf_unpack_tilize.py float matrix (1×1 … 8×8)
# ---------------------------------------------------------------------------
@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [DataFormat.Float16_b, DataFormat.Float32, DataFormat.Bfp8_b]
    ),
    rt_dim=[1, 2, 3, 4, 5, 6, 7, 8],
    ct_dim=[1, 2, 3, 4, 5, 6, 7, 8],
)
def test_perf_fast_tilize(
    perf_report, formats, rt_dim, ct_dim, workers_tensix_coordinates
):
    _skip_non_bh()

    # BFP / Float32 input not supported for tilize
    if formats.input_format in (DataFormat.Bfp8_b,):
        pytest.skip("Bfp8_b input not supported for fast tilize")

    # Width 1 uses standard tilize fallback — not representative of fast path
    if ct_dim < 2:
        pytest.skip("ct_dim < 2 uses standard tilize fallback")

    _run_fast_tilize_perf(
        perf_report, formats, rt_dim, ct_dim, workers_tensix_coordinates
    )


# ---------------------------------------------------------------------------
# Cross-format BFP output: Float16_b / Float32 → Bfp8_b / Bfp4_b
# ---------------------------------------------------------------------------
@pytest.mark.perf
@parametrize(
    formats=[
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Bfp8_b),
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Bfp4_b),
        InputOutputFormat(DataFormat.Float32, DataFormat.Bfp8_b),
        InputOutputFormat(DataFormat.Float32, DataFormat.Bfp4_b),
    ],
    rt_dim=[1, 2, 4, 8],
    ct_dim=[2, 4, 8],
)
def test_perf_fast_tilize_bfp(
    perf_report, formats, rt_dim, ct_dim, workers_tensix_coordinates
):
    _skip_non_bh()
    _run_fast_tilize_perf(
        perf_report, formats, rt_dim, ct_dim, workers_tensix_coordinates
    )


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------
def _run_fast_tilize_perf(
    perf_report, formats, rt_dim, ct_dim, workers_tensix_coordinates
):
    tile_count = rt_dim * ct_dim
    dimensions = (rt_dim * 32, ct_dim * 32)

    configuration = PerfConfig(
        "sources/fast_tilize_bh_test.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
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

    configuration.run(perf_report, location=workers_tensix_coordinates)
