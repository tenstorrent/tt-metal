# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.llk_params import PerfRunType
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import PerfConfig
from helpers.perf.relevance import _PACK_FORMATS, PIN_ALL, PerfRelevance
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    INPUT_DIMENSIONS,
    LOOP_FACTOR,
    TILE_COUNT,
    generate_input_dim,
)

UNPACK_TILIZE_RUN_TYPES = (
    PerfRunType.L1_TO_L1,
    PerfRunType.UNPACK_ISOLATE,
    PerfRunType.PACK_ISOLATE,
    PerfRunType.L1_CONGESTION,
)


class UnpackTilizeRelevance(PerfRelevance):
    """``perf_unpack_tilize`` / ``unpack_tilize_perf.cpp``: no math mode.

    All three remaining modes key off the same dimension triple. The two
    comments below are both scars from real failures, so change either set only
    with the kernel source open. The production perf test intentionally does not
    attach this policy.
    """

    run_types = UNPACK_TILIZE_RUN_TYPES
    _DIM_RUNTIMES = frozenset({INPUT_DIMENSIONS, TILE_COUNT, LOOP_FACTOR})
    unpack_runtimes = _DIM_RUNTIMES
    pack_templates = PIN_ALL
    # Unpack asserts FULL_RT_DIM * FULL_CT_DIM == TILE_CNT before PACK returns.
    # SPEED_OF_LIGHT inlines runtimes, so PACK must keep those values.
    pack_runtimes = _DIM_RUNTIMES
    cong_runtimes = _DIM_RUNTIMES
    # Blackhole tilize workaround keys off unpack_A_src, not only pack_src/pack_dst.
    pack_formats = _PACK_FORMATS | frozenset({"unpack_A_src"})


UNPACK_TILIZE_RELEVANCE = UnpackTilizeRelevance()

assert UNPACK_TILIZE_RELEVANCE.run_types == (
    PerfRunType.L1_TO_L1,
    PerfRunType.UNPACK_ISOLATE,
    PerfRunType.PACK_ISOLATE,
    PerfRunType.L1_CONGESTION,
)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,
            DataFormat.Bfp8_b,
            DataFormat.Fp8_e4m3,
        ]
    ),
    rt_dim=[1, 2, 3, 4, 5, 6, 7, 8],
    ct_dim=[1, 2, 3, 4, 5, 6, 7, 8],
)
def test_perf_unpack_tilize_float(
    perf_report,
    formats,
    rt_dim,
    ct_dim,
):
    if (
        formats.input_format == DataFormat.Fp8_e4m3
        or formats.output_format == DataFormat.Fp8_e4m3
    ) and get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip(
            "Unpack Tilize does not support Fp8_e4m3 format on non-BLACKHOLE architectures"
        )

    if formats.input_format == DataFormat.Bfp8_b:
        pytest.skip("Bfp8_b input not supported for unpack_tilize")

    _perf_unpack_tilize(
        perf_report,
        formats,
        rt_dim,
        ct_dim,
    )


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    rt_dim=[1, 2],
    ct_dim=[1, 2],
)
def test_perf_unpack_tilize_int(
    perf_report,
    formats,
    rt_dim,
    ct_dim,
):
    _perf_unpack_tilize(
        perf_report,
        formats,
        rt_dim,
        ct_dim,
    )


def _perf_unpack_tilize(
    perf_report,
    formats,
    rt_dim,
    ct_dim,
):
    tile_count = rt_dim * ct_dim
    dimensions = [rt_dim * 32, ct_dim * 32]

    configuration = PerfConfig(
        "sources/unpack_tilize_perf.cpp",
        formats,
        run_types=list(UNPACK_TILIZE_RUN_TYPES),
        templates=[],
        runtimes=[
            generate_input_dim(dimensions, dimensions),
            TILE_COUNT(tile_count),
            LOOP_FACTOR(256),
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
        unpack_to_dest=formats.input_format == DataFormat.Int32,
        # Keep this measurement full-fidelity regardless of the global
        # LLK_DISABLE_PERF_RELEVANCE setting.
        relevance=None,
    )

    configuration.run(perf_report)
