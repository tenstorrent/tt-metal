# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Per-phase math-thread cycles of the SFPU block_max8 and full/partial cycles of its FPU (GMPOOL) variant."""

import pytest
import torch
from helpers.llk_params import PerfRunType
from helpers.perf.core import PerfConfig
from tt_llk_harness import (
    ELEMENTS_PER_TILE,
    DataFormat,
    StimuliConfig,
    blackhole_only,
    input_output_formats,
    params,
)

LOOP_FACTOR = 64


@pytest.mark.perf
@blackhole_only
def test_perf_block_max8(perf_report):
    """Report cycles per call: divide each zone by LOOP_FACTOR.

    SFPU zones: DISPATCH, FULL_TILE, PARTIAL_TILE, MASK, REDUCE, COMPACT.
    FPU zones: FPU_FULL_TILE, FPU_PARTIAL_TILE.
    """
    formats = input_output_formats([DataFormat.Float16_b])[0]
    src = torch.zeros(ELEMENTS_PER_TILE, dtype=torch.bfloat16)
    PerfConfig(
        "sources/block_max8_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        # Compile-time, so the kernel reads no runtime parameters (required under speed-of-light).
        templates=[params.TILE_COUNT(1), params.LOOP_FACTOR(LOOP_FACTOR)],
        variant_stimuli=StimuliConfig(
            src,
            formats.input_format,
            src,
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
        ),
    ).run(perf_report, run_count=4)
