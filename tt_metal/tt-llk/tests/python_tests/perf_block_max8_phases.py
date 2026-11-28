# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Explicit phase microbenchmark for SFPU block_max8 and its FPU variant."""

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
def test_perf_block_max8_phases(perf_report):
    """Report each custom phase in cycles per call after dividing by LOOP_FACTOR."""
    formats = input_output_formats([DataFormat.Float16_b])[0]
    src = torch.zeros(ELEMENTS_PER_TILE, dtype=torch.bfloat16)
    PerfConfig(
        "sources/block_max8_phases_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
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
