# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Per-phase math-thread cycles of the SFPU block_max8 and full/partial cycles of its FPU (GMPOOL) variant."""

import pytest
import torch
from helpers.llk_params import PerfRunType
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from test_block_max8 import DST_INDICES, VALID_COUNTS
from tt_llk_harness import (
    ELEMENTS_PER_TILE,
    DataFormat,
    StimuliConfig,
    blackhole_only,
    input_output_formats,
    params,
)

LOOP_FACTOR = 64


def _run_perf_block_max8(perf_report, dst_index, valid, pattern, fpu_calls=1):
    formats = input_output_formats([DataFormat.Float16_b])[0]
    src = torch.zeros(ELEMENTS_PER_TILE, dtype=torch.bfloat16)
    runtimes = [
        params.TILE_COUNT(1),
        params.LOOP_FACTOR(LOOP_FACTOR),
        params.DEST_INDEX(dst_index),
        params.BLOCK_MAX8_PARAMS(valid, int(pattern == "negative_zero")),
        # Every row records the FPU axes. The SFPU sweep keeps the defaults
        # (one call, no denormal inject) so the CSV has one column set.
        params.BLOCK_MAX8_FPU_PARAMS(fpu_calls, int(pattern == "denormal")),
    ]
    PerfConfig(
        "sources/block_max8_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=runtimes,
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


# pattern is recorded only as inject_negative_zero / inject_denormal, and calls=1
# is the same kernel config the old SFPU sweep used. One row per recorded key.
_PERF_PATTERNS = ["random", "negative_zero", "denormal"]


@pytest.mark.perf
@blackhole_only
@parametrize(
    dst_index=DST_INDICES,
    valid=VALID_COUNTS,
    pattern=_PERF_PATTERNS,
    fpu_calls=[1, 3],
)
def test_perf_block_max8(perf_report, dst_index, valid, pattern, fpu_calls):
    """Report cycles per call: divide each zone by LOOP_FACTOR.

    calls=1 is the SFPU path (DISPATCH, FULL_TILE, PARTIAL_TILE, MASK, REDUCE,
    COMPACT). calls=3 is the FPU path (FPU_FULL_TILE, FPU_PARTIAL_TILE).
    """
    _run_perf_block_max8(perf_report, dst_index, valid, pattern, fpu_calls=fpu_calls)
