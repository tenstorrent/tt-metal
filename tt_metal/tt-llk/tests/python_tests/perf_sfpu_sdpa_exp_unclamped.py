# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
MATH_ISOLATE perf for the upper-unclamped exp SFPU kernel
(tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_sdpa_exp_unclamped.h, driven
through its metal face-walk wrapper `calculate_sdpa_exp_unclamped`).

The kernel is bf16-DEST only and the SDPA callers feed Float16_b, so the one format pair
is the kernel's native cost with no unpack/pack conversion folded in. Both SCALE_EN arms
are measured: the scaled body carries one more SFPU instruction (SFPMULI) per 4x8 slot.

Quote `marker == "TILE_LOOP"`, column `mean(MATH_ISOLATE)`, from
perf_data/latest/perf_sfpu_sdpa_exp_unclamped/perf_sfpu_sdpa_exp_unclamped.post.csv.
"""

import pytest
from conftest import skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, PerfRunType
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    SFPU_SCALE_EN,
    SFPU_UNARY_SCALAR,
    TILE_COUNT,
)

# 0.5 as a bfloat16 bit pattern: a non-identity scale so the SFPMULI cannot be folded away.
BF16_HALF = 0x3F00

# 128x64 bf16 input: 8 tiles, one full SyncHalf DEST bank per block.
TILE_CNT = 8


@skip_for_wormhole
@pytest.mark.perf
@parametrize(
    formats=[InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)],
    scale_en=[False, True],
    loop_factor=[16],
)
def test_perf_sfpu_sdpa_exp_unclamped(perf_report, formats, scale_en, loop_factor):
    configuration = PerfConfig(
        "sources/sfpu_sdpa_exp_unclamped_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            SFPU_SCALE_EN(scale_en=scale_en),
            SFPU_UNARY_SCALAR(value_bits=BF16_HALF),
        ],
        runtimes=[
            TILE_COUNT(TILE_CNT),
            LOOP_FACTOR(loop_factor),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=TILE_CNT,
            tile_count_B=TILE_CNT,
            tile_count_res=TILE_CNT,
        ),
        # The math kernel copies srcA to DEST itself (A2D), as the SDPA callers do.
        unpack_to_dest=False,
        # Pinned: the kernel static_asserts !is_fp32_dest_acc_en.
        dest_acc=DestAccumulation.No,
    )

    configuration.run(perf_report)
