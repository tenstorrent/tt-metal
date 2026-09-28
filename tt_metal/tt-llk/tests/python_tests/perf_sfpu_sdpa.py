# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
MATH_ISOLATE perf for the SDPA column-vector SFPU bodies in metal's experimental llk_sfpu
(ckernel_sfpu_sdpa.h and ckernel_sfpu_sdpa_fw.h), driven by sources/sfpu_sdpa_perf.cpp.

The bodies have no LLK API of their own; the functional coverage is test_sfpu_sdpa.py and
test_sfpu_sdpa_fw.py, and this module times the same dispatch (VectorMode::C, one call per
tile after an A2D datacopy). Both dest modes are measured: the bf16-DEST arm runs the sfpi
exp_21f, the fp32-DEST arm the Juffa fp32 exp. ExpPoly (the TTI polynomial the
SDPA_EXP_APPROX_MODE=false path runs) is an unaffected control for exp_21f changes.

Quote `marker == "TILE_LOOP"`, column `mean(MATH_ISOLATE)`, from
perf_data/latest/perf_sfpu_sdpa/perf_sfpu_sdpa.post.csv.
"""

import pytest
from conftest import skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    PerfRunType,
    SdpaPerfOp,
)
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    LOOP_FACTOR,
    SDPA_EXP_SCALE,
    SDPA_PERF_OP,
    TILE_COUNT,
)

# 0.25 as a bfloat16 bit pattern: a non-identity scale, so the scale multiply is not folded away.
BF16_QUARTER = 0x3E80

# 128x64 input: 8 tiles, one full bf16 SyncHalf DEST bank (two fp32 blocks of four).
TILE_CNT = 8


@skip_for_wormhole
@pytest.mark.perf
@parametrize(
    formats=[InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)],
    sdpa_perf_op=list(SdpaPerfOp),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    loop_factor=[16],
)
def test_perf_sfpu_sdpa(perf_report, formats, sdpa_perf_op, dest_acc, loop_factor):
    configuration = PerfConfig(
        "sources/sfpu_sdpa_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            SDPA_PERF_OP(sdpa_perf_op),
            # Read only by the SDPA headers' APPROX (the reciprocal bodies, not timed here).
            APPROX_MODE(ApproximationMode.No),
            SDPA_EXP_SCALE(scale_bf16=BF16_QUARTER),
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
        dest_acc=dest_acc,
    )

    configuration.run(perf_report)
