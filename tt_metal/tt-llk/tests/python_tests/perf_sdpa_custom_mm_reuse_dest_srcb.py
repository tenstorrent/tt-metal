# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf sweep of the Blackhole sdpa_custom_mm_reuse_dest_srcb LLK (sources/sdpa_custom_mm_reuse_dest_srcb_perf.cpp, the perf
twin of test_sdpa_custom_mm_reuse_dest_srcb.py); the figures are per V (in1) tile."""

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, MathFidelity, PerfRunType
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    MATH_FIDELITY,
    SDPA_CUSTOM_MM_REUSE_DEST,
    SDPA_REUSE_DEST_LAYOUT,
    TILE_COUNT,
)

pytestmark = [skip_for_wormhole, skip_for_quasar]

BF16 = DataFormat.Float16_b

# (kt, nt): nt 1 to 16 at kt 2 and 4 (the tt-llk test's kt), and the flash_mla chunk depth kt 8 at nt 1 and 4.
CALL_SHAPES = [(kt, nt) for kt in (2, 4) for nt in (1, 4, 8, 16)] + [(8, 1), (8, 4)]


@pytest.mark.perf
@parametrize(
    math_fidelity=[MathFidelity.LoFi],
    call_shape=CALL_SHAPES,
    # O at DEST tile 0 with P above it is the SDPA chunk's placement; P at tile 0 with O above it the functional test's.
    dst_first=[True, False],
)
def test_perf_sdpa_custom_mm_reuse_dest_srcb(
    perf_report, math_fidelity, call_shape, dst_first
):
    kt, nt = call_shape
    configuration = PerfConfig(
        "sources/sdpa_custom_mm_reuse_dest_srcb_perf.cpp",
        InputOutputFormat(BF16, BF16),
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
        ],
        templates=[
            MATH_FIDELITY(math_fidelity),
            SDPA_CUSTOM_MM_REUSE_DEST(kt_dim=kt, nt_dim=nt),
            SDPA_REUSE_DEST_LAYOUT(dst_first=dst_first),
        ],
        runtimes=[
            TILE_COUNT(kt * nt),
            LOOP_FACTOR(256 if kt * nt < 8 else 64),
        ],
        variant_stimuli=StimuliConfig(
            None,
            BF16,
            None,
            BF16,
            BF16,
            tile_count_A=kt * nt,
            tile_count_B=(kt + 1) // 2,
            tile_count_res=nt,
        ),
        dest_acc=DestAccumulation.No,
    )
    configuration.run(perf_report)
