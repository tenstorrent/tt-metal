# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf of the generic, SFPU-only DeepSeek MoE gate top-k (sources/sfpu_generic_moe_gate_topk_perf.cpp), PERF_STAGE 0
being the two datacopies alone; unit: one token."""

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, PerfRunType
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    MOE_GATE_NORMALIZE_PARAMS,
    MOE_GATE_TOPK,
    PERF_STAGE,
    TILE_COUNT,
)

pytestmark = [skip_for_wormhole, skip_for_quasar]

BF16 = DataFormat.Float16_b

# (selected experts, normalise, stage)
VARIANTS = [
    (8, False, 1),
    (8, True, 1),
    (16, False, 1),
    (16, True, 1),
    (4, True, 1),
    (12, True, 1),
    (8, False, 0),
]


@pytest.mark.perf
@parametrize(variant=VARIANTS)
def test_perf_sfpu_generic_moe_gate_topk(perf_report, variant):
    if len(variant) == 1:  # parametrize hands a single axis as a one-element tuple
        (variant,) = variant
    k, normalize, stage = variant
    configuration = PerfConfig(
        "sources/sfpu_generic_moe_gate_topk_perf.cpp",
        InputOutputFormat(BF16, BF16),
        run_types=[PerfRunType.L1_TO_L1, PerfRunType.MATH_ISOLATE],
        templates=[
            MOE_GATE_TOPK(
                num_selected_experts=k,
                num_total_experts=256,
                normalize=normalize,
                zero_tail=True,
                full_sort=True,
                generate_indices=True,
                scores_include_bias=False,
            ),
            MOE_GATE_NORMALIZE_PARAMS(eps_bits=0, scale_bits=0x3F800000),
            PERF_STAGE(stage),
        ],
        runtimes=[TILE_COUNT(1), LOOP_FACTOR(64)],
        variant_stimuli=StimuliConfig(
            None,
            BF16,
            None,
            BF16,
            BF16,
            tile_count_A=2,
            tile_count_B=1,
            tile_count_res=2,
        ),
        dest_acc=DestAccumulation.No,
        unpack_to_dest=False,
    )
    configuration.run(perf_report)
