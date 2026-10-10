# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf of the gate path of the Blackhole generalized MoE gate (sources/generalized_moe_gate_perf.cpp), ungrouped and
grouped, cut by PERF_STAGE; unit: one token."""

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    DestSync,
    MathFidelity,
    MathOperation,
    PerfRunType,
)
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    DEST_SYNC,
    GENERALIZED_MOE_GATE_PERF_PATH,
    LOOP_FACTOR,
    MATH_FIDELITY,
    MATH_OP,
    NUM_FACES,
    PERF_STAGE,
    TILE_COUNT,
)

pytestmark = [skip_for_wormhole, skip_for_quasar]

FORMATS = InputOutputFormat(DataFormat.Float16_b, DataFormat.UInt16)

# (grouped, stage, approximate reciprocal)
VARIANTS = [
    (grouped, stage, ApproximationMode.No)
    for grouped in (False, True)
    for stage in (2, 1, 0)
]
VARIANTS.append((False, 2, ApproximationMode.Yes))


@pytest.mark.perf
@parametrize(variant=VARIANTS)
def test_perf_generalized_moe_gate(perf_report, variant):
    if len(variant) == 1:  # parametrize hands a single axis as a one-element tuple
        (variant,) = variant
    grouped, stage, approx = variant
    configuration = PerfConfig(
        "sources/generalized_moe_gate_perf.cpp",
        FORMATS,
        run_types=[PerfRunType.L1_TO_L1, PerfRunType.MATH_ISOLATE],
        templates=[
            GENERALIZED_MOE_GATE_PERF_PATH("grouped" if grouped else "ungrouped"),
            MATH_OP(mathop=MathOperation.Elwadd),
            MATH_FIDELITY(MathFidelity.HiFi4),
            APPROX_MODE(approx),
            DEST_SYNC(DestSync.Half),
            PERF_STAGE(stage),
        ],
        runtimes=[TILE_COUNT(1), NUM_FACES(4, 4, 4), LOOP_FACTOR(32)],
        variant_stimuli=StimuliConfig(
            None,
            DataFormat.Float16_b,
            None,
            DataFormat.Float16_b,
            DataFormat.UInt16,
            tile_count_A=4,
            tile_count_B=1,
            tile_count_res=4,
            num_faces=4,
        ),
        dest_acc=DestAccumulation.No,
    )
    configuration.run(perf_report)
