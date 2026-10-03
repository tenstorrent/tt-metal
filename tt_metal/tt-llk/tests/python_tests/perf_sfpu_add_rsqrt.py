# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf of the add_rsqrt SFPU functor (sources/sfpu_add_rsqrt_perf.cpp), tile form and one-vector form, PERF_STAGE 0
being the datacopy frame alone; unit: one tile (one call for the one-vector form)."""

import struct

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    PerfRunType,
    VectorMode,
)
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    ITERATIONS,
    LOOP_FACTOR,
    PERF_STAGE,
    SFPU_FAST_APPROX,
    SFPU_INPUT_SCALE,
    SFPU_TYPED_BF16_STORE,
    SFPU_UNARY_SCALAR,
    TILE_COUNT,
    VECTOR_MODE,
)

pytestmark = [skip_for_wormhole, skip_for_quasar]

BF16 = DataFormat.Float16_b


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


# (approx, vector mode, iterations per call, stage)
VARIANTS = [
    (approx, mode, iterations, 1)
    for approx in (ApproximationMode.No, ApproximationMode.Yes)
    for mode, iterations in ((VectorMode.RC, 8), (VectorMode.RC_custom, 1))
]
VARIANTS.append((ApproximationMode.No, VectorMode.RC, 8, 0))


@pytest.mark.perf
@parametrize(variant=VARIANTS)
def test_perf_sfpu_add_rsqrt(perf_report, variant):
    if len(variant) == 1:  # parametrize hands a single axis as a one-element tuple
        (variant,) = variant
    approx, vector_mode, iterations, stage = variant
    configuration = PerfConfig(
        "sources/sfpu_add_rsqrt_perf.cpp",
        InputOutputFormat(BF16, BF16),
        run_types=[PerfRunType.L1_TO_L1, PerfRunType.MATH_ISOLATE],
        templates=[
            APPROX_MODE(approx),
            SFPU_FAST_APPROX(False),
            SFPU_TYPED_BF16_STORE(False),
            SFPU_INPUT_SCALE(_bits(1.0)),
            SFPU_UNARY_SCALAR(_bits(1e-6)),
            VECTOR_MODE(vector_mode),
            ITERATIONS(iterations),
            PERF_STAGE(stage),
        ],
        runtimes=[TILE_COUNT(1), LOOP_FACTOR(128)],
        variant_stimuli=StimuliConfig(
            None,
            BF16,
            None,
            BF16,
            BF16,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
        ),
        dest_acc=DestAccumulation.No,
    )
    configuration.run(perf_report)
