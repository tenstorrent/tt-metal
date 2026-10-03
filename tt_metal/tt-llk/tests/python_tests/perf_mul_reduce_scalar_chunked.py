# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf of the Blackhole chunked mul_reduce_scalar row (mul_reduce_scalar_chunked_tile,
sources/mul_reduce_scalar_chunked_perf.cpp); unit: one input tile of the row."""

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
    MUL_REDUCE_SCALAR_CHUNK_SIZE,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    TILE_COUNT,
)

pytestmark = [skip_for_wormhole, skip_for_quasar]

BF16 = DataFormat.Float16_b

# (fidelity, DEST slots including the accumulator, tiles per row)
VARIANTS = [
    (MathFidelity.LoFi, 8, 9),
    (MathFidelity.LoFi, 8, 14),
    (MathFidelity.LoFi, 8, 15),
    (MathFidelity.LoFi, 8, 28),
    (MathFidelity.HiFi2, 8, 14),
    (MathFidelity.HiFi4, 8, 14),
    (MathFidelity.LoFi, 4, 12),
]


@pytest.mark.perf
@parametrize(variant=VARIANTS)
def test_perf_mul_reduce_scalar_chunked(perf_report, variant):
    if len(variant) == 1:  # parametrize hands a single axis as a one-element tuple
        (variant,) = variant
    fidelity, chunk_size, num_tiles = variant
    configuration = PerfConfig(
        "sources/mul_reduce_scalar_chunked_perf.cpp",
        InputOutputFormat(BF16, BF16),
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
        ],
        templates=[MATH_FIDELITY(fidelity)],
        runtimes=[
            TILE_COUNT(num_tiles),
            MUL_REDUCE_SCALAR_CHUNK_SIZE(chunk_size),
            NUM_FACES_R_DIM(2, 2),
            NUM_FACES_C_DIM(2, 2),
            LOOP_FACTOR(32),
        ],
        variant_stimuli=StimuliConfig(
            None,
            BF16,
            None,
            BF16,
            BF16,
            tile_count_A=num_tiles,
            tile_count_B=num_tiles,
            tile_count_res=1,
            sfpu=False,
        ),
        dest_acc=DestAccumulation.No,
    )
    configuration.run(perf_report)
