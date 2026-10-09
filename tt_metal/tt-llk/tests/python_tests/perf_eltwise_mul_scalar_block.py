# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf of the Blackhole scalar block multiply (mul_tiles_bcast_scalar_block, sources/eltwise_mul_scalar_block_perf.cpp);
unit: one 32x32 tile, TILE_COUNT is the block size."""

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync, MathFidelity, PerfRunType
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    DEST_INDEX,
    DEST_SYNC,
    LOOP_FACTOR,
    MATH_FIDELITY,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
)

pytestmark = [skip_for_wormhole, skip_for_quasar]

BF16, FP32 = DataFormat.Float16_b, DataFormat.Float32

# (output format, dest_acc, block size, fidelity)
VARIANTS = [
    (FP32, acc, block, MathFidelity.LoFi)
    for acc in (DestAccumulation.No, DestAccumulation.Yes)
    for block in (1, 2, 4, 8)
]
VARIANTS += [
    (BF16, DestAccumulation.No, block, MathFidelity.LoFi) for block in (1, 4, 8)
]
VARIANTS += [
    (out_fmt, acc, block, fidelity)
    for fidelity in (MathFidelity.HiFi2, MathFidelity.HiFi4)
    for out_fmt, acc in (
        (FP32, DestAccumulation.No),
        (FP32, DestAccumulation.Yes),
        (BF16, DestAccumulation.No),
    )
    for block in (1, 8)
]


@pytest.mark.perf
@parametrize(variant=VARIANTS)
def test_perf_eltwise_mul_scalar_block(perf_report, variant):
    if len(variant) == 1:  # parametrize hands a single axis as a one-element tuple
        (variant,) = variant
    out_fmt, dest_acc, block, math_fidelity = variant
    configuration = PerfConfig(
        "sources/eltwise_mul_scalar_block_perf.cpp",
        InputOutputFormat(BF16, out_fmt),
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
        ],
        templates=[DEST_SYNC(DestSync.Half), MATH_FIDELITY(math_fidelity)],
        runtimes=[
            NUM_BLOCKS(1),
            NUM_TILES_IN_BLOCK(block),
            DEST_INDEX(0),
            TILE_COUNT(block),
            LOOP_FACTOR(128),
        ],
        variant_stimuli=StimuliConfig(
            None,
            BF16,
            None,
            BF16,
            out_fmt,
            tile_count_A=block,
            tile_count_B=1,
            tile_count_res=block,
        ),
        dest_acc=dest_acc,
    )
    configuration.run(perf_report)
