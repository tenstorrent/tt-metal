# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf of the Blackhole rmsnorm bcast-scalar dest-reuse multiply (rmsnorm_bcast_scalar_reuse_tiles,
sources/rmsnorm_bcast_scalar_dest_reuse_perf.cpp); unit: one 32x32 tile."""

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    DestAccumulation,
    MathFidelity,
    MathOperation,
    PerfRunType,
)
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    MATH_FIDELITY,
    MATH_OP,
    RMSNORM_DEST_REUSE,
    TILE_COUNT,
)

pytestmark = [skip_for_wormhole, skip_for_quasar]

BF16 = DataFormat.Float16_b

# (op, fidelity, tiles per row, clear_dest, whole_tile)
VARIANTS = []
for num_tiles in (1, 2, 4, 8):
    for fidelity in (MathFidelity.LoFi, MathFidelity.HiFi2, MathFidelity.HiFi4):
        VARIANTS.append((MathOperation.Elwmul, fidelity, num_tiles, True, False))
    VARIANTS.append((MathOperation.Elwadd, MathFidelity.LoFi, num_tiles, False, False))
    for fidelity in (MathFidelity.HiFi2, MathFidelity.HiFi3, MathFidelity.HiFi4):
        VARIANTS.append((MathOperation.Elwmul, fidelity, num_tiles, True, True))
VARIANTS.append((MathOperation.Elwmul, MathFidelity.HiFi3, 1, True, False))


@pytest.mark.perf
@parametrize(variant=VARIANTS)
def test_perf_rmsnorm_bcast_scalar_dest_reuse(perf_report, variant):
    if len(variant) == 1:  # parametrize hands a single axis as a one-element tuple
        (variant,) = variant
    mathop, fidelity, num_tiles, clear_dest, whole_tile = variant
    configuration = PerfConfig(
        "sources/rmsnorm_bcast_scalar_dest_reuse_perf.cpp",
        InputOutputFormat(BF16, BF16),
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
        ],
        templates=[
            MATH_OP(mathop=mathop),
            MATH_FIDELITY(fidelity),
            RMSNORM_DEST_REUSE(
                rmsnorm_num_tiles=num_tiles,
                rmsnorm_num_faces=4,
                clear_dest=clear_dest,
                unpack_full_transpose=False,
                rmsnorm_whole_tile=whole_tile,
            ),
        ],
        runtimes=[TILE_COUNT(num_tiles), LOOP_FACTOR(128)],
        variant_stimuli=StimuliConfig(
            None,
            BF16,
            None,
            BF16,
            BF16,
            tile_count_A=num_tiles,
            tile_count_B=1,
            tile_count_res=num_tiles,
        ),
        unpack_to_dest=False,
        dest_acc=DestAccumulation.No,
    )
    configuration.run(perf_report)
