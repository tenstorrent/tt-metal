# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Perf of the Blackhole mul_reduce_scalar row (mul_reduce_scalar_tile, sources/mul_reduce_scalar_perf.cpp): a row of
1, 2, 4 or 8 32x32 tile pairs multiplied into DEST and reduced to one scalar, at LoFi, HiFi2 and HiFi4, bf16 in and
out, one masked pack of the reduced tile per row. All four run types; the isolates mock the row's data-valid cadence
(four SrcA and four SrcB per tile for the multiply, one of each for the reduce phase). Unit: one input tile of the
row. The copy form (sum_reduce_scalar_tile) has no row here: as a perf kernel it hung the core in every run type
while its functional test passes, and the hang is not understood yet.
"""

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
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    TILE_COUNT,
)

pytestmark = [skip_for_wormhole, skip_for_quasar]

BF16 = DataFormat.Float16_b

# (fidelity, tiles per row)
VARIANTS = [
    (fidelity, num_tiles)
    for num_tiles in (1, 2, 4, 8)
    for fidelity in (MathFidelity.LoFi, MathFidelity.HiFi2, MathFidelity.HiFi4)
]


@pytest.mark.perf
@parametrize(variant=VARIANTS)
def test_perf_mul_reduce_scalar(perf_report, variant):
    fidelity, num_tiles = variant
    configuration = PerfConfig(
        "sources/mul_reduce_scalar_perf.cpp",
        InputOutputFormat(BF16, BF16),
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
        ],
        templates=[MATH_FIDELITY(fidelity)],
        runtimes=[TILE_COUNT(num_tiles), NUM_FACES_R_DIM(2, 2), NUM_FACES_C_DIM(2, 2), LOOP_FACTOR(64)],
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
