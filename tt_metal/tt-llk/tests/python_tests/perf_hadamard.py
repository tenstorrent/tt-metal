# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf of the Blackhole H128 Hadamard transform (hadamard_h128, sources/hadamard_perf.cpp); unit: one 128-element
vector."""

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync, MathFidelity, PerfRunType
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    HADAMARD,
    LOOP_FACTOR,
    MATH_FIDELITY,
    NUM_FACES,
    TILE_COUNT,
)

pytestmark = [skip_for_wormhole, skip_for_quasar]

BF16 = DataFormat.Float16_b

# (vectors per section, fidelity, normalise)
VARIANTS = [
    (num_vectors, fidelity, normalize)
    for num_vectors in (1, 4, 8)
    for fidelity in (MathFidelity.LoFi, MathFidelity.HiFi4)
    for normalize in (False, True)
]


@pytest.mark.perf
@parametrize(variant=VARIANTS)
def test_perf_hadamard(perf_report, variant):
    if len(variant) == 1:  # parametrize hands a single axis as a one-element tuple
        (variant,) = variant
    num_vectors, fidelity, normalize = variant
    configuration = PerfConfig(
        "sources/hadamard_perf.cpp",
        InputOutputFormat(BF16, BF16),
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
        ],
        templates=[
            HADAMARD(hadamard_normalize=normalize, h16_tile_index=0),
            MATH_FIDELITY(fidelity),
            DEST_SYNC(DestSync.Half),
        ],
        runtimes=[TILE_COUNT(num_vectors), NUM_FACES(1, 1, 1), LOOP_FACTOR(128)],
        variant_stimuli=StimuliConfig(
            None,
            BF16,
            None,
            BF16,
            BF16,
            tile_count_A=1,
            tile_count_B=num_vectors,
            tile_count_res=num_vectors,
            num_faces=1,
        ),
        dest_acc=DestAccumulation.No,
    )
    configuration.run(perf_report)
