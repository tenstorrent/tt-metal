# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf of the Blackhole H128 Hadamard transform; unit: one 128-element vector."""

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync, MathFidelity
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES, PerfConfig
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
        run_types=ALL_PERF_RUN_TYPES,
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


CALL_VARIANTS = [
    (calls, fidelity, normalize)
    for calls in (1, 16)
    for fidelity in (MathFidelity.LoFi, MathFidelity.HiFi4)
    for normalize in (False, True)
]


@pytest.mark.perf
@parametrize(variant=CALL_VARIANTS)
def test_perf_hadamard_call(perf_report, variant):
    """A fused-op caller's sequence: short init, one vector in its own DEST section, 8 rows packed to an address.
    One call is the latency of a call; 16 calls run back to back."""
    if len(variant) == 1:  # parametrize hands a single axis as a one-element tuple
        (variant,) = variant
    calls, fidelity, normalize = variant
    configuration = PerfConfig(
        "sources/hadamard_call_perf.cpp",
        InputOutputFormat(BF16, BF16),
        run_types=ALL_PERF_RUN_TYPES,
        templates=[
            HADAMARD(hadamard_normalize=normalize, h16_tile_index=0),
            MATH_FIDELITY(fidelity),
            DEST_SYNC(DestSync.Half),
        ],
        runtimes=[TILE_COUNT(1), NUM_FACES(1, 1, 1), LOOP_FACTOR(calls)],
        variant_stimuli=StimuliConfig(
            None,
            BF16,
            None,
            BF16,
            BF16,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
            num_faces=1,
        ),
        dest_acc=DestAccumulation.No,
    )
    configuration.run(perf_report)
