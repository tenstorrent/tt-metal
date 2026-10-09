# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf sweep of the Blackhole sdpa_custom_mm LLK (sources/sdpa_custom_mm_perf.cpp, the perf twin of
test_sdpa_custom_mm.py); the per-tile figures are per K (in1) tile."""

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, PerfRunType
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    IN_FACE_DIMS,
    LOOP_FACTOR,
    NUM_FACES,
    SDPA_CUSTOM_MM_FLAGS,
    TILE_COUNT,
)

pytestmark = [skip_for_wormhole, skip_for_quasar]

BF16 = DataFormat.Float16_b
# Depth of a Tensix semaphore (ckernel_structs.h: SEMAPHORE_BIT_COUNT 4).
SEMAPHORE_MAX_VALUE = 15

# (ct, kt, signal_granularity, transposed K): granularity 1 (the DeepSeek decode cadence) against one post per call, at
# ct 1 to 16, and FlashMLA decode's Q K^T call (k chunk 4 tiles, head_dim 576, K read transposed).
CALL_SHAPES = [
    (ct, kt, sg, False)
    for ct in (1, 4, 8, 16)
    for kt in (2, 8)
    for sg in sorted({1, ct})
    if ct // sg <= SEMAPHORE_MAX_VALUE
] + [(4, 18, 1, True)]


@pytest.mark.perf
@parametrize(
    in0_face_r_dim=[8],
    # FlashMLA reads K from its Bfp8_b KV cache.
    in1_format=[DataFormat.Float16_b, DataFormat.Bfp8_b],
    call_shape=CALL_SHAPES,
)
def test_perf_sdpa_custom_mm(perf_report, in0_face_r_dim, in1_format, call_shape):
    ct, kt, signal_granularity, transposed = call_shape
    if ct // signal_granularity > SEMAPHORE_MAX_VALUE:
        raise ValueError(
            f"ct_dim / signal_granularity = {ct // signal_granularity} FPU->SFPU posts per call exceed the "
            f"4-bit Tensix semaphore ({SEMAPHORE_MAX_VALUE}); the core would hang"
        )
    configuration = PerfConfig(
        "sources/sdpa_custom_mm_perf.cpp",
        InputOutputFormat(in1_format, BF16, BF16),
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
        ],
        templates=[
            CRK_TILE_DIMM(c_dimm=ct, r_dimm=1, k_dimm=kt),
            SDPA_CUSTOM_MM_FLAGS(
                signal_granularity=signal_granularity,
                read_transposed=transposed,
                mm_transpose=transposed,
            ),
        ],
        runtimes=[
            # in1 (SrcA) has 4 full faces; in0 (SrcB) and the result have 2 faces of M rows.
            NUM_FACES(num_faces=2, num_faces_A=4, num_faces_B=2),
            IN_FACE_DIMS(in0_face_r_dim=in0_face_r_dim),
            TILE_COUNT(kt * ct),
            LOOP_FACTOR(256 if kt * ct < 8 else 64),
        ],
        variant_stimuli=StimuliConfig(
            None,
            in1_format,
            None,
            BF16,
            BF16,
            tile_count_A=kt * ct,
            tile_count_B=kt,
            tile_count_res=ct,
        ),
        dest_acc=DestAccumulation.No,
    )
    configuration.run(perf_report)
