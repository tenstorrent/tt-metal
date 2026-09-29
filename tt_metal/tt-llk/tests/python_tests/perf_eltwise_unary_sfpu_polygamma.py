# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Polygamma at the orders ttnn issues.

perf_eltwise_unary_sfpu.py sees Polygamma at order 1 (trigamma) only, but the kernel's
per-row cost grows with the order: the power chains are loops over the bits of n, and so
is the tail's inv_z^n. The order is a template parameter here (SFPU_POLYGAMMA_ORDER), so
the rows carry a polygamma_order column the shared unary sweep does not have -- which is
why this is its own file: a perf CSV must have one column schema.

Float32 -> Float32 needs the 32-bit Dest on Blackhole and runs with dest_acc on; bf16 runs
with it off.
"""

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    FastMode,
    FusedSort,
    MathOperation,
    StableSort,
    Transpose,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES, PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import calculate_tile_and_face_counts
from helpers.test_variant_parameters import (
    APPROX_MODE,
    CLAMP_NEGATIVE,
    FAST_MODE,
    FUSED_SORT,
    ITERATIONS,
    LOOP_FACTOR,
    MATH_OP,
    NUM_FACES,
    SFPU_POLYGAMMA_ORDER,
    STABLE_SORT,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)

# 1 is the shared sweep's order, kept so this file's rows compare against it directly;
# 11 is the largest order ttnn accepts (through polygamma_bw's n + 1).
POLYGAMMA_ORDERS = [1, 2, 3, 11]


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True),
    polygamma_order=POLYGAMMA_ORDERS,
    input_dimensions=[[128, 64]],  # tile_cnt: 8
)
def test_perf_eltwise_unary_sfpu_polygamma(
    perf_report, formats, polygamma_order, input_dimensions
):
    dest_acc = (
        DestAccumulation.Yes
        if formats.input_format.is_32_bit()
        else DestAccumulation.No
    )
    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )
    tile_count_A, tile_count_B, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions, input_dimensions, face_r_dim=16, num_faces=4
    )
    PerfConfig(
        "sources/eltwise_unary_sfpu_perf.cpp",
        formats,
        run_types=ALL_PERF_RUN_TYPES,
        templates=[
            MATH_OP(mathop=MathOperation.Polygamma),
            APPROX_MODE(ApproximationMode.No),
            ITERATIONS(32),
            FAST_MODE(FastMode.No),
            STABLE_SORT(StableSort.No),
            FUSED_SORT(FusedSort.No),
            CLAMP_NEGATIVE(False),
            SFPU_POLYGAMMA_ORDER(polygamma_order),
        ],
        runtimes=[
            TILE_COUNT(tile_count_A),
            LOOP_FACTOR(16),
            NUM_FACES(num_faces=faces_to_generate),
            UNPACK_TRANS_FACES(Transpose.No),
            UNPACK_TRANS_WITHIN_FACE(Transpose.No),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count_A,
            tile_count_B=tile_count_B,
            tile_count_res=tile_count_A,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    ).run(perf_report)
