# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
from helpers.format_config import DataFormat, FormatConfig, is_dest_acc_needed
from helpers.llk_params import DestAccumulation, MathFidelity
from helpers.matmul_sweep import (
    generate_matmul_dimension_combinations,
    generate_tile_dims,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf import PerfRunType, perf_benchmark, update_report

# Important K dimensions to test
KT_DIMS = [1, 2, 3, 4, 8, 64]


def matmul_combos(
    formats: List[FormatConfig],
    dest_acc: List[DestAccumulation],
):
    def _dest_bank_max_tiles(format: FormatConfig, dest_acc: DestAccumulation):
        if is_dest_acc_needed(format) or dest_acc == DestAccumulation.Yes:
            return 4
        return 8

    unique_max_tiles = set(
        _dest_bank_max_tiles(fmt, acc) for fmt in formats for acc in dest_acc
    )
    dimensions = {
        max_tiles: generate_matmul_dimension_combinations(max_tiles, kt_dims=KT_DIMS)
        for max_tiles in unique_max_tiles
    }

    return [
        (format, accumulation, dims)
        for format in formats
        for accumulation in dest_acc
        for dims in dimensions[_dest_bank_max_tiles(format, accumulation)]
    ]


@pytest.mark.perf
@parametrize(
    test_name="matmul_perf",
    combos=matmul_combos(
        formats=input_output_formats(
            [
                DataFormat.Float16_b,
                DataFormat.Float16,
                DataFormat.Float32,
                DataFormat.Bfp8_b,
            ]
        ),
        dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    ),
    math_fidelity=[
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
)
def test_perf_matmul(perf_report, test_name, combos, math_fidelity):

    formats, dest_acc, (matrix_a, matrix_b) = combos

    if is_dest_acc_needed(formats) and dest_acc == DestAccumulation.No:
        pytest.skip("Dest accumulation must be enabled for this format")

    run_types = [
        PerfRunType.L1_TO_L1,
        PerfRunType.UNPACK_ISOLATE,
        PerfRunType.MATH_ISOLATE,
        PerfRunType.PACK_ISOLATE,
        PerfRunType.L1_CONGESTION,
    ]

    # Calculate all matmul dimensions using helper function
    dims = generate_tile_dims((matrix_a, matrix_b))

    test_config = {
        "formats": formats,
        "testname": test_name,
        "loop_factor": 16,
        "tile_cnt": dims.rt_dim * dims.ct_dim * dims.kt_dim,
        "input_A_dimensions": matrix_a,
        "input_B_dimensions": matrix_b,
        "output_dimensions": dims.output_dimensions,
        "rt_dim": dims.rt_dim,
        "ct_dim": dims.ct_dim,
        "kt_dim": dims.kt_dim,
        "dest_acc": dest_acc,
        "math_fidelity": math_fidelity,
    }

    results = perf_benchmark(test_config, run_types)
    update_report(perf_report, test_config, results)
