# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import (
    DestAccumulation,
    MathOperation,
    ReduceDimension,
    ReducePool,
)
from helpers.param_config import (
    input_output_formats,
    parametrize,
)
from helpers.perf import (
    ALL_RUN_TYPES,
    perf_benchmark,
    update_report,
)

REDUCE_MATHOP = {
    ReduceDimension.Row: MathOperation.ReduceRow,
    ReduceDimension.Column: MathOperation.ReduceColumn,
    ReduceDimension.Scalar: MathOperation.ReduceScalar,
}


@pytest.mark.perf
@parametrize(
    test_name="reduce_perf",
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,
            DataFormat.Bfp8_b,
        ]
    ),
    tile_count=16,
    dest_acc=[DestAccumulation.No],
    reduce_dim=[ReduceDimension.Row, ReduceDimension.Column, ReduceDimension.Scalar],
    pool_type=[ReducePool.Max, ReducePool.Average, ReducePool.Sum],
)
def test_perf_reduce(
    perf_report, test_name, formats, tile_count, dest_acc, reduce_dim, pool_type
):

    test_config = {
        "testname": test_name,
        "tile_cnt": tile_count,
        "formats": formats,
        "dest_acc": dest_acc,
        "pool_type": pool_type,
        "mathop": REDUCE_MATHOP[reduce_dim],
    }

    results = perf_benchmark(test_config, ALL_RUN_TYPES)
    update_report(perf_report, test_config, results)
