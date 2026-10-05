# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf sweep of the block reduce (_llk_unpack_AB_reduce_block_, _llk_math_reduce_block_).

REDUCE_ROW and REDUCE_SCALAR accumulate eight input tiles into one output tile, REDUCE_COL writes one output tile per input
tile; block_ct_dim is the number of tiles per call, 1 being the per tile calls, so each configuration carries its own
reference row. tile_cnt counts input tiles.
"""

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    DestAccumulation,
    MathFidelity,
    MathOperation,
    PerfRunType,
    ReduceDimension,
    ReducePool,
)
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    MATH_FIDELITY,
    MATH_OP,
    REDUCE_BLOCK_CT_DIM,
    REDUCE_POOL_TYPE,
    TILE_COUNT,
)

REDUCE_MATHOP = {
    ReduceDimension.Row: MathOperation.ReduceRow,
    ReduceDimension.Column: MathOperation.ReduceColumn,
    ReduceDimension.Scalar: MathOperation.ReduceScalar,
}


def _dest_accs(formats):
    if formats.input_format == DataFormat.Float32:
        return [DestAccumulation.Yes]
    if formats.input_format == DataFormat.Float16_b:
        return [DestAccumulation.No, DestAccumulation.Yes]
    return [DestAccumulation.No]


def _fidelities(pool_type):
    if pool_type == ReducePool.Max:
        return [MathFidelity.HiFi4]
    return [MathFidelity.LoFi, MathFidelity.HiFi4]


@skip_for_wormhole
@skip_for_quasar
@pytest.mark.perf
@parametrize(
    formats=[
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        InputOutputFormat(DataFormat.Bfp8_b, DataFormat.Bfp8_b),
        InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
    ],
    dest_acc=_dest_accs,
    reduce_dim=[ReduceDimension.Row, ReduceDimension.Column, ReduceDimension.Scalar],
    pool_type=[ReducePool.Max, ReducePool.Sum],
    math_fidelity=_fidelities,
    block_ct_dim=[1, 2, 4, 8],
)
def test_perf_reduce_block(
    perf_report,
    formats,
    dest_acc,
    reduce_dim,
    pool_type,
    math_fidelity,
    block_ct_dim,
):
    tile_count = 16
    configuration = PerfConfig(
        "sources/reduce_block_perf.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ],
        templates=[
            MATH_OP(mathop=REDUCE_MATHOP[reduce_dim]),
            REDUCE_POOL_TYPE(pool_type),
            MATH_FIDELITY(math_fidelity),
            REDUCE_BLOCK_CT_DIM(block_ct_dim),
        ],
        runtimes=[TILE_COUNT(tile_count), LOOP_FACTOR(64)],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        unpack_to_dest=False,
        dest_acc=dest_acc,
    )

    configuration.run(perf_report)
