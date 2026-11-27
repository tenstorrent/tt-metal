# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.constraints import get_valid_math_fidelities
from helpers.llk_params import (
    BroadcastType,
    DestAccumulation,
    DestSync,
    Transpose,
)
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from test_eltwise_binary import (
    BFP4_PERF_MATH_OPS,
    _run_eltwise_binary_test,
    get_base_perf_formats,
    get_bfp4_formats,
    get_eltwise_binary_math_ops,
    get_eltwise_binary_perf_acc_to_dest,
    get_eltwise_binary_perf_input_dimensions,
    get_eltwise_binary_perf_tile_dimensions,
    get_eltwise_binary_transpose,
)

PERF_LOOP_FACTOR = 32


@pytest.mark.perf
@parametrize(
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    dest_sync=[DestSync.Half],
    unpack_to_dest=[False],
    formats=get_base_perf_formats,
    mathop=lambda formats: get_eltwise_binary_math_ops(formats, is_perf=True),
    math_fidelity=lambda formats, mathop: get_valid_math_fidelities(formats, mathop),
    transpose_srca=get_eltwise_binary_transpose,
    tile_dimensions=lambda transpose_srca: get_eltwise_binary_perf_tile_dimensions(
        transpose_srca, BroadcastType.None_
    ),
    input_dimensions=get_eltwise_binary_perf_input_dimensions,
    acc_to_dest=get_eltwise_binary_perf_acc_to_dest,
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[PERF_LOOP_FACTOR],
    is_perf=[True],
)
def test_perf_eltwise_binary(
    perf_report,
    dest_acc,
    dest_sync,
    unpack_to_dest,
    formats,
    mathop,
    math_fidelity,
    transpose_srca,
    tile_dimensions,
    input_dimensions,
    acc_to_dest,
    run_types,
    loop_factor,
    is_perf,
):
    _run_eltwise_binary_test(
        dest_acc,
        dest_sync,
        unpack_to_dest,
        formats,
        BroadcastType.None_,
        mathop,
        math_fidelity,
        transpose_srca,
        input_dimensions,
        tile_dimensions,
        acc_to_dest,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )


@pytest.mark.perf
@parametrize(
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    dest_sync=[DestSync.Half],
    unpack_to_dest=[False],
    formats=lambda: get_bfp4_formats(),
    mathop=BFP4_PERF_MATH_OPS,
    math_fidelity=lambda formats, mathop: get_valid_math_fidelities(formats, mathop),
    transpose_srca=[Transpose.No],
    tile_dimensions=lambda transpose_srca: get_eltwise_binary_perf_tile_dimensions(
        transpose_srca, BroadcastType.None_
    ),
    input_dimensions=get_eltwise_binary_perf_input_dimensions,
    acc_to_dest=get_eltwise_binary_perf_acc_to_dest,
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[PERF_LOOP_FACTOR],
    is_perf=[True],
)
def test_perf_eltwise_binary_bfp4_b(
    perf_report,
    dest_acc,
    dest_sync,
    unpack_to_dest,
    formats,
    mathop,
    math_fidelity,
    transpose_srca,
    tile_dimensions,
    input_dimensions,
    acc_to_dest,
    run_types,
    loop_factor,
    is_perf,
):
    _run_eltwise_binary_test(
        dest_acc,
        dest_sync,
        unpack_to_dest,
        formats,
        BroadcastType.None_,
        mathop,
        math_fidelity,
        transpose_srca,
        input_dimensions,
        tile_dimensions,
        acc_to_dest,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
