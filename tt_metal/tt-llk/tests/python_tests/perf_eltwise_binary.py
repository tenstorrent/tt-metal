# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import (
    BroadcastType,
    DestAccumulation,
    DestSync,
    EltwiseBinaryReuseDestType,
    MathFidelity,
    Transpose,
)
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from test_eltwise_binary import (
    BASE_PERF_MATH_OPS,
    BFP4_PERF_MATH_OPS,
    DEST_REUSE_MATH_OPS,
    INT8_FORMAT,
    INT8_MATH_OPS,
    _get_valid_math_fidelity,
    _run_eltwise_binary_dest_reuse_test,
    _run_eltwise_binary_test,
    get_base_perf_formats,
    get_bfp4_formats,
    get_dest_reuse_formats,
    get_dest_reuse_perf_input_dimensions,
    get_dest_reuse_perf_output_dimensions,
    get_dest_reuse_perf_tile_dimensions,
    get_eltwise_binary_perf_acc_to_dest,
    get_eltwise_binary_perf_input_dimensions,
    get_eltwise_binary_perf_tile_dimensions,
)

PERF_LOOP_FACTOR = 32


@pytest.mark.perf
@parametrize(
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    dest_sync=[DestSync.Half],
    unpack_to_dest=[False],
    formats=get_base_perf_formats,
    broadcast_type=[
        BroadcastType.None_,
        BroadcastType.Row,
        BroadcastType.Column,
        BroadcastType.Scalar,
    ],
    math_op=BASE_PERF_MATH_OPS,
    math_fidelity=lambda formats, math_op: _get_valid_math_fidelity(formats, math_op),
    transpose_srca=[Transpose.Yes, Transpose.No],
    tile_dimensions=get_eltwise_binary_perf_tile_dimensions,
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
    broadcast_type,
    math_op,
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
        broadcast_type,
        math_op,
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
    broadcast_type=[
        BroadcastType.None_,
        BroadcastType.Row,
        BroadcastType.Column,
        BroadcastType.Scalar,
    ],
    math_op=BFP4_PERF_MATH_OPS,
    math_fidelity=lambda formats, math_op: _get_valid_math_fidelity(formats, math_op),
    transpose_srca=[Transpose.No],
    tile_dimensions=get_eltwise_binary_perf_tile_dimensions,
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
    broadcast_type,
    math_op,
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
        broadcast_type,
        math_op,
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
    reuse_dest_type=[
        EltwiseBinaryReuseDestType.DEST_TO_SRCA,
        EltwiseBinaryReuseDestType.DEST_TO_SRCB,
    ],
    math_op=DEST_REUSE_MATH_OPS,
    formats=get_dest_reuse_formats,
    dest_acc=[DestAccumulation.No],
    dest_sync=[DestSync.Half],
    unpack_to_dest=[False],
    math_fidelity=lambda formats, math_op: _get_valid_math_fidelity(formats, math_op),
    tile_dimensions=lambda: get_dest_reuse_perf_tile_dimensions(),
    input_dimensions=get_dest_reuse_perf_input_dimensions,
    output_dimensions=get_dest_reuse_perf_output_dimensions,
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[PERF_LOOP_FACTOR],
    is_perf=[True],
)
def test_perf_eltwise_binary_dest_reuse(
    perf_report,
    reuse_dest_type,
    math_op,
    formats,
    dest_acc,
    dest_sync,
    unpack_to_dest,
    math_fidelity,
    tile_dimensions,
    input_dimensions,
    output_dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    _run_eltwise_binary_dest_reuse_test(
        reuse_dest_type,
        math_op,
        formats,
        dest_acc,
        dest_sync,
        unpack_to_dest,
        math_fidelity,
        tile_dimensions,
        input_dimensions,
        output_dimensions,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )


@pytest.mark.perf
@parametrize(
    dest_acc=[DestAccumulation.Yes],
    dest_sync=[DestSync.Half],
    unpack_to_dest=[False],
    formats=INT8_FORMAT,
    broadcast_type=[BroadcastType.None_],
    math_op=INT8_MATH_OPS,
    math_fidelity=[MathFidelity.LoFi],
    transpose_srca=[Transpose.No],
    tile_dimensions=get_eltwise_binary_perf_tile_dimensions,
    input_dimensions=get_eltwise_binary_perf_input_dimensions,
    acc_to_dest=get_eltwise_binary_perf_acc_to_dest,
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[PERF_LOOP_FACTOR],
    is_perf=[True],
)
def test_perf_eltwise_binary_int8_format(
    perf_report,
    dest_acc,
    dest_sync,
    unpack_to_dest,
    formats,
    broadcast_type,
    math_op,
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
        broadcast_type,
        math_op,
        math_fidelity,
        transpose_srca,
        input_dimensions,
        tile_dimensions,
        acc_to_dest,
        int8_inputs=True,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
