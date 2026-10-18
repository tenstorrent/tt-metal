# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from helpers.llk_params import BroadcastType, DestAccumulation, DestSync, Transpose
from helpers.param_config import parametrize
from test_eltwise_binary import (
    BFP4_MATH_OPS,
    _get_valid_formats,
    _get_valid_math_fidelity,
    _run_eltwise_binary_test,
    get_bfp4_formats,
    get_eltwise_binary_acc_to_dest,
    get_eltwise_binary_input_dimensions,
    get_eltwise_binary_math_ops,
    get_eltwise_binary_tile_dimensions,
    get_eltwise_binary_transpose,
)

ELTWISE_BINARY_BROADCAST_TYPES = [
    BroadcastType.Row,
    BroadcastType.Column,
    BroadcastType.Scalar,
]


@parametrize(
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    dest_sync=[DestSync.Half],
    unpack_to_dest=[False],
    formats=lambda dest_acc: _get_valid_formats(dest_acc),
    broadcast_type=ELTWISE_BINARY_BROADCAST_TYPES,
    math_op=lambda formats: get_eltwise_binary_math_ops(formats),
    math_fidelity=lambda formats, math_op: _get_valid_math_fidelity(formats, math_op),
    transpose_srca=get_eltwise_binary_transpose,
    tile_dimensions=lambda transpose_srca, broadcast_type: get_eltwise_binary_tile_dimensions(
        transpose_srca, broadcast_type
    ),
    input_dimensions=lambda dest_acc, dest_sync, formats, tile_dimensions: get_eltwise_binary_input_dimensions(
        dest_acc,
        dest_sync,
        formats,
        tile_dimensions,
        existing_dimensions=[[256, 32]],
    ),
    acc_to_dest=get_eltwise_binary_acc_to_dest,
)
def test_eltwise_binary_broadcast(
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
    )


@parametrize(
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    dest_sync=[DestSync.Half],
    unpack_to_dest=[False],
    formats=get_bfp4_formats(),
    broadcast_type=ELTWISE_BINARY_BROADCAST_TYPES,
    math_fidelity=lambda formats: _get_valid_math_fidelity(formats),
    transpose_srca=Transpose.No,
    math_op=BFP4_MATH_OPS,
    tile_dimensions=lambda transpose_srca, broadcast_type: get_eltwise_binary_tile_dimensions(
        transpose_srca, broadcast_type
    ),
    input_dimensions=lambda dest_acc, dest_sync, formats, tile_dimensions: get_eltwise_binary_input_dimensions(
        dest_acc,
        dest_sync,
        formats,
        tile_dimensions,
        existing_dimensions=[[32, 32], [64, 32], [32, 64], [256, 32]],
    ),
    acc_to_dest=get_eltwise_binary_acc_to_dest,
)
def test_eltwise_binary_bfp4_b_broadcast(
    dest_acc,
    dest_sync,
    unpack_to_dest,
    formats,
    broadcast_type,
    math_fidelity,
    transpose_srca,
    math_op,
    input_dimensions,
    tile_dimensions,
    acc_to_dest,
):
    return _run_eltwise_binary_test(
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
    )
