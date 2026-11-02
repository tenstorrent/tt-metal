# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from helpers.constraints import get_valid_eltwise_binary_math_fidelities
from helpers.format_config import DataFormat
from helpers.llk_params import (
    BroadcastType,
    DestAccumulation,
    DestSync,
    MathOperation,
    Transpose,
)
from helpers.param_config import input_output_formats, parametrize
from test_eltwise_binary import (
    BFP4_MATH_OPS,
    _get_valid_formats,
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
    math_fidelity=lambda formats, math_op: get_valid_eltwise_binary_math_fidelities(
        formats, math_op
    ),
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
    math_fidelity=lambda formats: get_valid_eltwise_binary_math_fidelities(formats),
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


# SrcB in its own tile shape, as production feeds it: a [1,32] row for ROW and SCALAR, and a
# 16-column tile with SrcA's rows of faces for COL. Each SrcB tile sits in an SrcA-tile-sized L1
# slot with poison past its end (see _srcb_own_shape_operands).
SRCB_SHAPE_CASES = {
    BroadcastType.Row: [
        ([16, 16], [1, 32]),
        ([2, 32], [1, 32]),
        ([4, 32], [1, 32]),
        ([8, 32], [1, 32]),
        ([16, 32], [1, 32]),
        ([32, 32], [1, 32]),
    ],
    BroadcastType.Scalar: [
        ([16, 16], [1, 32]),
        ([2, 32], [1, 32]),
        ([8, 32], [1, 32]),
        ([16, 32], [1, 32]),
        ([32, 32], [1, 32]),
        ([32, 16], [1, 32]),
    ],
    BroadcastType.Column: [
        ([8, 32], [16, 16]),
        ([16, 32], [16, 16]),
        ([32, 32], [32, 16]),
    ],
}


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    broadcast_type=list(SRCB_SHAPE_CASES),
    tile_shapes=lambda broadcast_type: SRCB_SHAPE_CASES[broadcast_type],
    math_op=[MathOperation.Elwadd, MathOperation.Elwmul],
    # Only 32x32 SrcA tiles transpose, and scalar broadcast does not support it.
    transpose_srca=lambda broadcast_type, tile_shapes: (
        [Transpose.No, Transpose.Yes]
        if tile_shapes[0] == [32, 32] and broadcast_type != BroadcastType.Scalar
        else [Transpose.No]
    ),
)
def test_eltwise_binary_broadcast_srcb_shape(
    formats, broadcast_type, tile_shapes, math_op, transpose_srca
):
    """SrcB unpacks from its own tile shape, not SrcA's."""
    tile_dimensions, tile_dimensions_B = tile_shapes
    _run_eltwise_binary_test(
        DestAccumulation.No,
        DestSync.Half,
        False,
        formats,
        broadcast_type,
        math_op,
        get_valid_eltwise_binary_math_fidelities(formats, math_op)[-1],
        transpose_srca,
        [tile_dimensions[0] * 2, tile_dimensions[1] * 2],
        tile_dimensions,
        False,
        tile_dimensions_B=tile_dimensions_B,
    )
