# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from helpers.golden_generators import (
    BroadcastGolden,
    TransposeGolden,
    get_golden_generator,
)
from helpers.llk_params import BroadcastType, Transpose
from helpers.tilize_untilize import tilize_block, untilize_block


def broadcast_tile(tile, operation, node, operand):
    if node.broadcast_type == BroadcastType.None_:
        return tile

    tile_shape = operation.tile_shape
    tile_dims = (tile_shape.total_row_dim(), tile_shape.total_col_dim())
    num_faces = tile_shape.total_num_faces()
    broadcast = get_golden_generator(BroadcastGolden)(
        node.broadcast_type,
        tilize_block(
            tile,
            tile_dims,
            operand.data_format,
            num_faces,
            tile_dimensions=tile_dims,
        ),
        operand.data_format,
        num_faces,
        1,
        tile_shape.face_r_dim,
    )
    return untilize_block(
        broadcast,
        operand.data_format,
        tile_dims,
        tile_dimensions=tile_dims,
        num_faces=num_faces,
    ).reshape(tile_dims)


def transpose_tile(tile, config, operation, node):
    if (
        node.transpose_faces == Transpose.No
        and node.transpose_within_face == Transpose.No
    ):
        return tile

    tile_shape = operation.tile_shape
    tile_dims = (tile_shape.total_row_dim(), tile_shape.total_col_dim())
    transpose = get_golden_generator(TransposeGolden)
    if node.transpose_faces == Transpose.Yes:
        tile = transpose.transpose_faces_multi_tile(
            tile,
            config.sentinel.golden_math_format,
            1,
            tilize=True,
            untilize=True,
            input_dimensions=tile_dims,
        )
    if node.transpose_within_face == Transpose.Yes:
        tile = transpose.transpose_within_faces_multi_tile(
            tile,
            config.sentinel.golden_math_format,
            1,
            tilize=True,
            untilize=True,
            input_dimensions=tile_dims,
        )
    return tile.reshape(tile_dims)
