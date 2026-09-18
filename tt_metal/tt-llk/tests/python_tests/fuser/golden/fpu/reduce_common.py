# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from helpers.golden_generators import (
    ReduceGapoolGolden,
    ReduceGolden,
    get_golden_generator,
)
from helpers.llk_params import ReduceDimension, ReducePool
from helpers.tilize_untilize import tilize_block, untilize_block

from ..state import tile_dimensions


def reduce_tile(tensor_a, tensor_b, config, operation, node, block_max=False):
    output_format = config.sentinel.golden_math_format
    tile_shape = operation.tile_shape
    tile_dims = tile_dimensions(tile_shape)
    num_faces = tile_shape.total_num_faces()
    reduce_dim = ReduceDimension.Row if block_max else node.fpu.reduce_dim
    pool_type = ReducePool.Max if block_max else node.fpu.reduce_pool
    src_a = tilize_block(
        tensor_a,
        tile_dims,
        output_format,
        num_faces,
        tile_dimensions=tile_dims,
    ).flatten()

    if pool_type == ReducePool.Max:
        result = get_golden_generator(ReduceGolden)(
            src_a,
            reduce_dim,
            pool_type,
            output_format,
            tile_cnt=1,
            tile_shape=tile_shape,
        )
    else:
        src_b = tilize_block(
            tensor_b,
            tile_dims,
            output_format,
            num_faces,
            tile_dimensions=tile_dims,
        ).flatten()
        result = get_golden_generator(ReduceGapoolGolden)(
            src_a,
            src_b,
            output_format,
            reduce_dim,
            math_fidelity=node.math_fidelity,
            tile_cnt=1,
            tile_shape=tile_shape,
            input_format=node.src_a.data_format,
            dest_acc=config.dest_acc,
        )

    return untilize_block(
        result.flatten(),
        output_format,
        tile_dims,
        tile_dimensions=tile_dims,
        num_faces=num_faces,
    ).reshape(tile_dims)
