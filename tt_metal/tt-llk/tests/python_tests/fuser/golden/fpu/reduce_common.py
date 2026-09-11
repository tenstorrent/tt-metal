# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from helpers.golden_generators import ReduceGolden, get_golden_generator
from helpers.llk_params import ReduceDimension, ReducePool
from helpers.tilize_untilize import tilize_block, untilize_block


def reduce_tile(tensor_a, tensor_b, config, operation, node, block_max=False):
    output_format = config.sentinel.golden_math_format
    tile_shape = operation.tile_shape
    dimensions = operation.max_output_dimensions
    tile_dims = (tile_shape.total_row_dim(), tile_shape.total_col_dim())
    num_faces = tile_shape.total_num_faces()
    reduce_dim = ReduceDimension.Row if block_max else node.fpu.reduce_dim
    pool_type = ReducePool.Max if block_max else node.fpu.reduce_pool
    reduce = get_golden_generator(ReduceGolden)

    def reduce_one(tensor):
        result = reduce(
            tilize_block(
                tensor,
                dimensions,
                output_format,
                num_faces,
                tile_dimensions=tile_dims,
            ).flatten(),
            reduce_dim,
            pool_type,
            output_format,
            tile_cnt=1,
            tile_shape=tile_shape,
        )
        return untilize_block(
            result.flatten(),
            output_format,
            dimensions,
            tile_dimensions=tile_dims,
            num_faces=num_faces,
        ).flatten()

    src_reduced = reduce_one(tensor_a)
    if pool_type == ReducePool.Average:
        span = tile_dims[1] if reduce_dim == ReduceDimension.Row else tile_dims[0]
        result = src_reduced * span * tensor_b.flatten()[0].item()
    else:
        result = src_reduced
    return result.reshape(dimensions).to(src_reduced.dtype)
