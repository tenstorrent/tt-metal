# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from helpers.golden_generators import MatmulGolden, get_golden_generator


def matmul_golden(call, state, node, operation, config):
    tensor_a, tensor_b = state.source_registers.pop()
    result = get_golden_generator(MatmulGolden)(
        tensor_a,
        tensor_b,
        config.sentinel.golden_math_format,
        node.math_fidelity,
        input_A_dimensions=tensor_a.shape,
        input_B_dimensions=tensor_b.shape,
        tilize=False,
        input_A_format=node.src_a.data_format,
        input_B_format=node.src_b.data_format,
    ).reshape(tensor_a.shape[0], tensor_b.shape[1])
    a_rows = node.src_a.tile_shape.total_row_dim()
    b_cols = node.src_b.tile_shape.total_col_dim()
    rt = state.dest.block_tiles_y
    ct = state.dest.block_tiles_x
    for row in range(rt):
        for col in range(ct):
            state.dest.set(
                call.dest + row * ct + col,
                result[
                    row * a_rows : (row + 1) * a_rows,
                    col * b_cols : (col + 1) * b_cols,
                ],
            )
