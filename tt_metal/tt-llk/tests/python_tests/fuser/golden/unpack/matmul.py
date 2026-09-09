# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


def unpack_matmul_golden(call, state, node, operation, config):
    a_rows = node.src_a.tile_shape.total_row_dim()
    b_cols = node.src_b.tile_shape.total_col_dim()
    out_tiles_x = operation.max_output_dimensions[1] // b_cols
    row0, col0 = divmod(call.in0, out_tiles_x)
    rt = state.inputs.block_tiles_y
    ct = state.inputs.block_tiles_x
    tensor_a = state.inputs.view_a.row_major_region(
        slice(row0 * a_rows, (row0 + rt) * a_rows), slice(None)
    )
    tensor_b = state.inputs.view_b.row_major_region(
        slice(None), slice(col0 * b_cols, (col0 + ct) * b_cols)
    )
    state.source_registers.push(tensor_a, tensor_b)
