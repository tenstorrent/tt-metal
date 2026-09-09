# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


def reduce_tilize_a_golden(call, state, node, operation, config):
    src = node.src_a
    rows = src.tile_shape.total_row_dim()
    cols = src.tile_shape.total_col_dim()
    tile_y, tile_x = divmod(call.in0, src.tile_count_x)
    tile = state.inputs.view_a.tilized_region(
        slice(tile_y * rows, (tile_y + 1) * rows),
        slice(tile_x * cols, (tile_x + 1) * cols),
    )
    state.source_registers.push(tile, state.inputs.tile_b(call.in1))
