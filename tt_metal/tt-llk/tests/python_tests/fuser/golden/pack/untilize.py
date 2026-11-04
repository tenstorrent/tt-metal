# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from .common import prepare_tile


def untilize_golden(call, state, pack_node, operation, config):
    output = pack_node.output
    tile_rows = output.tile_shape.total_row_dim()
    tile_cols = output.tile_shape.total_col_dim()
    for tile_call in call.tiles:
        tile = prepare_tile(tile_call.dest, state, pack_node, operation, config)
        row, col = divmod(tile_call.out, output.tile_count_x)
        for local_row in range(tile_rows):
            offset = (row * tile_rows + local_row) * output.dimensions[
                1
            ] + col * tile_cols
            state.output[offset] = [tile[local_row]]
