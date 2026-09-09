# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from .common import append_tile


def pack_matmul_golden(call, state, pack_node, operation, config):
    out_tiles_x = pack_node.output.tile_count_x
    row0, col0 = divmod(call.out, out_tiles_x)
    rt = state.dest.block_tiles_y
    ct = state.dest.block_tiles_x
    for row in range(rt):
        for col in range(ct):
            append_tile(
                call.dest + row * ct + col,
                (row0 + row) * out_tiles_x + col0 + col,
                state,
                pack_node,
                config,
            )
