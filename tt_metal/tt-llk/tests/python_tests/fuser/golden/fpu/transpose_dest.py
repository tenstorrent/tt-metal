# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ..state import tile_operation
from ..transforms import transpose_tile


def transpose_dest_golden(call, state, node, operation, config):
    single = tile_operation(operation)
    for tile in call.tiles:
        state.source_registers.pop()
        result = transpose_tile(state.dest.get(tile.dest), config, single, node)
        state.dest.set(tile.dest, result)
