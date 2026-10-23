# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ..transforms import broadcast_tile


def unpack_sub_bcast_col_custom_golden(call, state, node, operation, config):
    tile_b = broadcast_tile(state.inputs.tile_b(call.in1), operation, node, node.src_b)
    for tile_call in call.tiles:
        state.source_registers.push(state.inputs.tile_a(tile_call.in0), tile_b)
