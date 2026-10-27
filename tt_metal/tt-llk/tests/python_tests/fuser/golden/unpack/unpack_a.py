# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from helpers.llk_params import BroadcastType, EltwiseBinaryReuseDestType

from ..transforms import broadcast_tile, transpose_tile


def unpack_a_golden(call, state, node, operation, config):
    for tile_call in call.tiles:
        tile = state.inputs.tile_a(tile_call.in0)
        if node.broadcast_type != BroadcastType.None_:
            tile_a = None
            tile_b = broadcast_tile(tile, operation, node, node.src_a)
        else:
            tile_a = transpose_tile(tile, config, operation, node)
            tile_b = None
        if node.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCA:
            tile_a, tile_b = None, tile_a
        state.source_registers.push(tile_a, tile_b)
