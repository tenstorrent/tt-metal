# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from helpers.llk_params import BroadcastType, Transpose

from ..transforms import broadcast_tile, transpose_tile


def unpack_ab_golden(call, state, node, operation, config):
    tile_a = state.inputs.tile_a(call.in0)
    tile_b = state.inputs.tile_b(call.in1)
    if node.broadcast_type != BroadcastType.None_:
        tile_b = broadcast_tile(tile_b, operation, node, node.src_b)
    if node.transpose_faces == Transpose.Yes:
        tile_a = transpose_tile(tile_a, config, operation, node)
    state.source_registers.push(tile_a, tile_b)
