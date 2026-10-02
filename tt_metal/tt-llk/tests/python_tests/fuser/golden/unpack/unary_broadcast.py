# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ..transforms import broadcast_tile


def unpack_unary_broadcast_golden(call, state, node, operation, config):
    tile = state.inputs.tile_a(call.in0)
    state.source_registers.push(None, broadcast_tile(tile, operation, node, node.src_a))
