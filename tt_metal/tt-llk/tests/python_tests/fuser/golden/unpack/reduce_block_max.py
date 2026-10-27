# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


def unpack_reduce_block_max_golden(call, state, node, operation, config):
    for tile_call in call.tiles:
        state.source_registers.push(state.inputs.tile_a(tile_call.in0), None)
