# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


def reduce_tilize_a_golden(call, state, node, operation, config):
    tile = state.inputs.view_a.strided_tile(call.in0)
    state.source_registers.push(tile, state.inputs.tile_b(call.in1))
