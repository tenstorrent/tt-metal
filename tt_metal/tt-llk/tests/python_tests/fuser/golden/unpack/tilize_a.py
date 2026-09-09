# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


def tilize_a_golden(call, state, node, operation, config):
    for tile_call in call.tiles:
        state.source_registers.push(
            state.inputs.view_a.tilized_tile(tile_call.in0), None
        )
