# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


def unpack_golden(call, state, node, operation, config):
    state.source_registers.push(
        state.inputs.tile_a(call.in0),
        state.inputs.tile_b(call.in1),
    )
