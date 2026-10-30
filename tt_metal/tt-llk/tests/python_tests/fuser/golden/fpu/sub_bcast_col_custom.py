# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from helpers.golden_generators import EltwiseBinaryGolden, get_golden_generator

from ..state import tile_dimensions


def sub_bcast_col_custom_golden(call, state, node, operation, config):
    dimensions = tile_dimensions(operation.tile_shape)
    for tile in call.tiles:
        tensor_a, tensor_b = state.source_registers.pop()
        result = get_golden_generator(EltwiseBinaryGolden)(
            node.fpu.operation,
            tensor_a,
            tensor_b,
            config.sentinel.golden_math_format,
            node.math_fidelity,
            tile_shape=operation.tile_shape,
        ).reshape(dimensions)
        state.dest.set(tile.dest, result)
