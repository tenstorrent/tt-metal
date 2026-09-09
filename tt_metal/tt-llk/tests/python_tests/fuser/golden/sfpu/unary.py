# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from helpers.golden_generators import UnarySFPUGolden, get_golden_generator


def unary_golden(call, state, node, operation, config):
    tile_count = len(state.dest)
    tile_dims = state.dest.tile_dims
    data_format = config.sentinel.golden_math_format
    tensor = state.dest.tilized(data_format)
    result = get_golden_generator(UnarySFPUGolden)(
        node.sfpu.operation,
        tensor,
        data_format,
        config.dest_acc,
        data_format,
        (tile_count * tile_dims[0], tile_dims[1]),
        node.sfpu.iterations,
        call.dest,
        node.sfpu.fill_const_value,
        skip_tilize=True,
    )
    state.dest.update_from_tilized(result, data_format)
