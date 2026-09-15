# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from helpers.chip_architecture import ChipArchitecture
from helpers.golden_generators import BinarySFPUGolden, get_golden_generator


def binary_golden(call, state, node, operation, config):
    tile_count = len(state.dest)
    tile_dims = state.dest.tile_dims
    data_format = config.sentinel.golden_math_format
    tensor = state.dest.tilized(data_format)
    iterations = node.sfpu.iterations
    if config.architecture in (ChipArchitecture.BLACKHOLE, ChipArchitecture.WORMHOLE):
        iterations = tile_dims[0] * tile_dims[1] // 32
    result = get_golden_generator(BinarySFPUGolden)(
        node.sfpu.operation,
        tensor,
        call.src0,
        call.src1,
        call.dest,
        iterations,
        (tile_count * tile_dims[0], tile_dims[1]),
        data_format,
        skip_tilize=True,
        tile_dimensions=tile_dims,
    )
    state.dest.update_from_tilized(result, data_format)
