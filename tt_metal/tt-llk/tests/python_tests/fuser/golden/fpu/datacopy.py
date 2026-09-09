# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from helpers.golden_generators import DataCopyGolden, get_golden_generator
from helpers.llk_params import BroadcastType

from ..state import tile_operation


def datacopy_golden(call, state, node, operation, config):
    single = tile_operation(operation)
    dimensions = single.max_output_dimensions
    for tile in call.tiles:
        tensor_a, tensor_b = state.source_registers.pop()
        source = tensor_b if node.broadcast_type != BroadcastType.None_ else tensor_a
        if source is None:
            source = torch.zeros(dimensions)
        result = get_golden_generator(DataCopyGolden)(
            source,
            config.sentinel.golden_math_format,
            num_faces=single.tile_shape.total_num_faces(),
            input_dimensions=dimensions,
            face_r_dim=single.tile_shape.face_r_dim,
            tile_shape=single.tile_shape,
        )
        state.dest.set(tile.dest, result.reshape(dimensions))
