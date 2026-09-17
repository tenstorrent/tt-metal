# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch

from ..state import tile_dimensions
from .reduce_common import reduce_tile


def reduce_block_max_golden(call, state, node, operation, config):
    dimensions = tile_dimensions(operation.tile_shape)
    tensor_a, _ = state.source_registers.pop()
    if tensor_a is None:
        tensor_a = torch.zeros(dimensions)
    reduced = reduce_tile(
        tensor_a,
        None,
        config,
        operation,
        node,
        block_max=True,
    )
    row_base = call.dest // state.dest.block_tiles_x * state.dest.block_tiles_x
    state.dest.set(row_base, torch.maximum(state.dest.get(row_base), reduced))
