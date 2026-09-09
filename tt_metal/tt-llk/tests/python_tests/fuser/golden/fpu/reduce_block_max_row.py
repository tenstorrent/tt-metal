# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch

from ..state import tile_operation
from .reduce_common import reduce_tile


def reduce_block_max_row_golden(call, state, node, operation, config):
    single = tile_operation(operation)
    dimensions = single.max_output_dimensions
    result = None
    for _ in call.tiles:
        tensor_a, _ = state.source_registers.pop()
        if tensor_a is None:
            tensor_a = torch.zeros(dimensions)
        reduced = reduce_tile(
            tensor_a,
            None,
            config,
            single,
            node,
            block_max=True,
        )
        result = reduced if result is None else torch.maximum(result, reduced)
    state.dest.set(call.dest, result)
