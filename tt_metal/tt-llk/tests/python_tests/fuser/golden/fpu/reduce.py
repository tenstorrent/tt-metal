# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from helpers.llk_params import ReducePool

from ..state import tile_operation
from .reduce_common import reduce_tile


def reduce_golden(call, state, node, operation, config):
    single = tile_operation(operation)
    dimensions = single.max_output_dimensions
    tensor_a, tensor_b = state.source_registers.pop_operands(dimensions)
    reduced = reduce_tile(tensor_a, tensor_b, config, single, node)
    if call.dest not in state.reduced_tiles:
        state.reduced_tiles.add(call.dest)
        state.dest.set(call.dest, reduced)
    elif node.fpu.reduce_pool == ReducePool.Max:
        state.dest.set(call.dest, torch.maximum(state.dest.get(call.dest), reduced))
    else:
        state.dest.set(call.dest, state.dest.get(call.dest) + reduced)
