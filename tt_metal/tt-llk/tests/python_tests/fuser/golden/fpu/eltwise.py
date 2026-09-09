# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from helpers.golden_generators import EltwiseBinaryGolden, get_golden_generator
from helpers.llk_params import AccToDest, EltwiseBinaryReuseDestType

from ..state import tile_operation


def _eltwise(call, state, node, operation, config, force_accumulate):
    single = tile_operation(operation)
    dimensions = single.max_output_dimensions
    for tile in call.tiles:
        tensor_a, tensor_b = state.source_registers.pop()
        accumulate = force_accumulate or node.acc_to_dest == AccToDest.Yes
        reuse_dest = node.reuse_dest != EltwiseBinaryReuseDestType.NONE
        tensor_dst = state.dest.get(tile.dest) if accumulate or reuse_dest else None
        if node.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCA:
            tensor_a = tensor_dst
            accumulate = False
        elif node.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCB:
            tensor_b = tensor_dst
            accumulate = False
        if tensor_a is None:
            tensor_a = torch.zeros(dimensions)
        if tensor_b is None:
            tensor_b = torch.zeros(dimensions)

        result = get_golden_generator(EltwiseBinaryGolden)(
            node.fpu.operation,
            tensor_a,
            tensor_b,
            config.sentinel.golden_math_format,
            node.math_fidelity,
            tile_shape=single.tile_shape,
        ).reshape(dimensions)
        if accumulate:
            result = result + tensor_dst
        state.dest.set(tile.dest, result)


def eltwise_golden(call, state, node, operation, config):
    _eltwise(call, state, node, operation, config, force_accumulate=False)


def eltwise_accumulate_golden(call, state, node, operation, config):
    _eltwise(call, state, node, operation, config, force_accumulate=True)
