# SPDX-License-Identifier: Apache-2.0
"""Experimental independent-row packing; leaves singleton and prefill unchanged.

Reuses the qualified packed routing implementation and its precision objects.
This is a component candidate, not a qualified serving implementation.
"""

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import ParallelMoE


def logical_reshape(value, shape):
    # Explicitly remove each singleton lane's tile padding before joining rows.
    return ttnn.to_layout(ttnn.reshape(ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT), shape), ttnn.TILE_LAYOUT)


def packed_batch_decode(model, value):
    batch, sequence = tuple(value.shape)[1:3]
    if sequence != 1 or batch == 1:
        return ParallelMoE.__call__(model, value)
    if not 1 < batch <= 128:
        raise ValueError("Packed routing capacity supports at most 128 independent rows")
    flat = logical_reshape(value, (1, 1, batch, 2560))
    # group_native is NOT legal here: its dispatcher requires N == 1.
    result = model.group_packed(flat)
    result = model.ops.scatter_reduce(result)
    return logical_reshape(result, (1, batch, 1, 640))
