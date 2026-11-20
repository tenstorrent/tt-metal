# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Stateless FP32 gated-delta recurrence for the MOSS Qwen3.5 prompt encoder.

The previous chunk solver performed FP32 matrix arithmetic with different
precision from the native CUDA scan. An explicit token recurrence keeps the
state, reductions and updates in FP32 SFPU operations. This favors accuracy for
batch-one prompt encoding; it is not a decode/KV-cache implementation.
"""

import ttnn


def gated_delta_rule_recurrent(query, key, value, beta, decay, *, device, valid_length):
    """Return padded [1,T,32,128] output and the final FP32 state.

    Query/key have 16 heads, expanded to the value's 32 heads after FP32 L2
    normalization. Only real prompt tokens update state. The caller owns the
    device and all input/output tensors; no host activation computation occurs.
    """
    batch, tokens, query_heads, key_width = query.shape
    if (batch, query_heads, key_width) != (1, 16, 128):
        raise ValueError("MOSS prompt recurrence requires query/key [1,T,16,128]")
    if tuple(key.shape) != tuple(query.shape) or tuple(value.shape) != (1, tokens, 32, 128):
        raise ValueError("MOSS prompt recurrence has inconsistent key/value shapes")
    if tuple(beta.shape) != (1, tokens, 32) or tuple(decay.shape) != (1, tokens, 32):
        raise ValueError("MOSS prompt recurrence requires beta/decay [1,T,32]")
    if not isinstance(valid_length, int) or not 1 <= valid_length <= min(tokens, 4096):
        raise ValueError("MOSS prompt recurrence requires 1..4096 real tokens")
    if device.get_num_devices() != 1:
        raise ValueError("Prompt recurrence runs on one explicitly owned device")

    def normalize(tensor):
        tensor = ttnn.typecast(tensor, ttnn.float32)
        squared_norm = ttnn.sum(ttnn.multiply(tensor, tensor), dim=-1, keepdim=True)
        return ttnn.multiply(tensor, ttnn.rsqrt(ttnn.add(squared_norm, 1e-6)))

    query = ttnn.repeat_interleave(normalize(query), 2, dim=2)
    key = ttnn.repeat_interleave(normalize(key), 2, dim=2)
    value, beta, decay = (ttnn.typecast(tensor, ttnn.float32) for tensor in (value, beta, decay))
    state = ttnn.zeros(
        (1, 32, 128, 128),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    outputs, segments = [], []
    for position in range(valid_length):
        q = ttnn.permute(query[:, position : position + 1], (0, 2, 1, 3))
        k = ttnn.permute(key[:, position : position + 1], (0, 2, 3, 1))
        v = ttnn.permute(value[:, position : position + 1], (0, 2, 1, 3))
        rate = ttnn.reshape(beta[:, position : position + 1], (1, 32, 1, 1))
        log_decay = ttnn.reshape(decay[:, position : position + 1], (1, 32, 1, 1))
        state = ttnn.multiply(state, ttnn.exp(log_decay))
        prediction = ttnn.sum(ttnn.multiply(k, state), dim=-2, keepdim=True)
        delta = ttnn.multiply(ttnn.subtract(v, prediction), rate)
        state = ttnn.add(state, ttnn.multiply(k, delta))
        scaled_query = ttnn.permute(ttnn.multiply(q, 128**-0.5), (0, 1, 3, 2))
        output = ttnn.sum(ttnn.multiply(scaled_query, state), dim=-2, keepdim=True)
        outputs.append(ttnn.permute(output, (0, 2, 1, 3)))
        if len(outputs) == 128:
            segments.append(outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=1))
            outputs.clear()
    if outputs:
        segments.append(outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=1))
    if tokens > valid_length:
        segments.append(
            ttnn.zeros(
                (1, tokens - valid_length, 32, 128),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
    return (segments[0] if len(segments) == 1 else ttnn.concat(segments, dim=1)), state
