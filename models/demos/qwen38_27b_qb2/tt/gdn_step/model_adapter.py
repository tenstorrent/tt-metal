# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-only bridge from the model's convolution outputs.

The model's opt-in decode path includes all preparation/layout work here.
The existing gated RMSNorm epilogue remains external;
this bridge is not the final fused P1/P2 implementation.
"""

from models.demos.qwen38_27b_qb2.tt.gdn_step import op


def step_from_flat(q, k, v, log_decay, beta, state, output):
    """Consume only token zero of [B,T,H*128] and update [B,HV,128,128].

    Q/K are raw convolution outputs, beta is already sigmoid-transformed, and
    log_decay already contains the softplus/A calculation. output is a caller-owned
    FP32 row-major [B*HV,128] buffer retained for the complete trace lifetime."""
    import ttnn

    batch, time_rows, qwidth = q.shape
    heads, value_heads = qwidth // 128, v.shape[-1] // 128
    if (
        time_rows not in (1, 32)
        or qwidth != heads * 128
        or tuple(k.shape) != tuple(q.shape)
        or tuple(v.shape) != (batch, time_rows, value_heads * 128)
        or tuple(state.shape) != (batch, value_heads, 128, 128)
        or value_heads % heads
        or tuple(log_decay.shape) != (batch, time_rows, value_heads)
        or tuple(beta.shape) != tuple(log_decay.shape)
    ):
        raise ValueError("Unsupported single-token GDN model geometry")

    def vector(tensor, count):
        row = ttnn.to_layout(tensor[:, :1, :], ttnn.ROW_MAJOR_LAYOUT)
        row = ttnn.reshape(row, [batch, count, 128])
        return ttnn.typecast(ttnn.to_layout(row, ttnn.TILE_LAYOUT), ttnn.float32)

    vectors = []
    for tensor in (q, k):
        raw = ttnn.to_layout(vector(tensor, heads), ttnn.ROW_MAJOR_LAYOUT)
        raw = ttnn.repeat_interleave(raw, value_heads // heads, dim=1)
        vectors.append(ttnn.reshape(raw, [batch * value_heads, 128]))
    values = ttnn.reshape(ttnn.to_layout(vector(v, value_heads), ttnn.ROW_MAJOR_LAYOUT), [batch * value_heads, 128])
    decay = ttnn.exp(log_decay[:, :1, :])
    gates = ttnn.concat([decay, ttnn.typecast(beta[:, :1, :], ttnn.float32)], dim=1)
    gates = ttnn.to_layout(ttnn.permute(gates, [0, 2, 1]), ttnn.ROW_MAJOR_LAYOUT)
    gates = ttnn.pad(ttnn.reshape(gates, [batch * value_heads, 2]), [(0, 0), (0, 6)], 0.0)
    flat_state = ttnn.reshape(state, [batch * value_heads, 128, 128])
    if flat_state.buffer_address() != state.buffer_address():
        raise ValueError("GDN state flattening must be a view, not a copy")
    # Real layer projection/gate tensors may reside in L1. The recurrence
    # descriptor intentionally addresses interleaved DRAM inputs, so preserve
    # that contract at this boundary. Already-DRAM operands need no copy.
    prepared = [ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG) for tensor in (*vectors, values, gates)]
    op.step(*prepared, flat_state, output, value_splits=4, input_buffer_items=2, normalize_qk=True)
    head_major = ttnn.to_layout(ttnn.reshape(output, [batch * value_heads, 1, 128]), ttnn.TILE_LAYOUT)
    return ttnn.reshape(head_major, [batch * value_heads, 32, 128], head_major.padded_shape)
