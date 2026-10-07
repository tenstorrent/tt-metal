# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Experimental device-only bridge from the model's convolution outputs.

This deliberately measures all preparation/layout work before integrating the
recurrence into a model. The existing gated RMSNorm epilogue remains external;
this bridge is not the final fused P1/P2 implementation.
"""

import math

from models.demos.qwen38_27b_qb2.experiments.gdn_step import op


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
    config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def vector(tensor, count):
        row = ttnn.to_layout(tensor[:, :1, :], ttnn.ROW_MAJOR_LAYOUT)
        row = ttnn.reshape(row, [batch, count, 128])
        return ttnn.typecast(ttnn.to_layout(row, ttnn.TILE_LAYOUT), ttnn.float32)

    normalized = []
    for tensor, scale in ((q, 1 / 128), (k, 1 / math.sqrt(128))):
        # RMSNorm * 1/sqrt(D) is L2 normalization. Q has one extra 1/sqrt(D).
        norm = ttnn.rms_norm(vector(tensor, heads), epsilon=1e-6 / 128, compute_kernel_config=config)
        norm = ttnn.mul(norm, scale)
        norm = ttnn.to_layout(norm, ttnn.ROW_MAJOR_LAYOUT)
        norm = ttnn.repeat_interleave(norm, value_heads // heads, dim=1)
        normalized.append(ttnn.reshape(norm, [batch * value_heads, 128]))
    values = ttnn.reshape(ttnn.to_layout(vector(v, value_heads), ttnn.ROW_MAJOR_LAYOUT), [batch * value_heads, 128])
    decay = ttnn.exp(log_decay[:, :1, :])
    gates = ttnn.concat([decay, ttnn.typecast(beta[:, :1, :], ttnn.float32)], dim=1)
    gates = ttnn.to_layout(ttnn.permute(gates, [0, 2, 1]), ttnn.ROW_MAJOR_LAYOUT)
    gates = ttnn.pad(ttnn.reshape(gates, [batch * value_heads, 2]), [(0, 0), (0, 6)], 0.0)
    flat_state = ttnn.reshape(state, [batch * value_heads, 128, 128])
    if flat_state.buffer_address() != state.buffer_address():
        raise ValueError("GDN state flattening must be a view, not a copy")
    op.step(*normalized, values, gates, flat_state, output, value_splits=4, input_buffer_items=2)
    head_major = ttnn.to_layout(ttnn.reshape(output, [batch * value_heads, 1, 128]), ttnn.TILE_LAYOUT)
    return ttnn.reshape(head_major, [batch * value_heads, 32, 128], head_major.padded_shape)
