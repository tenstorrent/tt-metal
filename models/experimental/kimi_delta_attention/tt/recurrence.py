# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fully device-resident composed KDA recurrence oracle."""

from __future__ import annotations

import ttnn
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import l2_norm_ttnn


def _token(tensor: ttnn.Tensor, index: int, shape: tuple[int, ...]) -> ttnn.Tensor:
    end = list(tensor.shape)
    start = [0] * len(end)
    start[1] = index
    end[1] = index + 1
    return ttnn.reshape(
        ttnn.slice(tensor, tuple(start), tuple(end), memory_config=ttnn.L1_MEMORY_CONFIG),
        shape,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def composed_kda_recurrence(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    gate: ttnn.Tensor,
    beta: ttnn.Tensor,
    initial_state: ttnn.Tensor,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Evaluate exact KDA recurrence with TTNN ops only.

    Inputs use `[B,T,H,D]`; state uses `[B,H,K,V]`. Arithmetic is promoted
    to FP32 to isolate composed-operation correctness from state-dtype policy.
    """
    batch, sequence, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    expected = {
        "k": (batch, sequence, heads, key_dim),
        "v": (batch, sequence, heads, value_dim),
        "gate": (batch, sequence, heads, key_dim),
        "beta": (batch, sequence, heads),
        "initial_state": (batch, heads, key_dim, value_dim),
    }
    tensors = {"k": k, "v": v, "gate": gate, "beta": beta, "initial_state": initial_state}
    for name, shape in expected.items():
        if tuple(tensors[name].shape) != shape:
            raise ValueError(f"{name} shape {tuple(tensors[name].shape)} != {shape}")

    q = ttnn.typecast(q, ttnn.float32)
    k = ttnn.typecast(k, ttnn.float32)
    v = ttnn.typecast(v, ttnn.float32)
    gate = ttnn.typecast(gate, ttnn.float32)
    beta = ttnn.typecast(beta, ttnn.float32)
    state = ttnn.typecast(initial_state, ttnn.float32)
    state = ttnn.to_memory_config(state, ttnn.DRAM_MEMORY_CONFIG)

    q = l2_norm_ttnn(q, dim=-1)
    q = ttnn.multiply(q, key_dim**-0.5, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    k = l2_norm_ttnn(k, dim=-1)
    outputs = []

    for index in range(sequence):
        q_t = _token(q, index, (batch, heads, key_dim))
        k_t = _token(k, index, (batch, heads, key_dim))
        v_t = _token(v, index, (batch, heads, value_dim))
        gate_t = _token(gate, index, (batch, heads, key_dim))
        beta_t = _token(beta, index, (batch, heads))

        decay = ttnn.exp(gate_t, memory_config=ttnn.L1_MEMORY_CONFIG)
        decay = ttnn.reshape(
            decay,
            (batch, heads, key_dim, 1),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        state = ttnn.multiply(state, decay, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        k_row = ttnn.reshape(
            k_t,
            (batch, heads, 1, key_dim),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        predicted = ttnn.matmul(k_row, state, memory_config=ttnn.L1_MEMORY_CONFIG)
        predicted = ttnn.reshape(
            predicted,
            (batch, heads, value_dim),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        residual = ttnn.subtract(v_t, predicted, memory_config=ttnn.L1_MEMORY_CONFIG)

        k_column = ttnn.reshape(
            k_t,
            (batch, heads, key_dim, 1),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        residual = ttnn.reshape(
            residual,
            (batch, heads, 1, value_dim),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        update = ttnn.matmul(k_column, residual, memory_config=ttnn.L1_MEMORY_CONFIG)
        beta_t = ttnn.reshape(beta_t, (batch, heads, 1, 1), memory_config=ttnn.L1_MEMORY_CONFIG)
        update = ttnn.multiply(update, beta_t, memory_config=ttnn.L1_MEMORY_CONFIG)
        state = ttnn.add(state, update, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        q_row = ttnn.reshape(q_t, (batch, heads, 1, key_dim), memory_config=ttnn.L1_MEMORY_CONFIG)
        output = ttnn.matmul(q_row, state, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        outputs.append(ttnn.reshape(output, (batch, 1, heads, value_dim)))

    if len(outputs) == 1:
        return outputs[0], state
    return ttnn.concat(outputs, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG), state


def chunk_kda_recurrence(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    gate: ttnn.Tensor,
    beta: ttnn.Tensor,
    initial_state: ttnn.Tensor,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Execute chunk-parallel KDA with FP32 recurrent state."""
    key_dim = gate.shape[-1]
    if len(q.shape) == 4:
        q = l2_norm_ttnn(q, dim=-1)
        k = l2_norm_ttnn(k, dim=-1)
    output, final_state = ttnn.transformer.chunk_kda(
        q,
        k,
        v,
        gate,
        beta,
        scale=key_dim**-0.5,
        initial_state=initial_state,
        output_final_state=True,
        output_head_major=len(q.shape) == 3,
        chunk_size=32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    assert final_state is not None
    if len(q.shape) == 4:
        output = ttnn.to_layout(output, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return output, final_state


def fused_kda_recurrence(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    gate: ttnn.Tensor,
    beta: ttnn.Tensor,
    initial_state: ttnn.Tensor,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Execute the fused T=1 recurrence after device-side preprocessing."""
    batch, sequence, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if sequence != 1:
        raise ValueError(f"fused KDA recurrence requires T=1, got T={sequence}")

    q = ttnn.typecast(q, ttnn.float32)
    k = ttnn.typecast(k, ttnn.float32)
    v = ttnn.typecast(v, ttnn.float32)
    gate = ttnn.typecast(gate, ttnn.float32)
    beta = ttnn.typecast(beta, ttnn.float32)
    state = initial_state if initial_state.dtype == ttnn.float32 else ttnn.typecast(initial_state, ttnn.float32)

    q = ttnn.multiply(
        l2_norm_ttnn(q, dim=-1),
        key_dim**-0.5,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    k = l2_norm_ttnn(k, dim=-1)
    q = ttnn.reshape(ttnn.permute(q, (0, 2, 1, 3)), (batch * heads, 1, key_dim))
    k = ttnn.reshape(ttnn.permute(k, (0, 2, 1, 3)), (batch * heads, 1, key_dim))
    v = ttnn.reshape(ttnn.permute(v, (0, 2, 1, 3)), (batch * heads, 1, value_dim))
    gate = ttnn.reshape(
        ttnn.permute(gate, (0, 2, 1, 3)),
        (batch * heads, 1, key_dim),
    )
    beta = ttnn.reshape(ttnn.permute(beta, (0, 2, 1)), (batch * heads, 1, 1))
    decay = ttnn.exp(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    decay = ttnn.transpose(decay, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    state = ttnn.reshape(state, (batch * heads, key_dim, value_dim))
    q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
    k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
    v = ttnn.to_memory_config(v, ttnn.DRAM_MEMORY_CONFIG)
    beta = ttnn.to_memory_config(beta, ttnn.DRAM_MEMORY_CONFIG)
    state = ttnn.to_memory_config(state, ttnn.DRAM_MEMORY_CONFIG)

    output, final_state = ttnn.transformer.kda_recurrent_step(
        q,
        k,
        v,
        decay,
        beta,
        state,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return (
        ttnn.permute(
            ttnn.reshape(output, (batch, heads, 1, value_dim)),
            (0, 2, 1, 3),
        ),
        ttnn.reshape(final_state, (batch, heads, key_dim, value_dim)),
    )
