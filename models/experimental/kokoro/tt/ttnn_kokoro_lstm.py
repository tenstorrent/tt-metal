# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Kokoro TTNN LSTM helpers.

TTNN does not currently provide a high-level `lstm` op, so we implement a
host-driven timestep loop using TTNN matmuls/activations.

This is intended for correctness/PCC bring-up first; performance tuning comes later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

import ttnn


@dataclass(frozen=True)
class LSTMParams:
    # Combined weights for efficiency:
    # gates = x @ W_x + h @ W_h + b, where W_x: [in, 4H], W_h: [H, 4H], b: [4H]
    w_x: ttnn.Tensor
    w_h: ttnn.Tensor
    b: ttnn.Tensor
    hidden_size: int


def preprocess_pytorch_lstm_1layer(
    lstm: torch.nn.LSTM,
    mesh_device,
    *,
    weights_dtype=ttnn.bfloat16,
) -> Tuple[LSTMParams, Optional[LSTMParams]]:
    """
    Convert a 1-layer PyTorch LSTM into TTNN gate matrices.

    Returns (forward_params, reverse_params_or_none).
    """
    assert lstm.num_layers == 1, "Only 1-layer LSTM supported in bring-up helper"
    hidden_size = lstm.hidden_size

    def pack(prefix: str) -> LSTMParams:
        w_ih = getattr(lstm, f"weight_ih_{prefix}").detach().cpu()  # [4H, in]
        w_hh = getattr(lstm, f"weight_hh_{prefix}").detach().cpu()  # [4H, H]
        b_ih = getattr(lstm, f"bias_ih_{prefix}").detach().cpu()
        b_hh = getattr(lstm, f"bias_hh_{prefix}").detach().cpu()
        b = (b_ih + b_hh).reshape(1, 1, 1, -1)

        # Store transposed to use ttnn.linear(..., transpose_b=True)
        w_x = ttnn.from_torch(
            w_ih,
            dtype=weights_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        w_h = ttnn.from_torch(
            w_hh,
            dtype=weights_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        b_tt = ttnn.from_torch(
            b, dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        return LSTMParams(w_x=w_x, w_h=w_h, b=b_tt, hidden_size=hidden_size)

    fwd = pack("l0")
    rev = pack("l0_reverse") if lstm.bidirectional else None
    return fwd, rev


def _lstm_step(
    x_bt: ttnn.Tensor,  # [B, 1, in] or [B, in]
    h: ttnn.Tensor,  # [B, H]
    c: ttnn.Tensor,  # [B, H]
    params: LSTMParams,
    *,
    compute_kernel_config=None,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    # Ensure 2D [B, in]
    if len(x_bt.shape) == 3 and x_bt.shape[1] == 1:
        x_bt = ttnn.reshape(x_bt, [x_bt.shape[0], x_bt.shape[2]], memory_config=memory_config)

    gates_x = ttnn.linear(
        x_bt,
        params.w_x,
        bias=None,
        transpose_b=True,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
    )
    gates_h = ttnn.linear(
        h,
        params.w_h,
        bias=None,
        transpose_b=True,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
    )
    gates = ttnn.add(gates_x, gates_h, memory_config=memory_config)
    gates = ttnn.add(gates, params.b, memory_config=memory_config)

    # Normalize gates to [B, 4H] for slicing
    if len(gates.shape) == 4 and gates.shape[1] == 1 and gates.shape[2] == 1:
        gates = ttnn.reshape(gates, [gates.shape[0], gates.shape[3]], memory_config=memory_config)

    H4 = gates.shape[-1]
    H = params.hidden_size
    assert H4 == 4 * H

    # Slice gates: i,f,g,o
    i = ttnn.slice(gates, [0, 0], [gates.shape[0], H], [1, 1])
    f = ttnn.slice(gates, [0, H], [gates.shape[0], 2 * H], [1, 1])
    g = ttnn.slice(gates, [0, 2 * H], [gates.shape[0], 3 * H], [1, 1])
    o = ttnn.slice(gates, [0, 3 * H], [gates.shape[0], 4 * H], [1, 1])

    i = ttnn.sigmoid(i, memory_config=memory_config)
    f = ttnn.sigmoid(f, memory_config=memory_config)
    g = ttnn.tanh(g, memory_config=memory_config)
    o = ttnn.sigmoid(o, memory_config=memory_config)

    c_new = ttnn.add(
        ttnn.multiply(f, c, memory_config=memory_config),
        ttnn.multiply(i, g, memory_config=memory_config),
        memory_config=memory_config,
    )
    h_new = ttnn.multiply(o, ttnn.tanh(c_new, memory_config=memory_config), memory_config=memory_config)
    return h_new, c_new


def bilstm_nlc(
    *,
    x_nlc: ttnn.Tensor,  # [B, L, in]
    fwd: LSTMParams,
    rev: LSTMParams,
    compute_kernel_config=None,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """
    1-layer BiLSTM over sequence dimension for NLC activations.

    Returns NLC: [B, L, 2H]
    """
    B, L, _ = x_nlc.shape
    H = fwd.hidden_size

    h0 = ttnn.zeros(
        [B, H], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=x_nlc.device(), memory_config=memory_config
    )
    c0 = ttnn.zeros(
        [B, H], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=x_nlc.device(), memory_config=memory_config
    )

    # Forward pass
    h_f = h0
    c_f = c0
    outs_f = []
    for t in range(L):
        xt = ttnn.slice(x_nlc, [0, t, 0], [B, t + 1, x_nlc.shape[2]], [1, 1, 1])
        h_f, c_f = _lstm_step(
            xt, h_f, c_f, fwd, compute_kernel_config=compute_kernel_config, memory_config=memory_config
        )
        outs_f.append(h_f)

    # Reverse pass
    h_b = h0
    c_b = c0
    outs_b = []
    for t in reversed(range(L)):
        xt = ttnn.slice(x_nlc, [0, t, 0], [B, t + 1, x_nlc.shape[2]], [1, 1, 1])
        h_b, c_b = _lstm_step(
            xt, h_b, c_b, rev, compute_kernel_config=compute_kernel_config, memory_config=memory_config
        )
        outs_b.append(h_b)
    outs_b = list(reversed(outs_b))

    # Stack and concat
    hs_f = ttnn.concat([ttnn.reshape(h, [B, 1, H], memory_config=memory_config) for h in outs_f], dim=1)
    hs_b = ttnn.concat([ttnn.reshape(h, [B, 1, H], memory_config=memory_config) for h in outs_b], dim=1)
    return ttnn.concat([hs_f, hs_b], dim=2)
