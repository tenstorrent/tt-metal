# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Kokoro TTNN LSTM helpers (TT-prefixed params, ``tt_`` entrypoints).

Host-driven timestep loop using ``ttnn.linear`` and activations (no high-level LSTM op).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

import ttnn


@dataclass(frozen=True)
class TTLSTMParams:
    """One-direction LSTM gate weights: ``x @ W_x + h @ W_h + b`` with ``W_*`` stored for ``transpose_b=True``."""

    w_x: ttnn.Tensor
    w_h: ttnn.Tensor
    b: ttnn.Tensor
    hidden_size: int


def preprocess_tt_lstm_1layer(
    lstm: torch.nn.LSTM,
    mesh_device,
    *,
    weights_dtype=ttnn.bfloat16,
) -> Tuple[TTLSTMParams, Optional[TTLSTMParams]]:
    """
    Convert a 1-layer PyTorch ``nn.LSTM`` into TTNN gate tensors.

    Returns ``(forward_params, reverse_params_or_none)``.
    """
    assert lstm.num_layers == 1, "Only 1-layer LSTM supported in bring-up helper"
    hidden_size = lstm.hidden_size

    def pack(prefix: str) -> TTLSTMParams:
        w_ih = getattr(lstm, f"weight_ih_{prefix}").detach().cpu()
        w_hh = getattr(lstm, f"weight_hh_{prefix}").detach().cpu()
        b_ih = getattr(lstm, f"bias_ih_{prefix}").detach().cpu()
        b_hh = getattr(lstm, f"bias_hh_{prefix}").detach().cpu()
        b = (b_ih + b_hh).reshape(1, 1, 1, -1)

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
        return TTLSTMParams(w_x=w_x, w_h=w_h, b=b_tt, hidden_size=hidden_size)

    fwd = pack("l0")
    rev = pack("l0_reverse") if lstm.bidirectional else None
    return fwd, rev


def _lstm_step(
    x_bt: ttnn.Tensor,
    h: ttnn.Tensor,
    c: ttnn.Tensor,
    params: TTLSTMParams,
    *,
    compute_kernel_config=None,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
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

    gs = tuple(int(s) for s in gates.shape)
    if len(gs) > 2:
        batch_leading = int(np.prod(gs[:-1], dtype=np.int64))
        gates = ttnn.reshape(gates, [batch_leading, gs[-1]], memory_config=memory_config)

    H4 = gates.shape[-1]
    H = params.hidden_size
    assert H4 == 4 * H

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


def _length_valid_mask_b1(
    *,
    batch: int,
    seq_len: int,
    sequence_lengths: Sequence[int],
    device,
    memory_config: ttnn.MemoryConfig,
) -> ttnn.Tensor:
    vm = np.zeros((batch, seq_len, 1), dtype=np.float32)
    for bi, le in enumerate(sequence_lengths):
        le = max(0, min(int(le), seq_len))
        vm[bi, :le, 0] = 1.0
    return ttnn.from_torch(
        torch.from_numpy(vm),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def _blend_state(
    valid_bh: ttnn.Tensor,
    new_state: ttnn.Tensor,
    old_state: ttnn.Tensor,
    *,
    memory_config: ttnn.MemoryConfig,
) -> ttnn.Tensor:
    one_m = ttnn.add(ttnn.multiply(valid_bh, -1.0, memory_config=memory_config), 1.0, memory_config=memory_config)
    return ttnn.add(
        ttnn.multiply(valid_bh, new_state, memory_config=memory_config),
        ttnn.multiply(one_m, old_state, memory_config=memory_config),
        memory_config=memory_config,
    )


def tt_bilstm_nlc(
    *,
    x_nlc: ttnn.Tensor,
    fwd: TTLSTMParams,
    rev: TTLSTMParams,
    compute_kernel_config=None,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    sequence_lengths: Optional[Sequence[int]] = None,
) -> ttnn.Tensor:
    """
    1-layer BiLSTM over sequence for NLC ``[B, L, in]``.

    With ``sequence_lengths``, timesteps ``t >= length[b]`` yield zero output and frozen state
    (``pack_padded_sequence`` semantics). Returns ``[B, L, 2H]``.
    """
    B, L, _ = x_nlc.shape
    H = fwd.hidden_size

    h0 = ttnn.zeros(
        [B, H], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=x_nlc.device(), memory_config=memory_config
    )
    c0 = ttnn.zeros(
        [B, H], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=x_nlc.device(), memory_config=memory_config
    )

    valid_all = None
    if sequence_lengths is not None:
        assert len(sequence_lengths) == B, "sequence_lengths must have one entry per batch row"
        valid_all = _length_valid_mask_b1(
            batch=B, seq_len=L, sequence_lengths=sequence_lengths, device=x_nlc.device(), memory_config=memory_config
        )

    h_f = h0
    c_f = c0
    outs_f = []
    for t in range(L):
        xt = ttnn.slice(x_nlc, [0, t, 0], [B, t + 1, x_nlc.shape[2]], [1, 1, 1])
        h_old, c_old = h_f, c_f
        h_new, c_new = _lstm_step(
            xt, h_f, c_f, fwd, compute_kernel_config=compute_kernel_config, memory_config=memory_config
        )
        if valid_all is not None:
            vt = ttnn.slice(valid_all, [0, t, 0], [B, t + 1, 1], [1, 1, 1])
            vt = ttnn.reshape(vt, [B, 1, 1], memory_config=memory_config)
            vt_h3 = ttnn.repeat(vt, (1, 1, H), memory_config=memory_config)
            vt_h = ttnn.reshape(vt_h3, [B, H], memory_config=memory_config)
            h_f = _blend_state(vt_h, h_new, h_old, memory_config=memory_config)
            c_f = _blend_state(vt_h, c_new, c_old, memory_config=memory_config)
            outs_f.append(ttnn.multiply(vt_h, h_new, memory_config=memory_config))
        else:
            h_f, c_f = h_new, c_new
            outs_f.append(h_f)

    h_b = h0
    c_b = c0
    outs_b_rev = []
    for t in reversed(range(L)):
        xt = ttnn.slice(x_nlc, [0, t, 0], [B, t + 1, x_nlc.shape[2]], [1, 1, 1])
        h_old, c_old = h_b, c_b
        h_new, c_new = _lstm_step(
            xt, h_b, c_b, rev, compute_kernel_config=compute_kernel_config, memory_config=memory_config
        )
        if valid_all is not None:
            vt = ttnn.slice(valid_all, [0, t, 0], [B, t + 1, 1], [1, 1, 1])
            vt = ttnn.reshape(vt, [B, 1, 1], memory_config=memory_config)
            vt_h3 = ttnn.repeat(vt, (1, 1, H), memory_config=memory_config)
            vt_h = ttnn.reshape(vt_h3, [B, H], memory_config=memory_config)
            h_b = _blend_state(vt_h, h_new, h_old, memory_config=memory_config)
            c_b = _blend_state(vt_h, c_new, c_old, memory_config=memory_config)
            outs_b_rev.append(ttnn.multiply(vt_h, h_new, memory_config=memory_config))
        else:
            h_b, c_b = h_new, c_new
            outs_b_rev.append(h_b)

    if valid_all is not None:
        ttnn.deallocate(valid_all)

    outs_b = list(reversed(outs_b_rev))

    hs_f = ttnn.concat([ttnn.reshape(h, [B, 1, H], memory_config=memory_config) for h in outs_f], dim=1)
    hs_b = ttnn.concat([ttnn.reshape(h, [B, 1, H], memory_config=memory_config) for h in outs_b], dim=1)
    return ttnn.concat([hs_f, hs_b], dim=2)
