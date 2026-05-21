# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of Kokoro :class:`~models.experimental.kokoro.reference.modules.DurationEncoder`.

Tensor layout:

* ``d_en_bct``: ``[B, d_model, T]`` (matches reference input before the internal permutes).
* ``style_bs``: ``[B, style_dim]``.
* ``sequence_lengths``: length ``B``; valid timesteps ``0 .. length[b]-1`` (same semantics as
  ``torch.nn.utils.rnn.pack_padded_sequence``).
* ``keep_mask_btl``: ``[B, T, 1]``, multiply mask with ``1`` at real tokens and ``0`` at padding
  (equivalent to ``~text_mask`` as ``float`` in the reference when ``text_mask`` marks padding).

All forward math runs on device; PyTorch appears only in :func:`preprocess_tt_duration_encoder`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch.nn as nn

import ttnn

from .tt_ada_layer_norm import TTAdaLayerNorm, TTAdaLayerNormParams, preprocess_tt_ada_layer_norm
from .tt_lstm import TTLSTMParams, preprocess_tt_lstm_1layer, tt_bilstm_nlc


def _cast_if_needed(x: ttnn.Tensor, dtype, *, memory_config: ttnn.MemoryConfig) -> tuple[ttnn.Tensor, bool]:
    if x.dtype == dtype:
        return x, False
    out = ttnn.typecast(x, dtype, memory_config=memory_config)
    return out, True


@dataclass(frozen=True)
class TTDurationEncoderLayerParams:
    """One BiLSTM + following :class:`TTAdaLayerNorm` (reference interleaves them)."""

    lstm_fwd: TTLSTMParams
    lstm_rev: TTLSTMParams
    adaln: TTAdaLayerNormParams


@dataclass(frozen=True)
class TTDurationEncoderParams:
    layers: tuple[TTDurationEncoderLayerParams, ...]
    d_model: int
    sty_dim: int


def preprocess_tt_duration_encoder(
    duration_encoder: nn.Module,
    device,
    *,
    weights_dtype=ttnn.bfloat16,
) -> TTDurationEncoderParams:
    """Upload PyTorch ``DurationEncoder`` weights for :class:`TTDurationEncoder`."""
    layers: list[TTDurationEncoderLayerParams] = []
    pending_fwd: TTLSTMParams | None = None
    pending_rev: TTLSTMParams | None = None

    for block in duration_encoder.lstms:
        if isinstance(block, nn.LSTM):
            fwd, rev = preprocess_tt_lstm_1layer(block, device, weights_dtype=weights_dtype)
            assert rev is not None
            pending_fwd, pending_rev = fwd, rev
        else:
            assert pending_fwd is not None and pending_rev is not None
            adaln = preprocess_tt_ada_layer_norm(block, device, weights_dtype=weights_dtype)
            layers.append(TTDurationEncoderLayerParams(lstm_fwd=pending_fwd, lstm_rev=pending_rev, adaln=adaln))
            pending_fwd, pending_rev = None, None

    assert pending_fwd is None and pending_rev is None, "DurationEncoder lstms must end with AdaLayerNorm"

    return TTDurationEncoderParams(
        layers=tuple(layers),
        d_model=int(duration_encoder.d_model),
        sty_dim=int(duration_encoder.sty_dim),
    )


class TTDurationEncoder:
    """Alternating packed-length BiLSTM and adaptive LayerNorm (reference ``DurationEncoder``)."""

    __slots__ = ("_adalns", "params")

    def __init__(self, params: TTDurationEncoderParams) -> None:
        self.params = params
        self._adalns = tuple(TTAdaLayerNorm(ly.adaln) for ly in params.layers)

    def forward(
        self,
        d_en_bct: ttnn.Tensor,
        style_bs: ttnn.Tensor,
        *,
        sequence_lengths: Sequence[int],
        keep_mask_btl: ttnn.Tensor,
        compute_kernel_config,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
        wire_dtype=None,
    ) -> ttnn.Tensor:
        """
        Returns:
            ``[B, T, d_model + style_dim]`` (same layout as reference after the final transpose).
        """
        p = self.params
        x = ttnn.permute(d_en_bct, (0, 2, 1), memory_config=memory_config)
        b, t_len, c_in = x.shape
        if c_in != p.d_model:
            raise ValueError(f"d_en channel dim {c_in} != d_model {p.d_model}")

        if wire_dtype is None:
            wire_dtype = x.dtype

        x, owns_x = _cast_if_needed(x, wire_dtype, memory_config=memory_config)
        style_wired, owns_style = _cast_if_needed(style_bs, wire_dtype, memory_config=memory_config)
        keep_wired, owns_mask = _cast_if_needed(keep_mask_btl, wire_dtype, memory_config=memory_config)

        b_style = int(style_wired.shape[0])
        if b != b_style:
            raise ValueError(f"batch mismatch: d_en {b} vs style {b_style}")

        s_bc = ttnn.reshape(style_wired, [b, 1, p.sty_dim], memory_config=memory_config)
        s_btl = ttnn.repeat(s_bc, (1, t_len, 1), memory_config=memory_config)
        # Keep s_bc alive: on some backends reshape can alias style_bs storage.
        x_cat = ttnn.concat([x, s_btl], dim=2, memory_config=memory_config)
        if owns_x:
            ttnn.deallocate(x)
        x = ttnn.multiply(x_cat, keep_wired, memory_config=memory_config)
        ttnn.deallocate(x_cat)

        lengths_list = [int(n) for n in sequence_lengths]

        for idx, layer in enumerate(p.layers):
            x_in = x
            x_lstm = tt_bilstm_nlc(
                x_nlc=x_in,
                fwd=layer.lstm_fwd,
                rev=layer.lstm_rev,
                compute_kernel_config=compute_kernel_config,
                memory_config=memory_config,
                sequence_lengths=lengths_list,
            )
            ttnn.deallocate(x_in)
            x_lstm, owns_cast = _cast_if_needed(x_lstm, wire_dtype, memory_config=memory_config)
            # Slice can alias ``x_lstm``; do not deallocate ``x_lstm`` until after AdaLN.
            x_d = ttnn.slice(x_lstm, [0, 0, 0], [b, t_len, p.d_model], [1, 1, 1], memory_config=memory_config)
            x_d = self._adalns[idx].forward(
                x_d,
                style_wired,
                compute_kernel_config=compute_kernel_config,
                memory_config=memory_config,
            )
            ttnn.deallocate(x_lstm)
            x_d, owns_d = _cast_if_needed(x_d, wire_dtype, memory_config=memory_config)
            x_cat = ttnn.concat([x_d, s_btl], dim=2, memory_config=memory_config)
            if owns_d:
                ttnn.deallocate(x_d)
            x = ttnn.multiply(x_cat, keep_wired, memory_config=memory_config)
            ttnn.deallocate(x_cat)

        ttnn.deallocate(s_btl)
        if owns_style:
            ttnn.deallocate(style_wired)
        if owns_mask:
            ttnn.deallocate(keep_wired)
        return x

    __call__ = forward
