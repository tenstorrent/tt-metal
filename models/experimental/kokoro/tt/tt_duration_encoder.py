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
from .tt_lstm import TTLSTMParams, build_fused_recurrent_weight, preprocess_tt_lstm_1layer, tt_bilstm_nlc
from .tt_trace_prep import (
    prep_cache_get as _prep_cache_get,
    prep_cache_set as _prep_cache_set,
    trace_weight_prep_enabled as _trace_weight_prep_enabled,
)


def _cast_if_needed(x: ttnn.Tensor, dtype, *, memory_config: ttnn.MemoryConfig) -> tuple[ttnn.Tensor, bool]:
    if x.dtype == dtype:
        return x, False
    out = ttnn.typecast(x, dtype, memory_config=memory_config)
    return out, True


def _upload_style_btl_nlc(
    style_bs: ttnn.Tensor,
    *,
    batch: int,
    seq_len: int,
    sty_dim: int,
    wire_dtype,
    memory_config: ttnn.MemoryConfig,
) -> tuple[ttnn.Tensor, bool]:
    """``[B, style_dim]`` → ``[B, T, sty_dim]`` via host expand (no device ``repeat``)."""
    shape = tuple(int(s) for s in style_bs.shape)
    if len(shape) == 3 and shape[0] == batch and shape[1] == seq_len and shape[2] == sty_dim:
        return _cast_if_needed(style_bs, wire_dtype, memory_config=memory_config)

    # This expand does a host round-trip (to_torch + from_torch) — both illegal inside trace capture.
    # The style vector is a constant per-utterance input, so under trace weight prep the expanded
    # ``[B, T, sty_dim]`` tensor is built once (keyed by the persistent input's identity) and reused;
    # the round-trip then runs only on the first (warmup) call. Returns own=False so the caller keeps
    # the cached tensor alive. Prep off = build-and-free each call (original behaviour).
    key = (id(style_bs), "dur_style_btl", batch, seq_len, sty_dim, str(wire_dtype), str(memory_config))
    if _trace_weight_prep_enabled():
        cached = _prep_cache_get(key)
        if cached is not None:
            return cached, False

    style_cpu = ttnn.to_torch(style_bs).float()
    while style_cpu.dim() > 2:
        style_cpu = style_cpu.squeeze(0)
    if int(style_cpu.shape[0]) != batch or int(style_cpu.shape[-1]) != sty_dim:
        raise ValueError(f"style shape {tuple(style_cpu.shape)} vs batch={batch} sty_dim={sty_dim}")

    expanded = style_cpu.unsqueeze(1).expand(batch, seq_len, sty_dim).contiguous()
    out = ttnn.from_torch(
        expanded,
        dtype=wire_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=style_bs.device(),
        memory_config=memory_config,
    )
    if _trace_weight_prep_enabled():
        _prep_cache_set(key, out)
        return out, False
    return out, True


@dataclass(frozen=True)
class TTDurationEncoderLayerParams:
    """One BiLSTM + following :class:`TTAdaLayerNorm` (reference interleaves them)."""

    lstm_fwd: TTLSTMParams
    lstm_rev: TTLSTMParams
    adaln: TTAdaLayerNormParams
    # Direction-fused recurrent weight (None for unidirectional); halves per-step LSTM ops on
    # the unpadded path. bf16 state -> bit-exact (see ``build_fused_recurrent_weight``).
    lstm_w_h_block: "ttnn.Tensor | None" = None


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
    pending_w_h_block: "ttnn.Tensor | None" = None

    for block in duration_encoder.lstms:
        if isinstance(block, nn.LSTM):
            fwd, rev = preprocess_tt_lstm_1layer(block, device, weights_dtype=weights_dtype)
            assert rev is not None
            pending_fwd, pending_rev = fwd, rev
            pending_w_h_block = build_fused_recurrent_weight(block, device, weights_dtype=weights_dtype)
        else:
            assert pending_fwd is not None and pending_rev is not None
            adaln = preprocess_tt_ada_layer_norm(block, device, weights_dtype=weights_dtype)
            layers.append(
                TTDurationEncoderLayerParams(
                    lstm_fwd=pending_fwd,
                    lstm_rev=pending_rev,
                    adaln=adaln,
                    lstm_w_h_block=pending_w_h_block,
                )
            )
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

        s_btl, owns_s_btl = _upload_style_btl_nlc(
            style_wired,
            batch=b,
            seq_len=t_len,
            sty_dim=p.sty_dim,
            wire_dtype=wire_dtype,
            memory_config=memory_config,
        )
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
                w_h_block=layer.lstm_w_h_block,
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

        if owns_s_btl:
            ttnn.deallocate(s_btl)
        if owns_style:
            ttnn.deallocate(style_wired)
        if owns_mask:
            ttnn.deallocate(keep_wired)
        return x

    __call__ = forward
