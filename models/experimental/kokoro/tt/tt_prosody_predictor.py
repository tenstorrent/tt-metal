# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of Kokoro :class:`~models.experimental.kokoro.reference.modules.ProsodyPredictor`.

The reference shuffles between ``BCT`` and ``NLC`` repeatedly. This port keeps every
intermediate in **NLC** ``[B, T, C]`` and only takes ``texts`` in ``BCT`` (to match the upstream
``KModel`` pipeline which produces it that way).

Tensor layout notes
-------------------
* ``texts_bct`` — ``[B, d_hid, T]`` (matches reference; permuted to NLC internally for
  :class:`TTDurationEncoder`).
* ``style_bs`` — ``[B, style_dim]`` (same as :class:`TTAdaLayerNorm`, :class:`TTAdaIN1d`).
* ``text_lengths`` — host ``LongTensor[B]`` (used for both BiLSTM packed-padded semantics and the
  ``keep_mask``).
* ``alignment_btTa`` — ``[B, T, T_aligned]``: the duration-expanded alignment matrix.
* ``text_mask_bt`` — ``[B, T]`` ``bool``, ``True`` where padded.
* ``en_nlc`` (returned) — ``[B, T_aligned, d_hid + style_dim]`` (reference returns
  ``[B, d_hid + style_dim, T_aligned]``; we drop the trailing transpose).
* :meth:`F0Ntrain` consumes ``en_nlc`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn

import ttnn

from .tt_adain_resblk_1d import TTAdainResBlk1d, TTAdainResBlk1dParams, preprocess_tt_adain_resblk_1d
from .tt_conv import TTConv1dParams, tt_conv1d_nlc
from .tt_duration_encoder import (
    TTDurationEncoder,
    TTDurationEncoderParams,
    preprocess_tt_duration_encoder,
)
from .tt_linear_norm import TTLinearNorm, TTLinearNormParams, preprocess_tt_linear_norm
from .tt_lstm import TTLSTMParams, preprocess_tt_lstm_1layer, tt_bilstm_nlc


def _to_fp32_if_needed(x: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> tuple[ttnn.Tensor, bool]:
    if x.dtype == ttnn.float32:
        return x, False
    out = ttnn.typecast(x, ttnn.float32, memory_config=memory_config)
    return out, True


@dataclass(frozen=True)
class TTProsodyPredictorParams:
    """Device-resident weights for :class:`TTProsodyPredictor`."""

    text_encoder: TTDurationEncoderParams
    lstm_fwd: TTLSTMParams
    lstm_rev: TTLSTMParams
    duration_proj: TTLinearNormParams
    shared_fwd: TTLSTMParams
    shared_rev: TTLSTMParams
    f0_blocks: tuple[TTAdainResBlk1dParams, ...]
    n_blocks: tuple[TTAdainResBlk1dParams, ...]
    f0_proj: TTConv1dParams
    n_proj: TTConv1dParams
    d_hid: int
    style_dim: int


def _conv1d_no_weightnorm_to_tt_params(conv: nn.Conv1d, *, weights_dtype) -> TTConv1dParams:
    """Upload a plain ``nn.Conv1d`` (no weight_norm) to TT in the layout :func:`tt_conv1d_nlc` expects."""
    w = conv.weight.detach().cpu().unsqueeze(-1)
    w_tt = ttnn.from_torch(
        w,
        dtype=weights_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    b_tt: Optional[ttnn.Tensor] = None
    if conv.bias is not None:
        b_tt = ttnn.from_torch(
            conv.bias.detach().cpu().reshape(1, 1, 1, -1),
            dtype=weights_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
    return TTConv1dParams(
        weight=w_tt,
        bias=b_tt,
        in_channels=int(conv.in_channels),
        out_channels=int(conv.out_channels),
        kernel_size=int(conv.kernel_size[0]),
        stride=int(conv.stride[0]),
        padding=int(conv.padding[0]),
        groups=int(conv.groups),
    )


def preprocess_tt_prosody_predictor(
    module: nn.Module,
    device: ttnn.Device,
    *,
    weights_dtype=ttnn.bfloat16,
    conv_weights_dtype=ttnn.float32,
) -> TTProsodyPredictorParams:
    """Upload a reference ``ProsodyPredictor`` to device.

    AdaIN MLPs and BiLSTM gates use ``weights_dtype`` (default bf16). Conv weights stay at
    ``conv_weights_dtype`` (default fp32) to keep ``ttnn.conv1d`` PCC vs the torch reference
    above 0.99 — bf16 conv weights typically land near ~0.9 vs reference (same trade-off as
    in :func:`preprocess_tt_adain_resblk_1d`).
    """
    text_encoder = preprocess_tt_duration_encoder(module.text_encoder, device, weights_dtype=weights_dtype)

    lstm_fwd, lstm_rev = preprocess_tt_lstm_1layer(module.lstm, device, weights_dtype=weights_dtype)
    assert lstm_rev is not None, "ProsodyPredictor.lstm must be bidirectional"

    duration_proj = preprocess_tt_linear_norm(module.duration_proj, device, weights_dtype=weights_dtype)

    shared_fwd, shared_rev = preprocess_tt_lstm_1layer(module.shared, device, weights_dtype=weights_dtype)
    assert shared_rev is not None, "ProsodyPredictor.shared must be bidirectional"

    f0_blocks = tuple(
        preprocess_tt_adain_resblk_1d(b, device, weights_dtype=weights_dtype, conv_weights_dtype=conv_weights_dtype)
        for b in module.F0
    )
    n_blocks = tuple(
        preprocess_tt_adain_resblk_1d(b, device, weights_dtype=weights_dtype, conv_weights_dtype=conv_weights_dtype)
        for b in module.N
    )

    f0_proj = _conv1d_no_weightnorm_to_tt_params(module.F0_proj, weights_dtype=conv_weights_dtype)
    n_proj = _conv1d_no_weightnorm_to_tt_params(module.N_proj, weights_dtype=conv_weights_dtype)

    d_hid = int(module.text_encoder.d_model)
    style_dim = int(module.text_encoder.sty_dim)

    return TTProsodyPredictorParams(
        text_encoder=text_encoder,
        lstm_fwd=lstm_fwd,
        lstm_rev=lstm_rev,
        duration_proj=duration_proj,
        shared_fwd=shared_fwd,
        shared_rev=shared_rev,
        f0_blocks=f0_blocks,
        n_blocks=n_blocks,
        f0_proj=f0_proj,
        n_proj=n_proj,
        d_hid=d_hid,
        style_dim=style_dim,
    )


def _keep_mask_btl(text_mask: torch.Tensor, *, device: ttnn.Device) -> ttnn.Tensor:
    """``text_mask[b,t] == True`` is padded; returned tensor is ``1`` on real tokens, ``0`` elsewhere."""
    keep = (~text_mask).to(torch.float32).unsqueeze(-1)
    return ttnn.from_torch(
        keep,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


class TTProsodyPredictor:
    """ProsodyPredictor (text encoder + BiLSTM + duration head + F0/N branches) on TT."""

    def __init__(self, device: ttnn.Device, params: TTProsodyPredictorParams) -> None:
        self.device = device
        self.params = params
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self._text_encoder = TTDurationEncoder(params.text_encoder)
        self._duration_proj = TTLinearNorm(params.duration_proj)
        self._f0_blocks = tuple(TTAdainResBlk1d(device, bp) for bp in params.f0_blocks)
        self._n_blocks = tuple(TTAdainResBlk1d(device, bp) for bp in params.n_blocks)

    def forward(
        self,
        texts_bct: ttnn.Tensor,
        style_bs: ttnn.Tensor,
        text_lengths: torch.LongTensor,
        alignment_btTa: ttnn.Tensor,
        text_mask_bt: torch.Tensor,
        *,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Returns:
            ``duration`` ``[B, T, max_dur]`` (NLC) and ``en_nlc`` ``[B, T_aligned, d_hid + style_dim]``.

            The reference returns ``en`` as ``[B, d_hid + style_dim, T_aligned]`` (BCT); this port
            keeps it in NLC since :meth:`F0Ntrain` accepts NLC directly.
        """
        ck = self.compute_kernel_config

        keep_mask = _keep_mask_btl(text_mask_bt, device=self.device)
        d_nlc = self._text_encoder.forward(
            d_en_bct=texts_bct,
            style_bs=style_bs,
            sequence_lengths=[int(n) for n in text_lengths.detach().cpu().tolist()],
            keep_mask_btl=keep_mask,
            compute_kernel_config=ck,
            memory_config=memory_config,
        )
        ttnn.deallocate(keep_mask)
        # d_nlc: [B, T, d_hid + style_dim]

        lengths_list: Sequence[int] = [int(n) for n in text_lengths.detach().cpu().tolist()]
        x_lstm = tt_bilstm_nlc(
            x_nlc=d_nlc,
            fwd=self.params.lstm_fwd,
            rev=self.params.lstm_rev,
            compute_kernel_config=ck,
            memory_config=memory_config,
            sequence_lengths=lengths_list,
        )
        # x_lstm: [B, T, d_hid]

        duration = self._duration_proj.forward(x_lstm, compute_kernel_config=ck, memory_config=memory_config)
        ttnn.deallocate(x_lstm)
        # duration: [B, T, max_dur]

        # ``en_BCT = d_NLC.permute(0,2,1) @ alignment`` ⇔ ``en_NLC = alignment^T @ d_NLC``.
        alignment_TaT = ttnn.permute(alignment_btTa, (0, 2, 1), memory_config=memory_config)
        en_nlc = ttnn.matmul(
            alignment_TaT,
            d_nlc,
            memory_config=memory_config,
            compute_kernel_config=ck,
        )
        ttnn.deallocate(alignment_TaT)
        ttnn.deallocate(d_nlc)
        # en_nlc: [B, T_aligned, d_hid + style_dim]

        return duration, en_nlc

    def F0Ntrain(
        self,
        en_nlc: ttnn.Tensor,
        style_bs: ttnn.Tensor,
        *,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
        use_fp32_boundary: bool = True,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Run the shared BiLSTM and the F0 / N decoder branches.

        Args:
            en_nlc: ``[B, T_aligned, d_hid + style_dim]`` (the NLC form of the reference's ``en``).
            style_bs: ``[B, style_dim]``.

        Returns:
            ``F0`` and ``N`` each shaped ``[B, T_out]`` where ``T_out = 2 * T_aligned``
            (the second F0/N block has ``upsample=True``).
        """
        ck = self.compute_kernel_config

        if use_fp32_boundary:
            en_fp32, owns_en = _to_fp32_if_needed(en_nlc, memory_config)
            if owns_en:
                ttnn.deallocate(en_nlc)
                en_nlc = en_fp32
            style_fp32, owns_style = _to_fp32_if_needed(style_bs, memory_config)
            if owns_style:
                ttnn.deallocate(style_bs)
                style_bs = style_fp32

        x_shared = tt_bilstm_nlc(
            x_nlc=en_nlc,
            fwd=self.params.shared_fwd,
            rev=self.params.shared_rev,
            compute_kernel_config=ck,
            memory_config=memory_config,
        )
        if use_fp32_boundary:
            # BiLSTM states are bf16; keep F0/N branch activations in fp32 to avoid ~Hz-level F0 drift.
            x_fp32, owns_x = _to_fp32_if_needed(x_shared, memory_config)
            if owns_x:
                ttnn.deallocate(x_shared)
                x_shared = x_fp32

        F0 = self._run_branch(
            x_shared,
            self._f0_blocks,
            self.params.f0_proj,
            style_bs,
            memory_config,
            preserve_fp32_on_upsample=use_fp32_boundary,
        )
        N = self._run_branch(x_shared, self._n_blocks, self.params.n_proj, style_bs, memory_config)
        ttnn.deallocate(x_shared)

        return F0, N

    __call__ = forward

    def _run_branch(
        self,
        x_nlc: ttnn.Tensor,
        blocks: Sequence[TTAdainResBlk1d],
        proj_params: TTConv1dParams,
        style_bs: ttnn.Tensor,
        memory_config: ttnn.MemoryConfig,
        *,
        preserve_fp32_on_upsample: bool = False,
    ) -> ttnn.Tensor:
        """Apply ``AdainResBlk1d`` stack then 1x1 conv, then squeeze the single-channel dim."""
        ck = self.compute_kernel_config
        y = x_nlc
        for block in blocks:
            # P7: only the upsample (conv_transpose) block needs dtype preservation; blanket
            # preserve_input_dtype on all F0 blocks regressed full-model PCC (~0.05).
            preserve = preserve_fp32_on_upsample and block._params.pool is not None
            y_new = block.forward(y, style_bs, memory_config=memory_config, preserve_input_dtype=preserve)
            if y is not x_nlc:
                ttnn.deallocate(y)
            y = y_new
        y = tt_conv1d_nlc(
            x_nlc=y,
            params=proj_params,
            device=self.device,
            compute_config=ck,
            memory_config=memory_config,
        )
        assert int(y.shape[-1]) == 1, f"Expected single-channel output, got {list(y.shape)}"
        return ttnn.squeeze(y, -1)
