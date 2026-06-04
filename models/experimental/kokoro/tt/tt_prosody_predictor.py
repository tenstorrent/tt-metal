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

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

import ttnn

from .tt_adain_resblk_1d import TTAdainResBlk1d, TTAdainResBlk1dParams, preprocess_tt_adain_resblk_1d
from .tt_conv import TTConv1dParams, tt_conv1d_nlc, upload_conv1d_params_from_module
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


def _conv1d_no_weightnorm_to_tt_params(conv: nn.Conv1d, device, *, weights_dtype) -> TTConv1dParams:
    """Upload a plain ``nn.Conv1d`` (no weight_norm) to device TILE for :func:`tt_conv1d_nlc`."""
    return upload_conv1d_params_from_module(conv, device, weights_dtype=weights_dtype)


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

    f0_proj = _conv1d_no_weightnorm_to_tt_params(module.F0_proj, device, weights_dtype=conv_weights_dtype)
    n_proj = _conv1d_no_weightnorm_to_tt_params(module.N_proj, device, weights_dtype=conv_weights_dtype)

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


# ``en_nlc = alignment_TaT @ d_nlc`` has shape M=T_aligned, K=T, N=d_hid+style_dim.
# N is a model constant (192 for the profiled config); M=T_aligned and K=T are
# sequence-dependent. A program-config / memory sweep on this exact shape
# (models/experimental/tt_symbiote/tests/test_kokoro_base_matmul_sweep.py,
# 64x32x192) found L1-width-sharded output fastest (~1.67x over DRAM), then a
# 1D-mcast "full-N-row" config (~1.10x, output left DRAM-interleaved). Both only
# run for batch==1; the default matmul handles every other case (incl. the B>1
# path and the long-sequence/OLA regime the rest of the model relies on). We
# bound M so the fast paths stay L1-resident and fall back otherwise.
_TILE = 32
_EN_M_CAP_SHARD = 2048  # max T_aligned (rows) for the L1-width-sharded output path
_EN_M_CAP_MCAST = 4096  # max T_aligned (rows) for the 1D-mcast full-N-row path


def _en_matmul_plan(alignment_TaT: ttnn.Tensor, d_nlc: ttnn.Tensor):
    """Pick (program_config, out_memory_config, reshard_back) for the en_nlc matmul.

    ``program_config`` / ``out_memory_config`` are ``None`` to mean "use the
    ttnn defaults" (the current behaviour). ``reshard_back`` is True when the
    output is produced sharded and must be converted back to the caller's
    ``memory_config`` to preserve the return contract.
    """
    B = int(d_nlc.shape[0])
    M = int(alignment_TaT.shape[-2])  # T_aligned
    N = int(d_nlc.shape[-1])  # d_hid + style_dim
    if B != 1 or (N % _TILE) != 0:
        return None, None, False
    cores = N // _TILE
    if M <= _EN_M_CAP_SHARD:
        out_mc = ttnn.create_sharded_memory_config(
            (M, N), ttnn.CoreGrid(y=1, x=cores), ttnn.ShardStrategy.WIDTH, ttnn.ShardOrientation.ROW_MAJOR
        )
        return None, out_mc, True
    if M <= _EN_M_CAP_MCAST:
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(cores, 1),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=math.ceil(M / _TILE),
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        return pc, None, False
    return None, None, False


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
        # Best swept config for this 64x32x192-class matmul (see _en_matmul_plan): a fast
        # L1-width-sharded / 1D-mcast path when feasible (B==1, bounded T_aligned), else the
        # default matmul. The sharded path reshards back to ``memory_config`` so downstream
        # (F0Ntrain) sees the same interleaved layout as before.
        en_pc, en_out_mc, en_reshard = _en_matmul_plan(alignment_TaT, d_nlc)
        en_nlc = ttnn.matmul(
            alignment_TaT,
            d_nlc,
            program_config=en_pc,
            memory_config=en_out_mc if en_out_mc is not None else memory_config,
            compute_kernel_config=ck,
        )
        if en_reshard:
            en_nlc = ttnn.to_memory_config(en_nlc, memory_config)
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
            fp32_state=True,
        )
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
            preserve_fp32_on_upsample=True,
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
