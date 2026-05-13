# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
TTNN port of Kokoro ``KokoroPredictor`` (ProsodyPredictor + TextEncoder).

Neural blocks run on device (bf16). Duration logits and matmul-based alignment use TTNN;
``pred_dur`` is derived from duration logits on host (no PyTorch module fallback).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

import ttnn
from models.experimental.kokoro.tt.ttnn_adain_resblk_encode import (
    AdainResBlk1d,
    infer_encode_dims,
    preprocess_encode_parameters,
)
from models.experimental.kokoro.tt.ttnn_kokoro_conv import Conv1dParams, conv1d_nlc
from models.experimental.kokoro.tt.ttnn_kokoro_lstm import LSTMParams, bilstm_nlc, preprocess_pytorch_lstm_1layer
from models.experimental.kokoro.tt.ttnn_kokoro_text_encoder import TtKokoroTextEncoder, preprocess_text_encoder


@dataclass(frozen=True)
class AdaLayerNormParams:
    fc_weight: ttnn.Tensor
    fc_bias: ttnn.Tensor
    ln_weight: ttnn.Tensor
    ln_bias: ttnn.Tensor
    eps: float


def _adalayernorm_nlc(
    *,
    x_nlc: ttnn.Tensor,  # [B,T,C]
    style_bc: ttnn.Tensor,  # [B,sty]
    params: AdaLayerNormParams,
    compute_kernel_config,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    # layer norm over last dim (C)
    y = ttnn.layer_norm(
        x_nlc,
        weight=params.ln_weight,
        bias=params.ln_bias,
        epsilon=params.eps,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
    )

    h = ttnn.linear(
        style_bc,
        params.fc_weight,
        bias=params.fc_bias,
        transpose_b=True,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
    )  # [B,2C]
    h = ttnn.reshape(h, [h.shape[0], 1, h.shape[-1]], memory_config=memory_config)  # [B,1,2C]
    c2 = h.shape[-1]
    c = c2 // 2
    gamma = ttnn.slice(h, [0, 0, 0], [h.shape[0], 1, c], [1, 1, 1])
    beta = ttnn.slice(h, [0, 0, c], [h.shape[0], 1, c2], [1, 1, 1])
    scale = ttnn.add(gamma, 1.0, memory_config=memory_config)
    y = ttnn.multiply(y, scale, memory_config=memory_config)
    y = ttnn.add(y, beta, memory_config=memory_config)
    return y


@dataclass(frozen=True)
class DurationEncoderLayerParams:
    lstm_fwd: LSTMParams
    lstm_rev: LSTMParams
    adaln: AdaLayerNormParams


@dataclass(frozen=True)
class DurationEncoderParams:
    layers: list[DurationEncoderLayerParams]
    d_model: int
    sty_dim: int


def preprocess_duration_encoder(
    duration_encoder: nn.Module, device: ttnn.Device, *, weights_dtype=ttnn.bfloat16
) -> DurationEncoderParams:
    layers: list[DurationEncoderLayerParams] = []

    for block in duration_encoder.lstms:
        if isinstance(block, nn.LSTM):
            fwd, rev = preprocess_pytorch_lstm_1layer(block, device, weights_dtype=weights_dtype)
            assert rev is not None
            last_lstm = (fwd, rev)
        else:
            # AdaLayerNorm
            fc_w = ttnn.from_torch(
                block.fc.weight.detach().cpu(), dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=device
            )
            fc_b = ttnn.from_torch(
                block.fc.bias.detach().cpu().reshape(1, 1, 1, -1),
                dtype=weights_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            # LN params are implicit in torch layer_norm; use ones/zeros with epsilon.
            c = block.channels
            ln_w = ttnn.from_torch(torch.ones(c), dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=device)
            ln_b = ttnn.from_torch(torch.zeros(c), dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=device)
            adaln = AdaLayerNormParams(fc_weight=fc_w, fc_bias=fc_b, ln_weight=ln_w, ln_bias=ln_b, eps=block.eps)
            fwd, rev = last_lstm  # type: ignore[misc]
            layers.append(DurationEncoderLayerParams(lstm_fwd=fwd, lstm_rev=rev, adaln=adaln))

    return DurationEncoderParams(layers=layers, d_model=duration_encoder.d_model, sty_dim=duration_encoder.sty_dim)


class TtKokoroDurationEncoder:
    def __init__(self, device: ttnn.Device, params: DurationEncoderParams):
        self.device = device
        self.params = params
        # WH B0: HiFi3 + fp32 dest acc avoids the HiFi4-fp32-accum HW bug warned in compute_kernel_config.cpp:65.
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

    def __call__(
        self,
        *,
        d_en_bct: ttnn.Tensor,  # [B,C,T]
        style_bs: ttnn.Tensor,  # [B, sty]
        input_lengths: torch.LongTensor,
        text_mask: torch.Tensor,  # [B,T] bool
    ) -> ttnn.Tensor:
        # Convert to NLC and concat style
        x = ttnn.permute(d_en_bct, (0, 2, 1))  # [B,T,C]
        B, T, C = x.shape
        s = ttnn.reshape(style_bs, [B, 1, style_bs.shape[-1]])
        s = ttnn.repeat(s, (1, T, 1))
        x = ttnn.concat([x, s], dim=2)

        # apply mask (masked -> 0), reuse across all layers
        m = ttnn.from_torch(
            (~text_mask).to(torch.float32).unsqueeze(-1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        x = ttnn.multiply(x, m)

        lengths_list = input_lengths.detach().cpu().tolist()
        for layer in self.params.layers:
            x = bilstm_nlc(
                x_nlc=x,
                fwd=layer.lstm_fwd,
                rev=layer.lstm_rev,
                compute_kernel_config=self.compute_kernel_config,
                sequence_lengths=lengths_list,
            )
            # AdaLayerNorm applies on d_model (not including style); slice first d_model dims
            x_d = ttnn.slice(x, [0, 0, 0], [x.shape[0], x.shape[1], self.params.d_model], [1, 1, 1])
            x_d = _adalayernorm_nlc(
                x_nlc=x_d,
                style_bc=style_bs,
                params=layer.adaln,
                compute_kernel_config=self.compute_kernel_config,
            )
            # concat style again for next layer / output
            x = ttnn.concat([x_d, s], dim=2)
            x = ttnn.multiply(x, m)

        ttnn.deallocate(m)
        return x  # [B,T,d_model+sty]


def _pred_alignment_from_duration(
    dur_ttnn: ttnn.Tensor,
    *,
    text_len: int,
    device: ttnn.Device,
) -> tuple[torch.LongTensor, ttnn.Tensor]:
    """Match reference ``KokoroPredictor`` discrete duration + one-hot alignment (host numpy + upload)."""
    dur_np = np.asarray(
        ttnn.to_torch(dur_ttnn).to(torch.float32).detach().cpu().numpy(),
        dtype=np.float64,
    )
    if dur_np.ndim >= 2:
        dur_row = np.reshape(dur_np[0], (-1,))[:text_len]
    else:
        dur_row = np.reshape(dur_np, (-1,))[:text_len]
    pred_dur_np = np.clip(np.round(dur_row), 1.0, None).astype(np.int64)
    idx = np.repeat(np.arange(text_len, dtype=np.int64), pred_dur_np)
    l_out = int(idx.shape[0])
    pred_aln = np.zeros((1, text_len, l_out), dtype=np.float32)
    pred_aln[0, idx, np.arange(l_out, dtype=np.int64)] = 1.0
    pred_aln_tt = ttnn.from_torch(
        torch.from_numpy(pred_aln),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    pred_dur = torch.from_numpy(pred_dur_np).long().squeeze()
    return pred_dur, pred_aln_tt


@dataclass(frozen=True)
class PredictorDurationParams:
    duration_encoder: DurationEncoderParams
    lstm_fwd: LSTMParams
    lstm_rev: LSTMParams
    duration_proj_w: ttnn.Tensor
    duration_proj_b: ttnn.Tensor
    max_dur: int
    sty_dim: int
    d_hid: int


def preprocess_predictor_duration(
    predictor: nn.Module, device: ttnn.Device, *, weights_dtype=ttnn.bfloat16
) -> PredictorDurationParams:
    dur_enc = preprocess_duration_encoder(predictor.text_encoder, device, weights_dtype=weights_dtype)
    lstm_fwd, lstm_rev = preprocess_pytorch_lstm_1layer(predictor.lstm, device, weights_dtype=weights_dtype)
    assert lstm_rev is not None
    dp = predictor.duration_proj.linear_layer
    # Keep the duration-projection weights in fp32 regardless of ``weights_dtype``: the output is rounded
    # to integer durations that drive alignment-matrix construction (``pred_aln``). bf16 rounding of
    # ``dur_logits`` flips ~1-2 integer durations per sentence, which shifts asr/F0/N output columns and
    # collapses their PCC to ~0.93 even though ``en`` itself stays at ~0.99 (LSTM-smoothed). Storing this
    # tiny 50×d_hid weight tensor at fp32 stabilises the rounding without measurable runtime cost.
    w = ttnn.from_torch(dp.weight.detach().cpu(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(
        dp.bias.detach().cpu().reshape(1, 1, 1, -1),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    return PredictorDurationParams(
        duration_encoder=dur_enc,
        lstm_fwd=lstm_fwd,
        lstm_rev=lstm_rev,
        duration_proj_w=w,
        duration_proj_b=b,
        max_dur=predictor.duration_proj.linear_layer.out_features,
        sty_dim=predictor.text_encoder.sty_dim,
        d_hid=predictor.text_encoder.d_model,
    )


class TtKokoroPredictorDuration:
    def __init__(self, device: ttnn.Device, params: PredictorDurationParams):
        self.device = device
        self.params = params
        # WH B0: HiFi3 + fp32 dest acc avoids the HiFi4-fp32-accum HW bug warned in compute_kernel_config.cpp:65.
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )
        self.duration_encoder = TtKokoroDurationEncoder(device, params.duration_encoder)

    def __call__(
        self,
        *,
        d_en_bct: ttnn.Tensor,  # [B, C, T]
        ref_s: torch.FloatTensor,  # [B,256] torch
        input_ids: torch.LongTensor,  # [B,T] torch
        input_lengths: torch.LongTensor,
        text_mask: torch.Tensor,  # [B,T] bool
        speed: float = 1.0,
        style_bs_tt: Optional[ttnn.Tensor] = None,
    ):
        if style_bs_tt is None:
            s_torch = ref_s[:, 128:].detach().cpu()
            s = ttnn.from_torch(s_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        else:
            s = style_bs_tt

        lengths_list = input_lengths.detach().cpu().tolist()
        d = self.duration_encoder(
            d_en_bct=d_en_bct,
            style_bs=s,
            input_lengths=input_lengths,
            text_mask=text_mask,
        )  # [B,T,d_hid+sty]

        x = bilstm_nlc(
            x_nlc=d,
            fwd=self.params.lstm_fwd,
            rev=self.params.lstm_rev,
            compute_kernel_config=self.compute_kernel_config,
            sequence_lengths=lengths_list,
        )
        # x is [B,T,d_hid]; cast to fp32 so the duration_proj matmul runs at full HiFi3 + fp32 dest precision.
        # The downstream ``round(dur)`` is integer-sensitive — even 0.04 absolute drift can flip a duration
        # and shift the alignment one frame, which collapses asr / F0 / N PCC even though ``en`` stays high.
        x_fp32 = x if x.dtype == ttnn.float32 else ttnn.typecast(x, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        dur_logits = ttnn.linear(
            x_fp32,
            self.params.duration_proj_w,
            bias=self.params.duration_proj_b,
            transpose_b=True,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if x_fp32 is not x:
            ttnn.deallocate(x_fp32)
        dur = ttnn.sigmoid(dur_logits)
        ttnn.deallocate(dur_logits)
        # sum over max_dur dim
        dur = ttnn.sum(dur, dim=-1, keepdim=False)  # [B,T]
        if speed != 1.0:
            dur = ttnn.multiply(dur, 1.0 / speed)

        pred_dur, pred_aln_tt = _pred_alignment_from_duration(dur, text_len=int(input_ids.shape[1]), device=self.device)
        return d, dur, pred_dur, pred_aln_tt


@dataclass(frozen=True)
class PredictorFullParams(PredictorDurationParams):
    shared_fwd: LSTMParams
    shared_rev: LSTMParams
    f0_blocks: list[AdainResBlk1d]
    n_blocks: list[AdainResBlk1d]
    f0_proj: Conv1dParams
    n_proj: Conv1dParams
    text_encoder: object


def preprocess_predictor_full(
    model: nn.Module, device: ttnn.Device, *, weights_dtype=ttnn.bfloat16
) -> PredictorFullParams:
    base = preprocess_predictor_duration(model.predictor, device, weights_dtype=weights_dtype)

    shared_fwd, shared_rev = preprocess_pytorch_lstm_1layer(model.predictor.shared, device, weights_dtype=weights_dtype)
    assert shared_rev is not None

    f0_blocks: list[AdainResBlk1d] = []
    for b in model.predictor.F0:
        di, do, sd = infer_encode_dims(b)
        f0_blocks.append(AdainResBlk1d(device, preprocess_encode_parameters(b, device), di, do, sd))
    n_blocks: list[AdainResBlk1d] = []
    for b in model.predictor.N:
        di, do, sd = infer_encode_dims(b)
        n_blocks.append(AdainResBlk1d(device, preprocess_encode_parameters(b, device), di, do, sd))

    def proj_params(conv: nn.Conv1d) -> Conv1dParams:
        w = conv.weight.detach().cpu()
        b = conv.bias.detach().cpu() if conv.bias is not None else None
        w_tt = ttnn.from_torch(w, dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        b_tt = (
            ttnn.from_torch(b.reshape(1, 1, 1, -1), dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
            if b is not None
            else None
        )
        return Conv1dParams(
            weight=w_tt,
            bias=b_tt,
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=conv.groups,
        )

    f0_proj = proj_params(model.predictor.F0_proj)
    n_proj = proj_params(model.predictor.N_proj)

    te_params = preprocess_text_encoder(model.text_encoder, device, weights_dtype=weights_dtype)

    return PredictorFullParams(
        **base.__dict__,
        shared_fwd=shared_fwd,
        shared_rev=shared_rev,
        f0_blocks=f0_blocks,
        n_blocks=n_blocks,
        f0_proj=f0_proj,
        n_proj=n_proj,
        text_encoder=te_params,
    )


class TtKokoroPredictor:
    def __init__(self, device: ttnn.Device, params: PredictorFullParams):
        self.device = device
        self.params = params
        # WH B0: HiFi3 + fp32 dest acc avoids the HiFi4-fp32-accum HW bug warned in compute_kernel_config.cpp:65.
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )
        self.duration_part = TtKokoroPredictorDuration(device, params)
        self.text_encoder = TtKokoroTextEncoder(device, params.text_encoder)

    def __call__(
        self,
        *,
        d_en_bct: ttnn.Tensor,
        ref_s: torch.FloatTensor,
        input_ids: torch.LongTensor,
        input_lengths: torch.LongTensor,
        text_mask: torch.Tensor,
        speed: float = 1.0,
        style_bs_tt: Optional[ttnn.Tensor] = None,
    ):
        if style_bs_tt is None:
            style = ttnn.from_torch(
                ref_s[:, 128:].detach().cpu(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )
        else:
            style = style_bs_tt

        d, dur, pred_dur, pred_aln_tt = self.duration_part(
            d_en_bct=d_en_bct,
            ref_s=ref_s,
            input_ids=input_ids,
            input_lengths=input_lengths,
            text_mask=text_mask,
            speed=speed,
            style_bs_tt=style,
        )

        d_bct = ttnn.permute(d, (0, 2, 1))  # [B,C,T]
        en = d_bct @ pred_aln_tt  # [B,C,sumdur]

        # Shared LSTM expects [B,L,C]
        en_nlc = ttnn.permute(en, (0, 2, 1))
        x_shared = bilstm_nlc(
            x_nlc=en_nlc,
            fwd=self.params.shared_fwd,
            rev=self.params.shared_rev,
            compute_kernel_config=self.compute_kernel_config,
        )
        x_shared_bcl = ttnn.permute(x_shared, (0, 2, 1))

        style_f32 = ttnn.typecast(style, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        f0 = ttnn.typecast(x_shared_bcl, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for blk in self.params.f0_blocks:
            f0 = blk(f0, style_f32)
        f0 = ttnn.typecast(f0, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        f0_nlc = ttnn.permute(f0, (0, 2, 1))
        f0_1 = conv1d_nlc(
            x_nlc=f0_nlc, params=self.params.f0_proj, device=self.device, compute_config=self.compute_kernel_config
        )
        f0_1 = ttnn.permute(f0_1, (0, 2, 1))  # [B,1,L]

        n = ttnn.typecast(x_shared_bcl, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for blk in self.params.n_blocks:
            n = blk(n, style_f32)
        n = ttnn.typecast(n, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        n_nlc = ttnn.permute(n, (0, 2, 1))
        n_1 = conv1d_nlc(
            x_nlc=n_nlc, params=self.params.n_proj, device=self.device, compute_config=self.compute_kernel_config
        )
        n_1 = ttnn.permute(n_1, (0, 2, 1))

        t_en_bct = self.text_encoder(input_ids, input_lengths, text_mask)  # [B,C,T]
        asr = t_en_bct @ pred_aln_tt  # [B,C,sumdur]

        return dict(
            d=d,
            duration=dur,
            pred_dur=pred_dur,
            pred_aln_trg=pred_aln_tt,
            en=en,
            F0_pred=f0_1,  # [B,1,L] TTNN
            N_pred=n_1,  # [B,1,L] TTNN
            t_en=t_en_bct,
            asr=asr,  # [B,C,sumdur] TTNN
        )
