# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
TTNN port of Kokoro predictor (incremental).

Duration logits run on device; ``pred_dur`` / alignment indices use small CPU tensors
(``round``, ``repeat_interleave``), then the alignment is uploaded for matmuls on device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

import ttnn
from models.experimental.kokoro.tt.ttnn_kokoro_conv import Conv1dParams, conv1d_nlc, weight_norm_weight
from models.experimental.kokoro.tt.ttnn_kokoro_lstm import LSTMParams, bilstm_nlc, preprocess_pytorch_lstm_1layer
from models.experimental.kokoro.tt.ttnn_kokoro_norm import AdaIN1dParams, InstanceNorm1dParams, adain_1d_nlc
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
    one = ttnn.full(gamma.shape, fill_value=1.0, dtype=gamma.dtype, layout=gamma.layout, device=gamma.device())
    scale = ttnn.add(one, gamma, memory_config=memory_config)
    y = ttnn.multiply(y, scale, memory_config=memory_config)
    y = ttnn.add(y, beta, memory_config=memory_config)
    return y


@dataclass(frozen=True)
class AdainResBlk1dParams:
    conv1: Conv1dParams
    conv2: Conv1dParams
    norm1: AdaIN1dParams
    norm2: AdaIN1dParams
    conv1x1: Optional[Conv1dParams]
    upsample: bool
    # Depthwise ConvTranspose1d(k=3,s=2,p=1,op=1) == conv1d(zero_insert(x), flip(w), p=1, groups=C)
    pool_conv: Optional[Conv1dParams]


def _preprocess_adain_resblk_1d(
    block: nn.Module, device: ttnn.Device, *, weights_dtype=ttnn.bfloat16
) -> AdainResBlk1dParams:
    def conv1d_params(conv: nn.Module) -> Conv1dParams:
        w = weight_norm_weight(conv.weight_v.detach().cpu(), conv.weight_g.detach().cpu())
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
            kernel_size=conv.kernel_size[0],
            stride=conv.stride[0],
            padding=conv.padding[0],
            groups=conv.groups,
        )

    def adain_params(adain: nn.Module) -> AdaIN1dParams:
        in_w = getattr(adain.norm, "weight", None)
        in_b = getattr(adain.norm, "bias", None)
        inst = InstanceNorm1dParams(
            weight=ttnn.from_torch(in_w.detach().cpu(), dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=device)
            if in_w is not None
            else None,
            bias=ttnn.from_torch(in_b.detach().cpu(), dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=device)
            if in_b is not None
            else None,
            eps=adain.norm.eps,
        )
        fc_w = ttnn.from_torch(
            adain.fc.weight.detach().cpu(), dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        fc_b = ttnn.from_torch(
            adain.fc.bias.detach().cpu().reshape(1, 1, 1, -1),
            dtype=weights_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        return AdaIN1dParams(fc_weight=fc_w, fc_bias=fc_b, instancenorm=inst)

    upsample = getattr(block, "upsample_type", "none") != "none"
    pool_conv = None
    if upsample:
        convt = block.pool
        assert (
            convt.kernel_size[0] == 3
            and convt.stride[0] == 2
            and convt.padding[0] == 1
            and convt.output_padding[0] == 1
        )
        assert convt.groups == convt.in_channels == convt.out_channels
        w = weight_norm_weight(convt.weight_v.detach().cpu(), convt.weight_g.detach().cpu())  # [C,1,3]
        w = torch.flip(w, dims=[2])
        b = convt.bias.detach().cpu() if convt.bias is not None else None
        w_tt = ttnn.from_torch(w, dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        b_tt = (
            ttnn.from_torch(b.reshape(1, 1, 1, -1), dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
            if b is not None
            else None
        )
        pool_conv = Conv1dParams(
            weight=w_tt,
            bias=b_tt,
            in_channels=convt.in_channels,
            out_channels=convt.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=convt.groups,
        )

    conv1 = conv1d_params(block.conv1)
    conv2 = conv1d_params(block.conv2)
    conv1x1 = conv1d_params(block.conv1x1) if getattr(block, "learned_sc", False) else None
    norm1 = adain_params(block.norm1)
    norm2 = adain_params(block.norm2)
    return AdainResBlk1dParams(
        conv1=conv1,
        conv2=conv2,
        norm1=norm1,
        norm2=norm2,
        conv1x1=conv1x1,
        upsample=upsample,
        pool_conv=pool_conv,
    )


def _adain_resblk1d_forward_bcl(
    *,
    x_bcl: ttnn.Tensor,
    style_bs: ttnn.Tensor,
    params: AdainResBlk1dParams,
    device: ttnn.Device,
    compute_kernel_config,
) -> ttnn.Tensor:
    # BCL -> NLC for conv wrappers
    x = ttnn.permute(x_bcl, (0, 2, 1))  # [B,L,C]
    shortcut = x
    if params.upsample:
        # Reference shortcut uses nearest upsample (not convtranspose pool)
        shortcut = ttnn.repeat_interleave(shortcut, repeats=2, dim=1)
    if params.conv1x1 is not None:
        shortcut = conv1d_nlc(
            x_nlc=shortcut, params=params.conv1x1, device=device, compute_config=compute_kernel_config
        )

    y = adain_1d_nlc(x_nlc=x, s_bc=style_bs, params=params.norm1, compute_kernel_config=compute_kernel_config)
    y = ttnn.leaky_relu(y, negative_slope=0.2)
    if params.upsample:
        # Reference residual uses depthwise ConvTranspose1d pool before conv1
        assert params.pool_conv is not None
        # zero-insert upsample by 2: [x0,0,x1,0,...], then depthwise conv1d with flipped kernel
        zeros = ttnn.zeros(y.shape, dtype=y.dtype, layout=y.layout, device=device)
        y2 = ttnn.reshape(y, [y.shape[0], y.shape[1], 1, y.shape[2]])
        z2 = ttnn.reshape(zeros, [zeros.shape[0], zeros.shape[1], 1, zeros.shape[2]])
        y = ttnn.concat([y2, z2], dim=2)
        # y is [B, L, 2, C] -> [B, 2L, C]
        y = ttnn.reshape(y, [y.shape[0], y.shape[1] * 2, y.shape[3]])
        ttnn.deallocate(zeros)
        ttnn.deallocate(y2)
        ttnn.deallocate(z2)
        y = conv1d_nlc(x_nlc=y, params=params.pool_conv, device=device, compute_config=compute_kernel_config)
    y = conv1d_nlc(x_nlc=y, params=params.conv1, device=device, compute_config=compute_kernel_config)
    y = adain_1d_nlc(x_nlc=y, s_bc=style_bs, params=params.norm2, compute_kernel_config=compute_kernel_config)
    y = ttnn.leaky_relu(y, negative_slope=0.2)
    y = conv1d_nlc(x_nlc=y, params=params.conv2, device=device, compute_config=compute_kernel_config)

    out = ttnn.add(y, shortcut)
    out = ttnn.multiply(out, 1.0 / (2.0**0.5))
    # conv ops can return sharded tensors; make interleaved before transpose
    try:
        out = ttnn.sharded_to_interleaved(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    except Exception:
        pass
    out = ttnn.permute(out, (0, 2, 1))  # back to BCL
    return out


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
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4
        )

    def __call__(
        self,
        *,
        d_en_bct: ttnn.Tensor,  # [B,C,T]
        style_bs: ttnn.Tensor,  # [B, sty]
        text_mask: torch.Tensor,  # [B,T] bool
    ) -> ttnn.Tensor:
        # Convert to NLC and concat style
        x = ttnn.permute(d_en_bct, (0, 2, 1))  # [B,T,C]
        B, T, C = x.shape
        s = ttnn.reshape(style_bs, [B, 1, style_bs.shape[-1]])
        s = ttnn.repeat(s, (1, T, 1))
        x = ttnn.concat([x, s], dim=2)

        # apply mask (masked -> 0)
        m = ttnn.from_torch(
            (~text_mask).to(torch.float32).unsqueeze(-1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        x = ttnn.multiply(x, m)
        ttnn.deallocate(m)

        for layer in self.params.layers:
            x = bilstm_nlc(
                x_nlc=x, fwd=layer.lstm_fwd, rev=layer.lstm_rev, compute_kernel_config=self.compute_kernel_config
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
            # re-mask
            m = ttnn.from_torch(
                (~text_mask).to(torch.float32).unsqueeze(-1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            x = ttnn.multiply(x, m)
            ttnn.deallocate(m)

        return x  # [B,T,d_model+sty]


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
    w = ttnn.from_torch(dp.weight.detach().cpu(), dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(
        dp.bias.detach().cpu().reshape(1, 1, 1, -1), dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=device
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
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4
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
    ):
        # style s is ref_s[:,128:]
        s_torch = ref_s[:, 128:].detach().cpu()
        s = ttnn.from_torch(s_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        d = self.duration_encoder(d_en_bct=d_en_bct, style_bs=s, text_mask=text_mask)  # [B,T,d_hid+sty]

        x = bilstm_nlc(
            x_nlc=d,
            fwd=self.params.lstm_fwd,
            rev=self.params.lstm_rev,
            compute_kernel_config=self.compute_kernel_config,
        )
        # x is [B,T,d_hid]
        dur_logits = ttnn.linear(
            x,
            self.params.duration_proj_w,
            bias=self.params.duration_proj_b,
            transpose_b=True,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        dur = ttnn.sigmoid(dur_logits)
        # sum over max_dur dim
        dur = ttnn.sum(dur, dim=-1, keepdim=False)  # [B,T]
        if speed != 1.0:
            dur = ttnn.multiply(dur, 1.0 / speed)

        # Host discrete ops for pred_dur and alignment (fallback).
        dur_host = ttnn.to_torch(dur).to(torch.float32)
        pred_dur = torch.round(dur_host).clamp(min=1).long().squeeze()

        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1]), pred_dur)
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), dtype=torch.float32)
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1.0
        pred_aln_trg = pred_aln_trg.unsqueeze(0)

        pred_aln_tt = ttnn.from_torch(pred_aln_trg, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        return d, dur, pred_dur, pred_aln_tt


@dataclass(frozen=True)
class PredictorFullParams(PredictorDurationParams):
    shared_fwd: LSTMParams
    shared_rev: LSTMParams
    f0_blocks: list[AdainResBlk1dParams]
    n_blocks: list[AdainResBlk1dParams]
    f0_proj: Conv1dParams
    n_proj: Conv1dParams
    text_encoder: object


def preprocess_predictor_full(
    model: nn.Module, device: ttnn.Device, *, weights_dtype=ttnn.bfloat16
) -> PredictorFullParams:
    base = preprocess_predictor_duration(model.predictor, device, weights_dtype=weights_dtype)

    shared_fwd, shared_rev = preprocess_pytorch_lstm_1layer(model.predictor.shared, device, weights_dtype=weights_dtype)
    assert shared_rev is not None

    f0_blocks = [_preprocess_adain_resblk_1d(b, device, weights_dtype=weights_dtype) for b in model.predictor.F0]
    n_blocks = [_preprocess_adain_resblk_1d(b, device, weights_dtype=weights_dtype) for b in model.predictor.N]

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
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4
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
    ):
        d, dur, pred_dur, pred_aln_tt = self.duration_part(
            d_en_bct=d_en_bct,
            ref_s=ref_s,
            input_ids=input_ids,
            input_lengths=input_lengths,
            text_mask=text_mask,
            speed=speed,
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

        style = ttnn.from_torch(
            ref_s[:, 128:].detach().cpu(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

        f0 = x_shared_bcl
        for blk in self.params.f0_blocks:
            f0 = _adain_resblk1d_forward_bcl(
                x_bcl=f0,
                style_bs=style,
                params=blk,
                device=self.device,
                compute_kernel_config=self.compute_kernel_config,
            )
        f0_nlc = ttnn.permute(f0, (0, 2, 1))
        f0_1 = conv1d_nlc(
            x_nlc=f0_nlc, params=self.params.f0_proj, device=self.device, compute_config=self.compute_kernel_config
        )
        f0_1 = ttnn.permute(f0_1, (0, 2, 1))  # [B,1,L]

        n = x_shared_bcl
        for blk in self.params.n_blocks:
            n = _adain_resblk1d_forward_bcl(
                x_bcl=n,
                style_bs=style,
                params=blk,
                device=self.device,
                compute_kernel_config=self.compute_kernel_config,
            )
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
