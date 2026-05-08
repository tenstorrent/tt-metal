# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
TTNN port of Kokoro ISTFTNet decoder front-end.

This file ports the "Decoder" stack up to the point where it calls the waveform `Generator`.
The generator + STFT/iSTFT will be ported next.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

import ttnn
from models.demos.kokoro.tt.ttnn_kokoro_conv import Conv1dParams, conv1d_nlc
from models.demos.kokoro.tt.ttnn_kokoro_predictor import (
    AdainResBlk1dParams,
    _adain_resblk1d_forward_bcl,
    _preprocess_adain_resblk_1d,
)


@dataclass(frozen=True)
class DecoderFrontParams:
    f0_conv: Conv1dParams
    n_conv: Conv1dParams
    asr_res: Conv1dParams
    encode: AdainResBlk1dParams
    decode: list[AdainResBlk1dParams]


def _preprocess_plain_conv1d(conv: nn.Conv1d, *, weights_dtype=ttnn.bfloat16) -> Conv1dParams:
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
        kernel_size=conv.kernel_size[0],
        stride=conv.stride[0],
        padding=conv.padding[0],
        groups=conv.groups,
    )


def preprocess_decoder_front(
    torch_decoder: nn.Module, device: ttnn.Device, *, weights_dtype=ttnn.bfloat16
) -> DecoderFrontParams:
    # F0_conv / N_conv are weight_norm wrappers, grab underlying conv via weight_v/weight_g
    # Reuse weight_norm math by constructing a fake weight_norm module signature like in predictor's helper.
    def weight_norm_conv_params(conv_wn: nn.Module) -> Conv1dParams:
        # conv_wn is weight_norm(nn.Conv1d)
        from models.demos.kokoro.tt.ttnn_kokoro_conv import weight_norm_weight

        w = weight_norm_weight(conv_wn.weight_v.detach().cpu(), conv_wn.weight_g.detach().cpu())
        b = conv_wn.bias.detach().cpu() if conv_wn.bias is not None else None
        w_tt = ttnn.from_torch(w, dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        b_tt = (
            ttnn.from_torch(b.reshape(1, 1, 1, -1), dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
            if b is not None
            else None
        )
        return Conv1dParams(
            weight=w_tt,
            bias=b_tt,
            in_channels=conv_wn.in_channels,
            out_channels=conv_wn.out_channels,
            kernel_size=conv_wn.kernel_size[0],
            stride=conv_wn.stride[0],
            padding=conv_wn.padding[0],
            groups=conv_wn.groups,
        )

    f0_conv = weight_norm_conv_params(torch_decoder.F0_conv)
    n_conv = weight_norm_conv_params(torch_decoder.N_conv)

    # asr_res is Sequential( weight_norm(Conv1d(512->64,k=1)) )
    asr_conv = torch_decoder.asr_res[0]
    asr_res = weight_norm_conv_params(asr_conv)

    encode = _preprocess_adain_resblk_1d(torch_decoder.encode, device, weights_dtype=weights_dtype)
    decode = [_preprocess_adain_resblk_1d(b, device, weights_dtype=weights_dtype) for b in torch_decoder.decode]
    return DecoderFrontParams(f0_conv=f0_conv, n_conv=n_conv, asr_res=asr_res, encode=encode, decode=decode)


class TtKokoroDecoderFront:
    def __init__(self, device: ttnn.Device, params: DecoderFrontParams):
        self.device = device
        self.params = params
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4
        )

    def __call__(
        self,
        *,
        asr_bct: ttnn.Tensor,
        f0_pred: torch.Tensor | ttnn.Tensor,
        n_pred: torch.Tensor | ttnn.Tensor,
        style_s: torch.Tensor | ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Returns the feature tensor that would be fed into `Generator` in the torch reference.

        Inputs:
        - asr_bct: [B, C=512, T] TTNN
        - f0_pred/n_pred: [B, T0] torch float
        - style_s: [B, style_dim] torch float (decoder uses ref_s[:, :128])
        """
        B, C, T = asr_bct.shape

        if isinstance(style_s, ttnn.Tensor):
            style = style_s
        else:
            style = ttnn.from_torch(
                style_s.detach().cpu(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )

        # F0_conv / N_conv expect [B,1,L] in torch; use NLC conv wrapper
        if isinstance(f0_pred, ttnn.Tensor):
            f0 = f0_pred
        else:
            f0 = ttnn.from_torch(
                f0_pred.unsqueeze(1).detach().cpu(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )
        if isinstance(n_pred, ttnn.Tensor):
            n = n_pred
        else:
            n = ttnn.from_torch(
                n_pred.unsqueeze(1).detach().cpu(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )

        f0_nlc = ttnn.permute(f0, (0, 2, 1))
        n_nlc = ttnn.permute(n, (0, 2, 1))
        f0_feat = conv1d_nlc(
            x_nlc=f0_nlc, params=self.params.f0_conv, device=self.device, compute_config=self.compute_kernel_config
        )
        n_feat = conv1d_nlc(
            x_nlc=n_nlc, params=self.params.n_conv, device=self.device, compute_config=self.compute_kernel_config
        )
        f0_feat = ttnn.permute(f0_feat, (0, 2, 1))  # [B,1,T’]
        n_feat = ttnn.permute(n_feat, (0, 2, 1))
        # concat requires consistent sharding; normalize to interleaved
        try:
            f0_feat = ttnn.sharded_to_interleaved(f0_feat, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        except Exception:
            pass
        try:
            n_feat = ttnn.sharded_to_interleaved(n_feat, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        except Exception:
            pass

        # asr_res conv 512->64
        asr_nlc = ttnn.permute(asr_bct, (0, 2, 1))
        asr_res_nlc = conv1d_nlc(
            x_nlc=asr_nlc, params=self.params.asr_res, device=self.device, compute_config=self.compute_kernel_config
        )
        asr_res = ttnn.permute(asr_res_nlc, (0, 2, 1))  # [B,64,T]
        try:
            asr_res = ttnn.sharded_to_interleaved(asr_res, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        except Exception:
            pass
        try:
            asr_bct = ttnn.sharded_to_interleaved(asr_bct, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        except Exception:
            pass

        # encode takes cat([asr, F0, N]) along channel dim
        x = ttnn.concat([asr_bct, f0_feat, n_feat], dim=1)
        x = _adain_resblk1d_forward_bcl(
            x_bcl=x,
            style_bs=style,
            params=self.params.encode,
            device=self.device,
            compute_kernel_config=self.compute_kernel_config,
        )

        res = True
        for blk in self.params.decode:
            if res:
                x = ttnn.concat([x, asr_res, f0_feat, n_feat], dim=1)
            x = _adain_resblk1d_forward_bcl(
                x_bcl=x,
                style_bs=style,
                params=blk,
                device=self.device,
                compute_kernel_config=self.compute_kernel_config,
            )
            if blk.upsample:
                res = False

        return x
