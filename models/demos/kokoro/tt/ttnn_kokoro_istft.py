# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
TTNN port of `CustomSTFT.inverse` (iSTFT) used in Kokoro ISTFTNet.

Implements overlap-add via ConvTranspose1d equivalence:
  conv_transpose1d(x, w, stride=hop) == conv1d(zero_insert(x, hop), flip(w), padding=0)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from models.demos.kokoro.reference.kokoro_istftnet import CustomSTFT
from models.demos.kokoro.tt.ttnn_kokoro_conv import Conv1dParams, conv1d_nlc


@dataclass(frozen=True)
class CustomIstftParams:
    n_fft: int
    hop_length: int
    freq_bins: int
    center: bool
    conv_real: Conv1dParams  # [B,L,F] -> [B,L,1]
    conv_imag: Conv1dParams


def preprocess_custom_istft(
    torch_stft: CustomSTFT, device: ttnn.Device, *, weights_dtype=ttnn.bfloat16
) -> CustomIstftParams:
    n_fft = int(torch_stft.n_fft)
    hop = int(torch_stft.hop_length)
    freq_bins = int(torch_stft.freq_bins)

    # torch weight_backward_real/imag are [F,1,K] (for conv_transpose1d with in=F,out=1)
    w_r = torch_stft.weight_backward_real.detach().cpu().to(torch.float32)  # [F,1,K]
    w_i = torch_stft.weight_backward_imag.detach().cpu().to(torch.float32)

    # Convert to conv1d weight [out=1, in=F, k] and flip kernel
    w_r_c = w_r.squeeze(1).transpose(0, 1).unsqueeze(0).flip(-1).contiguous()  # [1,F,K]
    w_i_c = w_i.squeeze(1).transpose(0, 1).unsqueeze(0).flip(-1).contiguous()
    w_r_tt = ttnn.from_torch(w_r_c, dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    w_i_tt = ttnn.from_torch(w_i_c, dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    conv_r = Conv1dParams(
        weight=w_r_tt,
        bias=None,
        in_channels=freq_bins,
        out_channels=1,
        kernel_size=n_fft,
        stride=1,
        padding=0,
        groups=1,
    )
    conv_i = Conv1dParams(
        weight=w_i_tt,
        bias=None,
        in_channels=freq_bins,
        out_channels=1,
        kernel_size=n_fft,
        stride=1,
        padding=0,
        groups=1,
    )
    return CustomIstftParams(
        n_fft=n_fft,
        hop_length=hop,
        freq_bins=freq_bins,
        center=bool(torch_stft.center),
        conv_real=conv_r,
        conv_imag=conv_i,
    )


def _zero_insert_upsample_nlc(x_nlc: ttnn.Tensor, *, stride: int, device: ttnn.Device) -> ttnn.Tensor:
    # Output length: (L-1)*stride + 1
    if stride == 1:
        return x_nlc
    x_rep = ttnn.repeat_interleave(x_nlc, repeats=stride, dim=1)
    out_len = (int(x_nlc.shape[1]) - 1) * stride + 1
    x_rep = ttnn.slice(
        x_rep, (0, 0, 0), (x_rep.shape[0], out_len, x_rep.shape[2]), memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    mask = torch.zeros((1, out_len, 1), dtype=torch.float32)
    mask[0, ::stride, 0] = 1.0
    mask_tt = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    x_rm = x_rep if x_rep.layout == ttnn.ROW_MAJOR_LAYOUT else ttnn.to_layout(x_rep, ttnn.ROW_MAJOR_LAYOUT)
    y = ttnn.multiply(x_rm, mask_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return y if x_nlc.layout == ttnn.ROW_MAJOR_LAYOUT else ttnn.to_layout(y, ttnn.TILE_LAYOUT)


def custom_istft_inverse(
    *,
    magnitude_bft: ttnn.Tensor,  # [B,F,frames]
    phase_bft: ttnn.Tensor,  # [B,F,frames]
    params: CustomIstftParams,
    device: ttnn.Device,
    length: int | None = None,
) -> ttnn.Tensor:
    # real_part = mag * cos(phase), imag_part = mag * sin(phase)
    real = ttnn.multiply(magnitude_bft, ttnn.cos(phase_bft), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    imag = ttnn.multiply(magnitude_bft, ttnn.sin(phase_bft), memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # [B,F,T] -> [B,T,F] (NLC) then zero-insert upsample along time by hop
    real_nlc = ttnn.permute(real, (0, 2, 1))
    imag_nlc = ttnn.permute(imag, (0, 2, 1))
    real_up = _zero_insert_upsample_nlc(real_nlc, stride=params.hop_length, device=device)
    imag_up = _zero_insert_upsample_nlc(imag_nlc, stride=params.hop_length, device=device)

    # conv1d to reconstruct waveform pieces
    real_rec_nlc = conv1d_nlc(
        x_nlc=real_up, params=params.conv_real, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    imag_rec_nlc = conv1d_nlc(
        x_nlc=imag_up, params=params.conv_imag, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    wave_nlc = ttnn.subtract(real_rec_nlc, imag_rec_nlc, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [B,L,1]
    wave_bt = ttnn.reshape(wave_nlc, (wave_nlc.shape[0], wave_nlc.shape[1]), memory_config=ttnn.DRAM_MEMORY_CONFIG)

    if params.center:
        pad = params.n_fft // 2
        wave_bt = ttnn.slice(
            wave_bt, (0, pad), (wave_bt.shape[0], wave_bt.shape[1] - pad), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    if length is not None:
        wave_bt = ttnn.slice(wave_bt, (0, 0), (wave_bt.shape[0], length), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return wave_bt
