# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
TTNN port of `CustomSTFT.transform` (forward STFT) used in Kokoro ISTFTNet.

Inverse iSTFT will be ported separately (it requires overlap-add with large stride).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import ttnn
from models.demos.kokoro.reference.kokoro_istftnet import CustomSTFT


@dataclass(frozen=True)
class CustomStftTransformParams:
    n_fft: int
    hop_length: int
    freq_bins: int
    center: bool
    pad_mode: str
    window_k: ttnn.Tensor  # [K]
    dft_real_kf: ttnn.Tensor  # [K, F]
    dft_imag_kf: ttnn.Tensor  # [K, F] (already includes negative sign like reference)


def preprocess_custom_stft_transform(
    torch_stft: CustomSTFT, device: ttnn.Device, *, weights_dtype=ttnn.bfloat16
) -> CustomStftTransformParams:
    """
    Build DFT matrices for matmul-based STFT.
    """
    assert torch_stft.pad_mode == "replicate"
    n_fft = int(torch_stft.n_fft)
    hop = int(torch_stft.hop_length)
    freq_bins = int(torch_stft.freq_bins)

    window = torch_stft.window.detach().cpu().to(torch.float32).numpy()  # [K]
    n = np.arange(n_fft, dtype=np.float32)
    k = np.arange(freq_bins, dtype=np.float32)
    angle = 2 * np.pi * np.outer(k, n) / float(n_fft)  # [F,K]
    dft_real_fk = np.cos(angle) * window  # [F,K]
    dft_imag_fk = (-np.sin(angle)) * window  # [F,K] matches reference

    # Store as [K,F] for matmul: [*,K] x [K,F]
    dft_real_kf = torch.from_numpy(dft_real_fk).to(torch.float32).transpose(0, 1).contiguous()
    dft_imag_kf = torch.from_numpy(dft_imag_fk).to(torch.float32).transpose(0, 1).contiguous()

    window_k = ttnn.from_torch(
        torch.from_numpy(window).to(torch.float32), dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    real_tt = ttnn.from_torch(dft_real_kf, dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    imag_tt = ttnn.from_torch(dft_imag_kf, dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    return CustomStftTransformParams(
        n_fft=n_fft,
        hop_length=hop,
        freq_bins=freq_bins,
        center=bool(torch_stft.center),
        pad_mode=str(torch_stft.pad_mode),
        window_k=window_k,
        dft_real_kf=real_tt,
        dft_imag_kf=imag_tt,
    )


def _replicate_pad_1d_bt(x_bt: ttnn.Tensor, *, pad_left: int, pad_right: int) -> ttnn.Tensor:
    # x_bt: [B, T]
    x = x_bt
    if pad_left > 0:
        left_val = ttnn.slice(x, (0, 0), (x.shape[0], 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [B,1]
        left = ttnn.repeat(left_val, (1, pad_left))
        x = ttnn.concat([left, x], dim=1)
    if pad_right > 0:
        right_val = ttnn.slice(
            x, (0, x.shape[1] - 1), (x.shape[0], x.shape[1]), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )  # [B,1]
        right = ttnn.repeat(right_val, (1, pad_right))
        x = ttnn.concat([x, right], dim=1)
    return x


def custom_stft_transform(
    *,
    waveform_bt: ttnn.Tensor,  # [B, T]
    params: CustomStftTransformParams,
    device: ttnn.Device,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """
    Matmul-based STFT:
      - frame extraction on device via slices/concat
      - apply Hann window
      - DFT via matmul with precomputed cos/sin matrices

    Returns (magnitude, phase):
      - magnitude: [B, F, frames]
      - phase:     [B, F, frames]
    """
    x = waveform_bt
    if x.layout != ttnn.ROW_MAJOR_LAYOUT:
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    if params.center:
        pad = params.n_fft // 2
        x = _replicate_pad_1d_bt(x, pad_left=pad, pad_right=pad)

    B, T = int(x.shape[0]), int(x.shape[1])
    K = params.n_fft
    hop = params.hop_length
    frames = (T - K) // hop + 1

    # Build frames tensor [B, frames, K]
    frame_list = []
    for i in range(frames):
        start = i * hop
        seg_bk = ttnn.slice(x, (0, start), (B, start + K), memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [B,K]
        seg_b1k = ttnn.reshape(seg_bk, (B, 1, K), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        frame_list.append(seg_b1k)
    frames_bfk = ttnn.concat(frame_list, dim=1)  # [B,frames,K]

    # Apply window: [K] -> [1,1,K] -> [B,frames,K]
    win_11k = ttnn.reshape(params.window_k, (1, 1, K), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    win_bfk = ttnn.repeat(win_11k, (B, frames, 1))
    frames_bfk = ttnn.multiply(frames_bfk, win_bfk)

    # Matmul: reshape to [B*frames, K]
    frames_2d = ttnn.reshape(frames_bfk, (B * frames, K), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    frames_2d = ttnn.to_layout(frames_2d, ttnn.TILE_LAYOUT)

    real_2d = ttnn.matmul(frames_2d, params.dft_real_kf)
    imag_2d = ttnn.matmul(frames_2d, params.dft_imag_kf)

    real_bff = ttnn.reshape(real_2d, (B, frames, params.freq_bins), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    imag_bff = ttnn.reshape(imag_2d, (B, frames, params.freq_bins), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    real_bfF = ttnn.permute(real_bff, (0, 2, 1))  # [B,F,frames]
    imag_bfF = ttnn.permute(imag_bff, (0, 2, 1))

    mag = ttnn.sqrt(ttnn.add(ttnn.pow(real_bfF, 2.0), ttnn.add(ttnn.pow(imag_bfF, 2.0), 1e-14)))
    phase = ttnn.atan2(imag_bfF, real_bfF)
    return mag, phase
