# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `mel_spectrogram` (coqui/XTTS-v2
`hifigan_decoder.speaker_encoder.torch_spec.1`).

The submodule is a `torchaudio.transforms.MelSpectrogram` (n_fft=512, win=400,
hop=160, power=2, center=True, reflect pad, onesided, Hann window, then a 64-band
mel filterbank). A short-time Fourier transform is a windowed DFT, which is a pure
matmul once the frames are extracted:

    frames[i, n] = padded[i*hop + n]                       (overlapping windows)
    real = frames @ (window ⊙ cos_basis)                   # DFT, window folded in
    imag = frames @ (window ⊙ -sin_basis)                  # -> [n_frames, n_freqs]
    power = real^2 + imag^2                                 # power=2 spectrogram
    mel   = fb^T @ power^T                                  # [n_mels, n_frames]

Native ttnn: the frames are cut on-device with `ttnn.slice`, the DFT and the mel
projection are `ttnn.matmul`, and the power/accumulate are elementwise ops — all in
float32. Only the reflect boundary-pad (center=True) is done host-side, since ttnn
has no reflect pad; it is pure data movement on the raw waveform, not spectral compute.
"""

from __future__ import annotations

import math

import ttnn

from models.demos.xtts_v2._stubs.mel_scale import build as _b_mel_scale


HF_MODEL_ID = "coqui/XTTS-v2"


def build(device, torch_module):
    """Bind the STFT window / mel filterbank and return a native ttnn forward closure."""
    import torch

    mspec = torch_module
    spectro = mspec.spectrogram
    n_fft = int(spectro.n_fft)
    hop = int(spectro.hop_length)
    win_length = int(spectro.win_length)
    power = float(spectro.power) if spectro.power is not None else 2.0
    center = bool(spectro.center)
    pad_mode = str(getattr(spectro, "pad_mode", "reflect"))
    extra_pad = int(getattr(spectro, "pad", 0))
    n_freqs = n_fft // 2 + 1

    window = spectro.window.detach()                     # [win_length]
    # Zero-pad the analysis window to n_fft, centered (torchaudio convention).
    w_full = torch.zeros(n_fft, dtype=torch.float32)
    off = (n_fft - win_length) // 2
    w_full[off : off + win_length] = window.float()

    n = torch.arange(n_fft).float()
    k = torch.arange(n_freqs).float()
    ang = 2.0 * math.pi * torch.outer(n, k) / n_fft      # [n_fft, n_freqs]
    w_cos = (w_full.unsqueeze(1) * torch.cos(ang)).contiguous()   # window folded into basis
    w_sin = (w_full.unsqueeze(1) * -torch.sin(ang)).contiguous()

    Wcos = ttnn.from_torch(w_cos, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    Wsin = ttnn.from_torch(w_sin, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    # mel filterbank projection: graduated leaf stub (mel_scale)
    mel_scale = _b_mel_scale(device, mspec.mel_scale)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    DRAM = ttnn.DRAM_MEMORY_CONFIG

    def forward(x, *args, **kwargs):
        if not isinstance(x, ttnn.Tensor):
            x = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        # Boundary reflect-pad on host (ttnn has no reflect pad); data movement only.
        xh = ttnn.to_torch(x).float().reshape(1, -1)
        if extra_pad > 0:
            xh = torch.nn.functional.pad(xh, (extra_pad, extra_pad))
        if center:
            xh = torch.nn.functional.pad(xh, (n_fft // 2, n_fft // 2), mode=pad_mode)
        xp = ttnn.from_torch(xh.contiguous(), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

        L = int(xp.shape[1])
        n_frames = 1 + (L - n_fft) // hop
        frames = ttnn.concat(
            [ttnn.slice(xp, [0, i * hop], [1, i * hop + n_fft]) for i in range(n_frames)],
            dim=0,
            memory_config=DRAM,
        )  # [n_frames, n_fft]
        ttnn.deallocate(xp)
        frames = ttnn.to_layout(frames, ttnn.TILE_LAYOUT)

        real = ttnn.matmul(frames, Wcos, compute_kernel_config=compute_config, memory_config=DRAM)
        imag = ttnn.matmul(frames, Wsin, compute_kernel_config=compute_config, memory_config=DRAM)
        ttnn.deallocate(frames)
        spec = ttnn.add(
            ttnn.multiply(real, real, memory_config=DRAM),
            ttnn.multiply(imag, imag, memory_config=DRAM),
            memory_config=DRAM,
        )  # [n_frames, n_freqs], power=2
        ttnn.deallocate(real)
        ttnn.deallocate(imag)
        if power != 2.0:
            spec = ttnn.pow(spec, power / 2.0)

        spec_t = ttnn.permute(spec, (1, 0))              # [n_freqs, n_frames]
        spec_t = ttnn.reshape(spec_t, (1, n_freqs, n_frames))
        mel = mel_scale(spec_t)                          # [1, n_mels, n_frames]
        return mel

    return forward


def mel_spectrogram(*args, **kwargs):
    raise RuntimeError(
        "mel_spectrogram requires build(device, torch_module) to bind the STFT window "
        "and mel filterbank; the bare callable has no parameters."
    )
