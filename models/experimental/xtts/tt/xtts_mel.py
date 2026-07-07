# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN XTTS-v2 speaker-encoder mel frontend (``torch_spec``) — fully on-device.

Mirrors ``reference/xtts_mel.py``. ttnn has no FFT, and the STFT-as-strided-conv1d
approach OOMs L1 (a 512-tap kernel blows the CB allocation — the same limit tt_dit
hit with its conv3d STFT). So, like tt_dit's ``MelSTFT``, the DFT is a **matmul**
against a windowed cos/sin basis — but the framing is done **on device** via
``ttnn.gather`` (tt_dit frames on the host with ``unfold``; we avoid that host
round-trip). The gather index map folds in both the frame layout (``m*hop + n``)
and the ``center=True`` reflect padding, so it is a data-independent constant
(function of the sequence length only), not a host op on the activations.

Pipeline (all on device): preemphasis conv -> gather-frame -> DFT matmul (basis
[512, 514]) -> real^2+imag^2 power -> mel-filterbank matmul -> [1, 64, T].
"""

import math

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.xtts.reference.xtts_mel import HOP_LENGTH, N_FFT, N_MELS, PREEMPH, WIN_LENGTH

N_FREQS = N_FFT // 2 + 1  # 257
CENTER_PAD = N_FFT // 2  # 256


def _reflect_index(positions: torch.Tensor, length: int) -> torch.Tensor:
    """torch 'reflect' index map: fold arbitrary integer positions into [0, length)
    by mirroring without repeating the edge (matches F.pad(..., mode='reflect'))."""
    period = 2 * (length - 1)
    r = positions % period  # torch % follows divisor sign -> non-negative
    return torch.where(r < length, r, period - r)


def _dft_basis(window_400: torch.Tensor) -> torch.Tensor:
    """Windowed real-DFT basis ``[N_FFT, 2*N_FREQS]``: columns 0..256 = cos, 257..513
    = -sin, each scaled by the win_length window centered in an N_FFT frame."""
    win = torch.zeros(N_FFT)
    off = (N_FFT - WIN_LENGTH) // 2
    win[off : off + WIN_LENGTH] = window_400
    n = torch.arange(N_FFT).float()
    k = torch.arange(N_FREQS).float().unsqueeze(1)  # [257, 1]
    ang = 2 * math.pi * k * n / N_FFT  # [257, 512]
    cos_b = torch.cos(ang) * win
    sin_b = -torch.sin(ang) * win
    return torch.cat([cos_b, sin_b], dim=0).t().contiguous()  # [512, 514]


class TtMelFrontend(LightweightModule):
    """waveform ``[1, L]`` -> log-free power mel ``[1, 64, T]`` (log lives in the encoder)."""

    def __init__(self, device, ref):
        super().__init__()
        self.device = device
        self.basis = ttnn.from_torch(_dft_basis(ref.window), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
        self.mel_fb = ttnn.from_torch(ref.mel_fb.float(), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
        self._index_cache = {}  # L -> (index tensor [1, T*N_FFT], T)

    def _frame_index(self, length):
        if length not in self._index_cache:
            num_frames = 1 + length // HOP_LENGTH  # center=True frame count
            m = torch.arange(num_frames).unsqueeze(1)  # [T, 1]
            n = torch.arange(N_FFT).unsqueeze(0)  # [1, 512]
            pos = m * HOP_LENGTH + n - CENTER_PAD  # position in the (unpadded) signal
            idx = _reflect_index(pos, length).reshape(1, -1).to(torch.int32)  # [1, T*512]
            idx_dev = ttnn.from_torch(idx, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.uint32)
            self._index_cache[length] = (idx_dev, num_frames)
        return self._index_cache[length]

    def _preemphasize(self, wav):  # wav: [1, L, 1] ROW_MAJOR
        # y[t] = x[t] - 0.97*x[t-1], with reflect at the start (x[-1] -> x[1]).
        # Done as shift-and-subtract (ttnn.conv1d misreads C_in=1 kernels).
        length = wav.shape[1]
        first = ttnn.slice(wav, [0, 1, 0], [1, 2, 1])  # x[1]  (reflect for t=0)
        head = ttnn.slice(wav, [0, 0, 0], [1, length - 1, 1])  # x[0 .. L-2]
        prev = ttnn.concat([first, head], dim=1)  # [1, L, 1] = x[t-1]
        return ttnn.sub(wav, ttnn.mul(prev, PREEMPH))

    def forward(self, wav):  # wav: ttnn [1, L, 1] ROW_MAJOR
        length = wav.shape[1]
        x = self._preemphasize(wav)  # [1, L, 1]
        x = ttnn.to_layout(ttnn.reshape(x, [1, length]), ttnn.TILE_LAYOUT)  # gather needs TILE

        idx, num_frames = self._frame_index(length)
        framed = ttnn.gather(x, dim=1, index=idx)  # [1, T*512] TILE
        framed = ttnn.reshape(ttnn.to_layout(framed, ttnn.ROW_MAJOR_LAYOUT), [num_frames, N_FFT])
        framed = ttnn.to_layout(framed, ttnn.TILE_LAYOUT)

        spec = ttnn.matmul(framed, self.basis)  # [T, 514]
        real = ttnn.slice(spec, [0, 0], [num_frames, N_FREQS])
        imag = ttnn.slice(spec, [0, N_FREQS], [num_frames, 2 * N_FREQS])
        power = ttnn.add(ttnn.mul(real, real), ttnn.mul(imag, imag))  # [T, 257]

        mel = ttnn.matmul(power, self.mel_fb)  # [T, 257] @ [257, 64] -> [T, 64]
        mel = ttnn.permute(mel, (1, 0))  # [64, T]
        return ttnn.reshape(mel, [1, N_MELS, num_frames])
