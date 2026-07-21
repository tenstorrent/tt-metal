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
# Frames processed per gather+reshape. The [1, nf*512] -> [nf, 512] ROW_MAJOR reshape
# grows circular buffers ~linearly with nf; on p150 (1.5 MB L1) ~180 frames is the
# ceiling, so chunk well under it and concat, letting the frontend handle long audio.
FRAME_CHUNK = 128


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
        # Gather + reshape the frames in chunks so the ROW_MAJOR reshape stays in L1,
        # then concat to the full [T, 512] frame matrix (a no-op single chunk for short
        # audio, so the validated short-audio path is unchanged).
        chunks = []
        for start in range(0, num_frames, FRAME_CHUNK):
            nf = min(FRAME_CHUNK, num_frames - start)
            idx_c = ttnn.slice(idx, [0, start * N_FFT], [1, (start + nf) * N_FFT])  # [1, nf*512] TILE
            g = ttnn.gather(x, dim=1, index=idx_c)  # [1, nf*512] TILE
            g = ttnn.reshape(ttnn.to_layout(g, ttnn.ROW_MAJOR_LAYOUT), [nf, N_FFT])
            chunks.append(ttnn.to_layout(g, ttnn.TILE_LAYOUT))
        framed = chunks[0] if len(chunks) == 1 else ttnn.concat(chunks, dim=0)  # [T, 512] TILE

        spec = ttnn.matmul(framed, self.basis)  # [T, 514]
        real = ttnn.slice(spec, [0, 0], [num_frames, N_FREQS])
        imag = ttnn.slice(spec, [0, N_FREQS], [num_frames, 2 * N_FREQS])
        power = ttnn.add(ttnn.mul(real, real), ttnn.mul(imag, imag))  # [T, 257]

        mel = ttnn.matmul(power, self.mel_fb)  # [T, 257] @ [257, 64] -> [T, 64]
        mel = ttnn.permute(mel, (1, 0))  # [64, T]
        return ttnn.reshape(mel, [1, N_MELS, num_frames])


# ---------------------------------------------------------------------------
# Conditioning mel (``wav_to_mel``) on device — the GPT/perceiver style-embedding mel.
# Same DFT-as-matmul + on-device gather-framing as the speaker frontend above, but with the
# conditioning params (n_fft 2048 / win 1024 hann / hop 256 / 80 mels / fmax 8000), NO
# preemphasis, and a log-clamp + divide-by-mel_norms tail — a faithful port of
# ``reference/xtts_conditioning.wav_to_mel`` so the conditioning input is computed on device
# (removing the last host tensor op). The mel filterbank is librosa htk+slaney, precomputed.
# ---------------------------------------------------------------------------
from models.experimental.xtts.reference.xtts_conditioning import (  # noqa: E402
    MEL_FMAX as C_FMAX,
    MEL_FMIN as C_FMIN,
    MEL_HOP as C_HOP,
    MEL_N_FFT as C_NFFT,
    MEL_SR as C_SR,
    MEL_WIN as C_WIN,
    N_MELS as C_NMELS,
)

C_NFREQS = C_NFFT // 2 + 1  # 1025
C_CENTER_PAD = C_NFFT // 2  # 1024
C_FRAME_CHUNK = 32  # n_fft is 4x the speaker frontend's, so quarter the frame chunk to stay in L1


def _cond_dft_basis():
    """Windowed real-DFT basis ``[C_NFFT, 2*C_NFREQS]`` (hann ``C_WIN`` centered in ``C_NFFT``)."""
    win = torch.zeros(C_NFFT)
    off = (C_NFFT - C_WIN) // 2
    win[off : off + C_WIN] = torch.hann_window(C_WIN, dtype=torch.float32)
    n = torch.arange(C_NFFT).float()
    k = torch.arange(C_NFREQS).float().unsqueeze(1)
    ang = 2 * math.pi * k * n / C_NFFT
    cos_b = torch.cos(ang) * win
    sin_b = -torch.sin(ang) * win
    return torch.cat([cos_b, sin_b], dim=0).t().contiguous()  # [C_NFFT, 2*C_NFREQS]


class TtConditioningMel(LightweightModule):
    """On-device ``wav_to_mel``: waveform ``[1, L, 1]`` (22.05 kHz) -> normalized log-mel
    ``[1, 80, T]`` — the trace-friendly replacement for the host ``wav_to_mel``."""

    def __init__(self, device, mel_norms):
        super().__init__()
        self.device = device
        self.basis = ttnn.from_torch(_cond_dft_basis(), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
        import librosa

        fb = librosa.filters.mel(
            sr=C_SR, n_fft=C_NFFT, n_mels=C_NMELS, fmin=C_FMIN, fmax=C_FMAX, htk=True, norm="slaney"
        )
        self.mel_fb = ttnn.from_torch(
            torch.from_numpy(fb).t().contiguous().float(), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32
        )  # [C_NFREQS, 80]
        self.mel_norms = ttnn.from_torch(
            mel_norms.float().reshape(1, C_NMELS, 1), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32
        )  # [1, 80, 1]
        self._index_cache = {}

    def _frame_index(self, length):
        if length not in self._index_cache:
            num_frames = 1 + length // C_HOP
            m = torch.arange(num_frames).unsqueeze(1)
            n = torch.arange(C_NFFT).unsqueeze(0)
            pos = m * C_HOP + n - C_CENTER_PAD
            idx = _reflect_index(pos, length).reshape(1, -1).to(torch.int32)
            idx_dev = ttnn.from_torch(idx, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.uint32)
            self._index_cache[length] = (idx_dev, num_frames)
        return self._index_cache[length]

    def forward(self, wav):  # wav: ttnn [1, L, 1] ROW_MAJOR fp32
        length = wav.shape[1]
        x = ttnn.to_layout(ttnn.reshape(wav, [1, length]), ttnn.TILE_LAYOUT)  # no preemphasis for the cond mel
        idx, num_frames = self._frame_index(length)
        chunks = []
        for start in range(0, num_frames, C_FRAME_CHUNK):
            nf = min(C_FRAME_CHUNK, num_frames - start)
            idx_c = ttnn.slice(idx, [0, start * C_NFFT], [1, (start + nf) * C_NFFT])
            g = ttnn.gather(x, dim=1, index=idx_c)
            g = ttnn.reshape(ttnn.to_layout(g, ttnn.ROW_MAJOR_LAYOUT), [nf, C_NFFT])
            chunks.append(ttnn.to_layout(g, ttnn.TILE_LAYOUT))
        framed = chunks[0] if len(chunks) == 1 else ttnn.concat(chunks, dim=0)  # [T, C_NFFT]

        spec = ttnn.matmul(framed, self.basis)  # [T, 2*C_NFREQS]
        real = ttnn.slice(spec, [0, 0], [num_frames, C_NFREQS])
        imag = ttnn.slice(spec, [0, C_NFREQS], [num_frames, 2 * C_NFREQS])
        power = ttnn.add(ttnn.mul(real, real), ttnn.mul(imag, imag))  # [T, C_NFREQS]
        mel = ttnn.matmul(power, self.mel_fb)  # [T, 80]
        mel = ttnn.reshape(ttnn.permute(mel, (1, 0)), [1, C_NMELS, num_frames])  # [1, 80, T]
        mel = ttnn.log(ttnn.clamp(mel, 1e-5, 1e30))  # log(clamp(mel, min=1e-5))
        return ttnn.divide(mel, self.mel_norms)  # / mel_norms  (broadcast [1,80,1])
