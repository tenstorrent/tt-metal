# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.xtts.reference.xtts_mel import HOP_LENGTH, N_FFT, N_MELS, PREEMPH, WIN_LENGTH

N_FREQS = N_FFT // 2 + 1  # 257

# Flat-signal stage stays DRAM: ROW_MAJOR reshape [1,N]->[rows,hop] sizes CBs off the single
# page and clashes with any live L1 buffer. Pass L1 explicitly on matmuls (default is DRAM).
L1 = ttnn.L1_MEMORY_CONFIG

# Caps one framing reshape's input page (~4.3x becomes CB use); whole 4s cond window OOMs L1.
RESHAPE_PAGE_BUDGET = 192 * 1024


def _flat_signal(wav, length):
    # Prefer [1, L]: [1, L, 1] ROW_MAJOR uses 4-byte pages and a costly reshape to flatten.
    """Flatten waveform to [1, L] for framing."""
    return wav if len(wav.shape) == 2 else ttnn.reshape(wav, [1, length])


class _Framer:
    def __init__(self, device, n_fft: int, hop: int):
        """Initialize STFT framer caches and chunk sizing."""
        self.device = device
        self.n_fft = n_fft
        self.hop = hop
        self.center_pad = n_fft // 2
        self.rows_per_frame = -(-n_fft // hop)
        # Multiple of 32 so non-trailing chunks stay tile-aligned for dim-0 concat.
        rows_budget = max(self.rows_per_frame, RESHAPE_PAGE_BUDGET // (hop * 4))
        self.frame_chunk = max(32, ((rows_budget - self.rows_per_frame + 1) // 32) * 32)
        self._rev_cache = {}
        self._geom_cache = {}

    def num_frames(self, length: int) -> int:
        """Return number of frames for a signal length."""
        return 1 + length // self.hop

    def _geometry(self, length: int):
        """Cache reflect-pad geometry for a signal length."""
        if length not in self._geom_cache:
            frames = self.num_frames(length)
            rows_per_frame = self.rows_per_frame
            num_rows = (frames - 1) + rows_per_frame
            right = num_rows * self.hop - self.center_pad - length
            assert 0 <= right < length, (
                f"signal of {length} samples is too short for n_fft={self.n_fft}: the right "
                f"reflect-pad would need {right} samples, more than one mirror can supply"
            )
            self._geom_cache[length] = (frames, rows_per_frame, num_rows, right)
        return self._geom_cache[length]

    def _anti_identity(self, p: int):
        """Build or fetch an anti-diagonal identity for reflect pad."""
        if p not in self._rev_cache:
            self._rev_cache[p] = ttnn.from_torch(
                torch.flip(torch.eye(p), dims=[0]), layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.float32
            )
        return self._rev_cache[p]

    def __call__(self, x, length: int):
        """Center-pad and frame a waveform into STFT windows."""
        frames, rows_per_frame, num_rows, right = self._geometry(length)
        cp = self.center_pad
        parts = [ttnn.matmul(ttnn.slice(x, [0, 1], [1, cp + 1]), self._anti_identity(cp)), x]
        if right:
            parts.append(
                ttnn.matmul(ttnn.slice(x, [0, length - 1 - right], [1, length - 1]), self._anti_identity(right))
            )
        # ttnn.concat is fp32-exact only when last dims are multiples of 32; else rounds to bf16.
        xpad = ttnn.concat(parts, dim=1)

        pieces = [
            self._frame_chunk(xpad, start, min(self.frame_chunk, frames - start), rows_per_frame)
            for start in range(0, frames, self.frame_chunk)
        ]
        if len(pieces) == 1:
            return pieces[0]
        out = ttnn.concat(pieces, dim=0, memory_config=L1)
        for p in pieces:
            ttnn.deallocate(p)
        return out

    def _frame_chunk(self, xpad, start, nf, rows_per_frame):
        """Frame one chunk of the padded signal into windows."""
        hop = self.hop
        rows_needed = nf + rows_per_frame - 1
        seg = ttnn.slice(xpad, [0, start * hop], [1, (start + rows_needed) * hop])
        rows = ttnn.reshape(ttnn.to_layout(seg, ttnn.ROW_MAJOR_LAYOUT), [rows_needed, hop])
        ttnn.deallocate(seg)
        blocks = [ttnn.slice(rows, [k, 0], [k + nf, hop], memory_config=L1) for k in range(rows_per_frame)]
        ttnn.deallocate(rows)
        wide = ttnn.concat(blocks, dim=1, memory_config=L1) if len(blocks) > 1 else blocks[0]
        if len(blocks) > 1:
            for b in blocks:
                ttnn.deallocate(b)
        if wide.shape[1] != self.n_fft:  # hop may not divide n_fft (speaker: 4x160 -> trim to 512)
            trimmed = ttnn.slice(wide, [0, 0], [nf, self.n_fft], memory_config=L1)
            ttnn.deallocate(wide)
            wide = trimmed
        return ttnn.to_layout(wide, ttnn.TILE_LAYOUT, memory_config=L1)


def _stacked_mel_fb(mel_fb: torch.Tensor) -> torch.Tensor:
    """Stack mel filterbank for real/imag magnitude path."""
    return torch.cat([mel_fb, mel_fb], dim=0).contiguous()


def _dft_basis(window_400: torch.Tensor) -> torch.Tensor:
    """Build windowed DFT cosine/sine basis for speaker mel."""
    win = torch.zeros(N_FFT)
    off = (N_FFT - WIN_LENGTH) // 2
    win[off : off + WIN_LENGTH] = window_400
    n = torch.arange(N_FFT).float()
    k = torch.arange(N_FREQS).float().unsqueeze(1)
    ang = 2 * math.pi * k * n / N_FFT
    cos_b = torch.cos(ang) * win
    sin_b = -torch.sin(ang) * win
    return torch.cat([cos_b, sin_b], dim=0).t().contiguous()


class TtMelFrontend(LightweightModule):
    def __init__(self, device, ref):
        """Load speaker-mel DFT and filterbank tensors."""
        super().__init__()
        self.device = device
        self.basis = ttnn.from_torch(_dft_basis(ref.window), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
        self.mel_fb = ttnn.from_torch(
            _stacked_mel_fb(ref.mel_fb.float()), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32
        )
        self.framer = _Framer(device, N_FFT, HOP_LENGTH)

    def _preemphasize(self, x, length):
        # Shift-and-subtract: ttnn.conv1d misreads C_in=1 kernels.
        """Apply first-order pre-emphasis to the waveform."""
        first = ttnn.slice(x, [0, 1], [1, 2])
        head = ttnn.slice(x, [0, 0], [1, length - 1])
        prev = ttnn.concat([first, head], dim=1)
        return ttnn.sub(x, ttnn.mul(prev, PREEMPH))

    def forward(self, wav):
        """Compute speaker-encoder mel spectrogram from wav."""
        length = wav.shape[1]
        x = self._preemphasize(_flat_signal(wav, length), length)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        num_frames = self.framer.num_frames(length)
        framed = self.framer(x, length)

        spec = ttnn.matmul(framed, self.basis, memory_config=L1)
        mel = ttnn.matmul(ttnn.square(spec, memory_config=L1), self.mel_fb, memory_config=L1)
        mel = ttnn.permute(mel, (1, 0), memory_config=L1)
        return ttnn.reshape(mel, [1, N_MELS, num_frames], memory_config=L1)


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


def _cond_dft_basis():
    """Build windowed DFT basis for conditioning mel."""
    win = torch.zeros(C_NFFT)
    off = (C_NFFT - C_WIN) // 2
    win[off : off + C_WIN] = torch.hann_window(C_WIN, dtype=torch.float32)
    n = torch.arange(C_NFFT).float()
    k = torch.arange(C_NFREQS).float().unsqueeze(1)
    ang = 2 * math.pi * k * n / C_NFFT
    cos_b = torch.cos(ang) * win
    sin_b = -torch.sin(ang) * win
    return torch.cat([cos_b, sin_b], dim=0).t().contiguous()


class TtConditioningMel(LightweightModule):
    def __init__(self, device, mel_norms):
        """Load conditioning-mel basis, filterbank, and norms."""
        super().__init__()
        self.device = device
        self.basis = ttnn.from_torch(_cond_dft_basis(), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
        import librosa

        fb = librosa.filters.mel(
            sr=C_SR, n_fft=C_NFFT, n_mels=C_NMELS, fmin=C_FMIN, fmax=C_FMAX, htk=True, norm="slaney"
        )
        self.mel_fb = ttnn.from_torch(
            _stacked_mel_fb(torch.from_numpy(fb).t().contiguous().float()),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            dtype=ttnn.float32,
        )
        # Reciprocal so the tail is a multiply (cheaper than divide); recip in fp64 on host.
        self.mel_norms_recip = ttnn.from_torch(
            (1.0 / mel_norms.double()).float().reshape(1, C_NMELS, 1),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            dtype=ttnn.float32,
        )
        self.framer = _Framer(device, C_NFFT, C_HOP)

    def forward(self, wav):
        """Compute log-normalized conditioning mel from wav."""
        length = wav.shape[1]
        x = ttnn.to_layout(_flat_signal(wav, length), ttnn.TILE_LAYOUT)
        num_frames = self.framer.num_frames(length)
        framed = self.framer(x, length)

        spec = ttnn.matmul(framed, self.basis, memory_config=L1)
        mel = ttnn.matmul(ttnn.square(spec, memory_config=L1), self.mel_fb, memory_config=L1)
        mel = ttnn.reshape(ttnn.permute(mel, (1, 0), memory_config=L1), [1, C_NMELS, num_frames], memory_config=L1)
        mel = ttnn.log(ttnn.clamp(mel, 1e-5, 1e30, memory_config=L1), memory_config=L1)
        return ttnn.multiply(mel, self.mel_norms_recip, memory_config=L1)
