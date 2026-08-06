# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN XTTS-v2 speaker-encoder mel frontend (``torch_spec``) — fully on-device.

Mirrors ``reference/xtts_mel.py``. ttnn has no FFT, and the STFT-as-strided-conv1d
approach OOMs L1 (a 512-tap kernel blows the CB allocation — the same limit tt_dit
hit with its conv3d STFT). So, like tt_dit's ``MelSTFT``, the DFT is a **matmul**
against a windowed cos/sin basis — and the framing is done **on device** too
(tt_dit frames on the host with ``unfold``; we avoid that host round-trip), as a
reflect-pad plus contiguous row slices of a ``[rows, hop]`` view of the padded
signal — see :class:`_Framer`, which replaced a ``ttnn.gather`` that was 84% of
the demo's whole traced replay time.

Pipeline (all on device): preemphasis conv -> frame -> DFT matmul (basis
[512, 514]) -> real^2+imag^2 power -> mel-filterbank matmul -> [1, 64, T].
"""

import math

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.xtts.reference.xtts_mel import HOP_LENGTH, N_FFT, N_MELS, PREEMPH, WIN_LENGTH

N_FREQS = N_FFT // 2 + 1  # 257
CENTER_PAD = N_FFT // 2  # 256


class _Framer:
    """Signal ``[1, L]`` -> STFT frame matrix ``[T, n_fft]`` (``center=True`` reflect padding).

    Framing used to be a ``ttnn.gather`` per chunk over a ``[1, chunk*n_fft]`` index map, and it
    was **84% of the whole demo's traced replay time** — 5.77 s for the conditioning mel and
    1.21 s for the speaker mel, against 0.19 ms for the DFT matmul they feed. The reason is the
    gather kernel's work split: ``gather_program_factory`` splits over ``Ht``, the index tensor's
    TILE-ROW count, and a flat ``[1, chunk*n_fft]`` index map is ONE tile row — so the whole
    gather runs on a single core, re-reading the full ``Wt_input``-tile signal row for every
    output tile. 684 ms per 32-frame chunk. Chunking made it worse, not better: the cost is
    per-chunk single-core work, so 9 chunks paid it 9 times.

    Framing needs no indexing at all. Consecutive frames start exactly ``hop`` samples apart, so
    reflect-pad the signal once, view it as ``[rows, hop]`` (a free ROW_MAJOR reshape) and frame
    ``m`` is simply rows ``m .. m+ceil(n_fft/hop)-1`` read end to end. So the frame matrix is
    ``ceil(n_fft/hop)`` CONTIGUOUS row slices of that view, concatenated along the columns and
    trimmed to ``n_fft``: ``wide[m, j] == xpad[m*hop + j]`` by construction. All bulk copies, no
    per-element indexing. The framing itself is bit-exact — 0.0 max abs err against a host
    reference, given the same padded signal.

    Framing alone: 5770 -> 1.0 ms at the conditioning shape. Whole frontend, in model:
    conditioning mel 5801 -> 33 ms, speaker mel 1234 -> 25 ms, at PCC 0.9999993 and 0.99999996
    against their torch references. Whole demo: 7.045 -> 0.068 s of traced setup, RTF 1.15 -> 0.19.

    The one part that is not a copy is the reflect padding itself, which reverses a prefix and a
    suffix of the signal. ttnn has no ``flip``, and a small ``ttnn.gather`` over just those
    samples hits the same single-core path (35 ms), so the reversal is a matmul against a
    constant anti-identity — 1.1 ms for both pads, at the cost of one ``[p, p]`` fp32 constant
    per distinct pad length (4 MB at n_fft=2048). That matmul carries bf16-level rounding (~2e-3
    relative) on the padded samples and no compute_kernel_config fixes it (HiFi4 + fp32_dest_acc
    measures identically), but it only touches the ``n_fft/2`` samples at each end and the PCCs
    above are measured with it in place.
    """

    def __init__(self, device, n_fft: int, hop: int):
        self.device = device
        self.n_fft = n_fft
        self.hop = hop
        self.center_pad = n_fft // 2
        self._rev_cache = {}  # pad length p -> anti-identity [p, p]
        self._geom_cache = {}  # signal length L -> geometry tuple

    def num_frames(self, length: int) -> int:
        return 1 + length // self.hop  # center=True frame count

    def _geometry(self, length: int):
        """``(T, rows_per_frame, num_rows, right_pad)`` for a signal of ``length`` samples."""
        if length not in self._geom_cache:
            frames = self.num_frames(length)
            rows_per_frame = -(-self.n_fft // self.hop)
            num_rows = (frames - 1) + rows_per_frame
            right = num_rows * self.hop - self.center_pad - length
            # A single reflection can only mirror `length - 1` samples. Signals shorter than
            # ~n_fft would need the gather path's modular (repeated) reflection; fail loudly
            # rather than silently mis-pad. Every real input here is seconds of audio.
            assert 0 <= right < length, (
                f"signal of {length} samples is too short for n_fft={self.n_fft}: the right "
                f"reflect-pad would need {right} samples, more than one mirror can supply"
            )
            self._geom_cache[length] = (frames, rows_per_frame, num_rows, right)
        return self._geom_cache[length]

    def _anti_identity(self, p: int):
        """Constant ``[p, p]`` reversal matrix (``x @ J`` = ``x`` reversed)."""
        if p not in self._rev_cache:
            self._rev_cache[p] = ttnn.from_torch(
                torch.flip(torch.eye(p), dims=[0]), layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.float32
            )
        return self._rev_cache[p]

    def __call__(self, x, length: int):
        """``x``: ttnn ``[1, length]`` TILE fp32. Returns ``[T, n_fft]`` TILE fp32."""
        frames, rows_per_frame, num_rows, right = self._geometry(length)
        cp = self.center_pad
        # Reflect pad: xpad[p] == x[reflect(p - center_pad)]. Left is x[1..cp] reversed, right is
        # x[L-2 .. L-1-right] reversed -- both a reversal of a plain slice, hence the anti-identity.
        parts = [ttnn.matmul(ttnn.slice(x, [0, 1], [1, cp + 1]), self._anti_identity(cp)), x]
        if right:
            parts.append(
                ttnn.matmul(ttnn.slice(x, [0, length - 1 - right], [1, length - 1]), self._anti_identity(right))
            )
        # NOTE: ttnn.concat is fp32-exact only when every input's last dim is a multiple of 32;
        # with a mid-tile width (which `length` is, in general) it falls back to a path that
        # rounds the WHOLE result to bf16 precision, pass-through regions included. Measured:
        # tile-aligned widths 0.0 err, unaligned 3.1e-3. The column concat below is aligned
        # (every block is `hop` wide) and so is exact; only this pad concat takes the hit.
        xpad = ttnn.concat(parts, dim=1)  # [1, num_rows*hop]

        # [rows, hop] view -> frame m is rows m .. m+rows_per_frame-1, so each column block is one
        # contiguous row slice. Row slices + a column concat, no indexing.
        rows = ttnn.reshape(ttnn.to_layout(xpad, ttnn.ROW_MAJOR_LAYOUT), [num_rows, self.hop])
        blocks = [ttnn.slice(rows, [k, 0], [k + frames, self.hop]) for k in range(rows_per_frame)]
        wide = ttnn.concat(blocks, dim=1) if len(blocks) > 1 else blocks[0]
        if wide.shape[1] != self.n_fft:  # hop does not divide n_fft (speaker: 4x160 -> trim to 512)
            wide = ttnn.slice(wide, [0, 0], [frames, self.n_fft])
        return ttnn.to_layout(wide, ttnn.TILE_LAYOUT)


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
        self.framer = _Framer(device, N_FFT, HOP_LENGTH)

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
        x = ttnn.to_layout(ttnn.reshape(x, [1, length]), ttnn.TILE_LAYOUT)

        num_frames = self.framer.num_frames(length)
        framed = self.framer(x, length)  # [T, 512] TILE

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
        self.framer = _Framer(device, C_NFFT, C_HOP)

    def forward(self, wav):  # wav: ttnn [1, L, 1] ROW_MAJOR fp32
        length = wav.shape[1]
        x = ttnn.to_layout(ttnn.reshape(wav, [1, length]), ttnn.TILE_LAYOUT)  # no preemphasis for the cond mel
        num_frames = self.framer.num_frames(length)
        framed = self.framer(x, length)  # [T, C_NFFT]

        spec = ttnn.matmul(framed, self.basis)  # [T, 2*C_NFREQS]
        real = ttnn.slice(spec, [0, 0], [num_frames, C_NFREQS])
        imag = ttnn.slice(spec, [0, C_NFREQS], [num_frames, 2 * C_NFREQS])
        power = ttnn.add(ttnn.mul(real, real), ttnn.mul(imag, imag))  # [T, C_NFREQS]
        mel = ttnn.matmul(power, self.mel_fb)  # [T, 80]
        mel = ttnn.reshape(ttnn.permute(mel, (1, 0)), [1, C_NMELS, num_frames])  # [1, 80, T]
        mel = ttnn.log(ttnn.clamp(mel, 1e-5, 1e30))  # log(clamp(mel, min=1e-5))
        return ttnn.divide(mel, self.mel_norms)  # / mel_norms  (broadcast [1,80,1])
