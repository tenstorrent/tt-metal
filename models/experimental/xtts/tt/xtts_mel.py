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

Pipeline (all on device): preemphasis -> frame -> DFT matmul (basis [512, 514])
-> square -> mel-filterbank matmul -> [1, 64, T]. The ``real^2+imag^2`` reduction
is folded into the filterbank (:func:`_stacked_mel_fb`), so it costs no ops.

PERF (traced, p150; both frontends went ~21-25x in one pass):

  * ``[1, L]``, never ``[1, L, 1]`` — a trailing-1 ROW_MAJOR shape means 4-byte
    pages, and reshaping out of it cost **31.8 ms**, 96% of the conditioning mel.
    See :func:`_flat_signal`. This was by far the largest term.
  * framing without ``ttnn.gather`` — see :class:`_Framer`, the fix that took the
    demo's traced setup from 7.045 s to 0.068 s. Its reshape is chunked to a fixed
    L1 budget (:data:`RESHAPE_PAGE_BUDGET`), so ANY audio length works: unchunked,
    one 4 s conditioning window already overflowed L1.
  * ``real^2+imag^2`` folded into the filterbank — 5 ops become 1.
  * the tail multiplies by a precomputed reciprocal instead of dividing.

Whole-forward traced: speaker 24.19 -> 1.14 ms, conditioning 33.06 -> 1.34 ms,
and the demo's traced setup 0.068 -> 0.012 s.
"""

import math

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.xtts.reference.xtts_mel import HOP_LENGTH, N_FFT, N_MELS, PREEMPH, WIN_LENGTH

N_FREQS = N_FFT // 2 + 1  # 257

# Activations in L1, weights (DFT basis, filterbank, anti-identity) in DRAM — but only from the
# framing on; the [1, L] flat-signal stage stays in DRAM. The blocker is the ROW_MAJOR reshape
# [1, N] -> [num_rows, hop] in _Framer: a ROW_MAJOR page is the LAST dim, so [1, N] is ONE page of
# N*4 bytes (352 KB at the 4 s conditioning chunk) and the reshape's circular buffers are sized by
# it — ~1.48 MB of the 1.5 MB L1. Anything of real size sitting in L1 across that op dies with
# "statically allocated circular buffers clash with L1 buffers". Measured with a 256 KB decoy
# tensor live: the untilize feeding it is fine either way (L1 output included), the reshape fails
# even with a DRAM output. So everything the reshape can see — the preemphasis, the reflect pad,
# xpad, the untilized view — is left in DRAM. From the row slices on, every tensor is [T, *],
# pages across cores normally, and lives in L1.
# ttnn.matmul defaults its output to DRAM (the eltwise ops inherit the input's config instead), so
# the matmuls need this passed explicitly or they drag the chain back out to DRAM.
L1 = ttnn.L1_MEMORY_CONFIG

# L1 budget for ONE framing reshape's input page, which is what caps the audio length a single
# reshape can take and therefore why _Framer chunks. The reshape's circular buffers come out at
# ~4.3x the page (page = rows*hop*4 bytes), measured on p150 by growing the signal until the clash:
#
#   conditioning (hop 256)  3.50 s  309 KB page  OK      |  4.00 s  352 KB page  CBs 1.52 MB  CLASH
#   speaker      (hop 160)  4.00 s  252 KB page  OK      |  8.00 s  502 KB page  CBs 2.17 MB  CLASH
#
# L1 is 1.5 MB, so a whole 4 s conditioning window (coqui's gpt_cond_chunk_len) does not fit, and a
# reference longer than ~4 s does not fit the speaker frontend either — both fatal for long
# reference audio, and the 4 s case fails even though it "just fits" at 1.52 MB, because any live L1
# buffer clashes with the CB region. 192 KB keeps the CBs near 0.83 MB, i.e. ~0.7 MB of L1 left for
# whatever the caller has live.
RESHAPE_PAGE_BUDGET = 192 * 1024


def _flat_signal(wav, length):
    """Accept the waveform as ``[1, L]`` (what callers should pass) or the legacy ``[1, L, 1]``.

    PASS ``[1, L]``. A ROW_MAJOR tensor's PAGE is its last dim, so ``[1, L, 1]`` is L pages of
    ONE fp32 each — 4-byte pages — and reshaping that to ``[1, L]`` (a single 274 KB page) has to
    repack all L of them. Measured on p150 at L=68679: the reshape alone is **31.8 ms**, which
    was 96% of the conditioning mel and ~45% of the entire traced setup. The tilize everyone
    suspects is innocent: ``to_layout(TILE)`` of an already-``[1, L]`` tensor is 0.130 ms.

    Placing the waveform as ``[1, L]`` in the first place costs nothing — it is a host-side
    ``reshape`` before ``from_torch`` — and makes this a no-op view. The rank-3 branch is kept
    only so older callers/tests still work; it is the slow path, not the intended one.
    """
    return wav if len(wav.shape) == 2 else ttnn.reshape(wav, [1, length])


class _Framer:
    """Signal ``[1, L]`` -> STFT frame matrix ``[T, n_fft]`` (``center=True`` reflect padding).

    Framing used to be a ``ttnn.gather`` per chunk over a ``[1, chunk*n_fft]`` index map, and it
    was **84% of the whole demo's traced replay time** — 5.77 s for the conditioning mel and
    1.21 s for the speaker mel, against 0.19 ms for the DFT matmul they feed.

    The reason is that ``ttnn.gather`` is QUADRATIC in the audio length on this shape. A general
    gather cannot assume anything about its indices — any output element may reference any input
    element — and ``SingleRowMultiCore`` (the factory selected here, since ``Wt_input`` 2147 > the
    ``GATHER_WT_THRESHOLD`` of 60) gives each core only a TWO-tile input CB. So every output tile
    streams the whole ``Wt_input``-tile signal row past it. Cost is the PRODUCT of the two tile
    counts; measured on p150 and linear in each factor independently over a 32x range::

        t ~= 156 ns * Wt_index * Wt_input          (Wt_index = T*n_fft/32, Wt_input = L/32)

    which predicts every observed number to within ~1%: cond 17216*2147 -> 5.75 s (measured
    5.77), speaker 4992*1558 -> 1.21 s (measured 1.21). Both tile counts grow with the audio
    length, hence quadratic. NOTE the chunking was NOT the problem and removing it is not the
    fix: total ``Wt_index`` is the same however you slice it, so one unchunked call measures
    5.68 s against the chunked 5.77 s — the chunk loop cost only ~7%, from padding the last chunk.

    Framing needs no indexing at all — which removes the ``Wt_input`` factor entirely and makes
    the cost linear in the OUTPUT size. Consecutive frames start exactly ``hop`` samples apart, so
    reflect-pad the signal once, view it as ``[rows, hop]`` (a free ROW_MAJOR reshape) and frame
    ``m`` is simply rows ``m .. m+ceil(n_fft/hop)-1`` read end to end. So the frame matrix is
    ``ceil(n_fft/hop)`` CONTIGUOUS row slices of that view, concatenated along the columns and
    trimmed to ``n_fft``: ``wide[m, j] == xpad[m*hop + j]`` by construction. All bulk copies, no
    per-element indexing. The framing itself is bit-exact — 0.0 max abs err against a host
    reference, given the same padded signal.

    The reshape is done in CHUNKS of ``frame_chunk`` frames (see ``RESHAPE_PAGE_BUDGET``), because
    its circular buffers are sized by the padded signal's single ROW_MAJOR page and one whole 4 s
    conditioning window already exceeds L1. Chunk boundaries fall on frames and consecutive chunks
    re-read the ``rows_per_frame - 1`` rows they overlap on, so this is invisible in the result.

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
        self.rows_per_frame = -(-n_fft // hop)
        # Frames per reshape (see RESHAPE_PAGE_BUDGET). A multiple of 32 so every chunk but the
        # trailing one is tile-aligned in height, which keeps the dim-0 concat on whole tiles.
        rows_budget = max(self.rows_per_frame, RESHAPE_PAGE_BUDGET // (hop * 4))
        self.frame_chunk = max(32, ((rows_budget - self.rows_per_frame + 1) // 32) * 32)
        self._rev_cache = {}  # pad length p -> anti-identity [p, p]
        self._geom_cache = {}  # signal length L -> geometry tuple

    def num_frames(self, length: int) -> int:
        return 1 + length // self.hop  # center=True frame count

    def _geometry(self, length: int):
        """``(T, rows_per_frame, num_rows, right_pad)`` for a signal of ``length`` samples."""
        if length not in self._geom_cache:
            frames = self.num_frames(length)
            rows_per_frame = self.rows_per_frame
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

        # Frame in chunks of at most self.frame_chunk frames, then concat along dim 0. The chunking
        # is purely an L1 budget for the reshape (see RESHAPE_PAGE_BUDGET) — every chunk boundary
        # falls on a frame, and consecutive chunks re-read the rows_per_frame-1 rows they overlap on,
        # so the result is identical to framing in one go (verified bit-exact, 0.0 max abs err).
        pieces = [
            self._frame_chunk(xpad, start, min(self.frame_chunk, frames - start), rows_per_frame)
            for start in range(0, frames, self.frame_chunk)
        ]
        if len(pieces) == 1:
            return pieces[0]
        # dim-0 concat: the LAST dim is n_fft, a multiple of 32, so this is fp32-exact even when the
        # trailing chunk's mid-tile height sends it down the untilize/retilize path (see the concat
        # note above; that path is cheap here because a row stick is only n_fft*4 bytes).
        out = ttnn.concat(pieces, dim=0, memory_config=L1)
        for p in pieces:
            ttnn.deallocate(p)
        return out

    def _frame_chunk(self, xpad, start, nf, rows_per_frame):
        """Frames ``start .. start+nf`` of the padded signal as ``[nf, n_fft]`` TILE fp32.

        Only the ``(nf + rows_per_frame - 1)`` rows those frames span are reshaped, which is what
        bounds the reshape's L1: its input is a SINGLE ROW_MAJOR page of ``rows*hop*4`` bytes and
        its circular buffers come out at ~4.3x that (see RESHAPE_PAGE_BUDGET). The slice off ``xpad``
        is taken in TILE layout, where ``start*hop`` and the width are tile-aligned (``hop`` is a
        multiple of 32 in both frontends), so it is a whole-tile copy.

        DRAM, not L1, up to and including the reshape — see the L1 note at the top of the file.
        """
        hop = self.hop
        rows_needed = nf + rows_per_frame - 1
        seg = ttnn.slice(xpad, [0, start * hop], [1, (start + rows_needed) * hop])  # [1, rows*hop]
        rows = ttnn.reshape(ttnn.to_layout(seg, ttnn.ROW_MAJOR_LAYOUT), [rows_needed, hop])
        ttnn.deallocate(seg)
        blocks = [ttnn.slice(rows, [k, 0], [k + nf, hop], memory_config=L1) for k in range(rows_per_frame)]
        ttnn.deallocate(rows)
        wide = ttnn.concat(blocks, dim=1, memory_config=L1) if len(blocks) > 1 else blocks[0]
        if len(blocks) > 1:
            for b in blocks:
                ttnn.deallocate(b)
        if wide.shape[1] != self.n_fft:  # hop does not divide n_fft (speaker: 4x160 -> trim to 512)
            trimmed = ttnn.slice(wide, [0, 0], [nf, self.n_fft], memory_config=L1)
            ttnn.deallocate(wide)
            wide = trimmed
        return ttnn.to_layout(wide, ttnn.TILE_LAYOUT, memory_config=L1)


def _stacked_mel_fb(mel_fb: torch.Tensor) -> torch.Tensor:
    """``[N_FREQS, N_MELS]`` filterbank -> ``[2*N_FREQS, N_MELS]``, folding the power reduction in.

    The mel is ``mel[:, j] = sum_k (real[:, k]^2 + imag[:, k]^2) * fb[k, j]`` and the DFT matmul
    already emits ``[real | imag]`` side by side, so squaring the WHOLE spectrum and matmul'ing it
    against ``vstack(fb, fb)`` reproduces that sum exactly — the two halves land on the same output
    column by construction. That replaces ``slice + slice + mul + mul + add`` with a single
    ``ttnn.square``: 5 ops become 1, at the cost of doubling the matmul's K (which is cheap — the
    matmul is 0.029 ms against 0.138 ms for the chain it removes).
    """
    return torch.cat([mel_fb, mel_fb], dim=0).contiguous()


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
        # See _stacked_mel_fb: the real^2+imag^2 reduction is folded into the filterbank.
        self.mel_fb = ttnn.from_torch(
            _stacked_mel_fb(ref.mel_fb.float()), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32
        )  # [2*N_FREQS, N_MELS]
        self.framer = _Framer(device, N_FFT, HOP_LENGTH)

    def _preemphasize(self, x, length):  # x: [1, L] ROW_MAJOR
        # y[t] = x[t] - 0.97*x[t-1], with reflect at the start (x[-1] -> x[1]).
        # Done as shift-and-subtract (ttnn.conv1d misreads C_in=1 kernels).
        first = ttnn.slice(x, [0, 1], [1, 2])  # x[1]  (reflect for t=0)
        head = ttnn.slice(x, [0, 0], [1, length - 1])  # x[0 .. L-2]
        prev = ttnn.concat([first, head], dim=1)  # [1, L] = x[t-1]
        return ttnn.sub(x, ttnn.mul(prev, PREEMPH))

    def forward(self, wav):  # wav: ttnn [1, L] ROW_MAJOR (see _flat_signal)
        length = wav.shape[1]
        x = self._preemphasize(_flat_signal(wav, length), length)  # [1, L] ROW_MAJOR
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)  # DRAM: one-row-wide (un)tilize, see _Framer

        num_frames = self.framer.num_frames(length)
        framed = self.framer(x, length)  # [T, 512] TILE

        spec = ttnn.matmul(framed, self.basis, memory_config=L1)  # [T, 514] = [real | imag]
        # real^2+imag^2 is folded into the filterbank (see _stacked_mel_fb), so one square + one
        # matmul replaces slice/slice/mul/mul/add/matmul.
        mel = ttnn.matmul(ttnn.square(spec, memory_config=L1), self.mel_fb, memory_config=L1)  # [T, 514] -> [T, 64]
        mel = ttnn.permute(mel, (1, 0), memory_config=L1)  # [64, T]
        return ttnn.reshape(mel, [1, N_MELS, num_frames], memory_config=L1)


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
        self.mel_fb = ttnn.from_torch(  # [2*C_NFREQS, 80] — power reduction folded in
            _stacked_mel_fb(torch.from_numpy(fb).t().contiguous().float()),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            dtype=ttnn.float32,
        )
        # RECIPROCAL of mel_norms: the tail divides by it, and a multiply is cheaper than a
        # divide for the same result. Reciprocated on host in fp64 so it costs no accuracy.
        self.mel_norms_recip = ttnn.from_torch(
            (1.0 / mel_norms.double()).float().reshape(1, C_NMELS, 1),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            dtype=ttnn.float32,
        )  # [1, 80, 1]
        self.framer = _Framer(device, C_NFFT, C_HOP)

    def forward(self, wav):  # wav: ttnn [1, L] ROW_MAJOR fp32 (see _flat_signal)
        length = wav.shape[1]
        # no preemphasis for the cond mel
        x = ttnn.to_layout(_flat_signal(wav, length), ttnn.TILE_LAYOUT)  # DRAM, see _Framer
        num_frames = self.framer.num_frames(length)
        framed = self.framer(x, length)  # [T, C_NFFT]

        spec = ttnn.matmul(framed, self.basis, memory_config=L1)  # [T, 2*C_NFREQS] = [real | imag]
        # real^2+imag^2 folded into the filterbank (see _stacked_mel_fb).
        mel = ttnn.matmul(ttnn.square(spec, memory_config=L1), self.mel_fb, memory_config=L1)  # -> [T, 80]
        mel = ttnn.reshape(ttnn.permute(mel, (1, 0), memory_config=L1), [1, C_NMELS, num_frames], memory_config=L1)
        mel = ttnn.log(ttnn.clamp(mel, 1e-5, 1e30, memory_config=L1), memory_config=L1)  # log(clamp(mel, min=1e-5))
        return ttnn.multiply(mel, self.mel_norms_recip, memory_config=L1)  # / mel_norms, as a multiply
