# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""HiFT at a padded length that reproduces upstream's call at the real length (docs/VALIDATION.md, "Masked end padding").

A bucketed HiFT call runs `run` mel frames of which only the first `valid` are real. Upstream runs the real frames
alone: every conv sees zeros past the end (its own zero padding), the STFT reflects the source at its end, and the
iSTFT normalizes by the windows of the real frames only. HiFT looks ahead, so any padded content reaches back into
the last real frames; padding with silence mel silenced the last ~25 ms of every utterance (notes: B28).

The padded call matches upstream's when:
- the mel input, every conv's output and each stage's input sum are zeroed past the real length at their own rate
  (mel frames, x8, x40, then the STFT frames, 120 per mel frame + 1). Snake, leaky ReLU and ELU keep zeros at zero,
  so a zeroed tensor stays zeroed until the next conv;
- f0 is zero past the real frames. SineGen2's phase is a cumsum of the per-frame radians, interpolated back up
  linearly, and upstream's interpolation clamps at the last real frame; a zero f0 past it keeps the phase flat
  there, which is the same thing;
- the 8 source samples after the real end are the reflection of the last real ones (`torch.stft`'s centering), and
  the STFT frames past the real ones are zeroed;
- the iSTFT's magnitude is zeroed past the real frames, and its last few output samples are rescaled from the
  padded call's window normalization to the real one's (`istft_end_gain`).

The first streaming call is padded in front and is not affected: its front is the utterance's start.
"""
from __future__ import annotations

import numpy as np
import torch

import ttnn


def stft_frames(mel_frames: int, upsample_scale: int, hop: int) -> int:
    """`torch.stft(center=True)` frames for `mel_frames` of source: `mel_frames * upsample_scale / hop + 1`."""
    return mel_frames * upsample_scale // hop + 1


def mask_1d(length: int, valid: int) -> np.ndarray:
    """1.0 for the first `valid` positions of `length`, 0.0 after."""
    assert 0 < valid <= length, (valid, length)
    m = np.zeros(length, dtype=np.float32)
    m[:valid] = 1.0
    return m


def reflect_source_end(source, valid_samples: int, pad: int):
    """In place, on a host source `[..., L]`: the `pad` samples after `valid_samples` become the reflection of the
    real ones, as `torch.stft(center=True)` pads the end of a signal of `valid_samples` (the edge sample itself is
    not repeated). Returns `source`."""
    if valid_samples + pad <= source.shape[-1]:
        for k in range(pad):
            source[..., valid_samples + k] = source[..., valid_samples - 2 - k]
    return source


def _envelope_at(positions, n_frames: int, hop: int, sq: np.ndarray) -> np.ndarray:
    """The overlap-added window^2 (what `torch.istft` divides by) at `positions` of the uncropped output, for
    `n_frames` frames: frame j covers [hop * j, hop * j + n_fft)."""
    n_fft = len(sq)
    out = np.zeros(len(positions), dtype=np.float64)
    for i, p in enumerate(positions):
        for j in range(max(0, -(-(p - n_fft + 1) // hop)), min(n_frames - 1, p // hop) + 1):
            out[i] += sq[p - hop * j]
    return out


def istft_end_gain(valid_frames: int, run_frames: int, upsample_scale: int, n_fft: int, hop: int, window):
    """The output samples whose `torch.istft(center=True)` normalization differs between `run_frames` and
    `valid_frames` of mel (a padded frame's window overlaps the last real samples), and the factor that turns the
    padded call's value into the real call's there. Only the last `n_fft` samples can differ. Returns
    `(index, gain)`; `index` counts from the start of the waveform, and both are empty when nothing differs."""
    sq = np.asarray(window, dtype=np.float64) ** 2
    valid_len = valid_frames * upsample_scale
    index = np.arange(max(0, valid_len - n_fft), valid_len)
    positions = index + n_fft // 2  # center=True crops n_fft // 2 from the front
    env_run = _envelope_at(positions, stft_frames(run_frames, upsample_scale, hop), hop, sq)
    env_valid = _envelope_at(positions, stft_frames(valid_frames, upsample_scale, hop), hop, sq)
    keep = env_run != env_valid
    return index[keep], (env_run[keep] / env_valid[keep]).astype(np.float32)


class ValidMasks:
    """Device masks for one padded HiFT call: 1.0 over the real part of each length, 0.0 after, shaped to broadcast
    over the channels of a channels-last `[1, L, C]` tensor (`[1, L, 1]`), or over the bins of a `[1, bins, L]` one
    (`[1, 1, L]`). Made on the host per call (their content depends on the real length, their shape only on the
    bucket), so the programs stay those of the bucket. Free them with `release()` once the call is done."""

    def __init__(self, device):
        self.device = device
        self._made: dict[tuple, ttnn.Tensor] = {}

    def get(self, length: int, valid: int, dtype, across: str = "channels") -> ttnn.Tensor:
        key = (length, valid, dtype, across)
        if key not in self._made:
            shape = (1, length, 1) if across == "channels" else (1, 1, length)
            m = torch.from_numpy(mask_1d(length, valid)).reshape(shape)
            self._made[key] = ttnn.from_torch(m, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        return self._made[key]

    def release(self) -> None:
        for t in self._made.values():
            ttnn.deallocate(t)
        self._made.clear()


def apply_mask(t, mask):
    """`t * mask`, freeing `t`; `t` itself when `mask` is None (an unpadded call)."""
    if mask is None:
        return t
    out = ttnn.multiply(t, mask)
    ttnn.deallocate(t)
    return out
