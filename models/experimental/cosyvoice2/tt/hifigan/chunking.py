# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""HiFT in fixed-size mel chunks, with upstream's streaming cache: the schedule and the host-side stitching.

Pure Python and torch, no ttnn: scripts/hift_streaming_reference.py (the reference venv) loads this file by path to
run upstream's own HiFT on the same schedule.

Upstream's streaming HiFT (`CosyVoice2Model.token2wav`, stream=True) keeps three things from each call:
- the last `overlap` = 8 mel frames, which the next call re-synthesizes as its first frames;
- the last 8 x 480 = 3,840 samples of the NSF source, which replace the start of the next call's source
  (`HiFTGenerator.inference(cache_source=...)`), so the sine phase runs on without a jump;
- the last 3,840 output samples, which are held back and crossfaded with the next call's first 3,840
  (`fade_in_out`, a Hamming window of 7,680).

Here every call is exactly `chunk` = 512 frames, one geometry. Chunk i starts `chunk - overlap` = 504 frames after
chunk i-1, and the last chunk is anchored to the end of the mel instead, so nothing is padded. The anchored chunk
overlaps its predecessor by more than 8 frames. Its whole overlap takes the predecessor's source (the upstream
mechanism, over a longer span). The crossfade is still over the predecessor's held-back 8 frames; the anchored chunk's
output before those is discarded, since it was already emitted.

A mel shorter than `chunk` is not chunked: the pipeline runs it in one pass at a small bucket.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

CHUNK_FRAMES = 512
OVERLAP_FRAMES = 8  # upstream CosyVoice2Model.mel_cache_len
HOP = 480  # output samples per mel frame


@dataclass(frozen=True)
class Chunk:
    start: int  # first mel frame of this call
    carry: int  # frames at the start whose source comes from the previous call (0 for the first)
    fade_from: int  # local frame where the crossfade with the previous call's held-back output starts
    emit_to: int  # local frame (exclusive) up to which this call's output is emitted; the rest is held back


def chunk_schedule(frames: int, chunk: int = CHUNK_FRAMES, overlap: int = OVERLAP_FRAMES) -> list[Chunk]:
    """The calls that cover `frames` mel frames (`frames >= chunk`). Each call runs exactly `chunk` frames."""
    if frames < chunk:
        raise ValueError(f"{frames} frames is shorter than one chunk ({chunk}); run it in one pass")
    starts, s = [], 0
    while s + chunk < frames:
        starts.append(s)
        s += chunk - overlap
    starts.append(frames - chunk)  # the last call, anchored to the end (== s when it lines up)
    out = []
    for i, start in enumerate(starts):
        last = i == len(starts) - 1
        carry = 0 if i == 0 else starts[i - 1] + chunk - start
        out.append(
            Chunk(
                start=start,
                carry=carry,
                fade_from=0 if i == 0 else carry - overlap,
                emit_to=chunk if last else chunk - overlap,
            )
        )
    return out


def carried_source(prev_source: torch.Tensor, c: Chunk, chunk: int = CHUNK_FRAMES) -> torch.Tensor:
    """The previous call's source over this call's first `c.carry` frames (its last `c.carry x HOP` samples)."""
    return prev_source[(chunk - c.carry) * HOP :]


def speech_window(overlap: int = OVERLAP_FRAMES) -> np.ndarray:
    """Upstream's `np.hamming(2 * source_cache_len)`."""
    return np.hamming(2 * overlap * HOP)


def fade_in_out(new: torch.Tensor, old: torch.Tensor, window: np.ndarray) -> torch.Tensor:
    """Upstream's `cosyvoice.utils.common.fade_in_out` on 1-D audio: the first half of the window fades `new` in,
    the second half fades `old`'s last samples out, over `len(window) // 2` samples. Computed in float64, as
    upstream's float32-tensor-times-float64-array does, and returned in `new`'s dtype."""
    n = len(window) // 2
    w = torch.from_numpy(window)
    out = new.clone()
    out[:n] = (new[:n].double() * w[:n] + old[-n:].double() * w[n:]).to(new.dtype)
    return out


def stitch(chunk_outputs: list[torch.Tensor], schedule: list[Chunk], crossfade: bool = True) -> torch.Tensor:
    """The audio, from each call's 1-D output (`chunk x HOP` samples each). Each call emits from its crossfade
    start up to its held-back tail; the first `overlap x HOP` emitted samples are crossfaded with the previous
    call's held-back tail. `crossfade=False` switches hard to the new call there instead: the negative control."""
    window = speech_window()
    pieces, held = [], None
    for c, out in zip(schedule, chunk_outputs):
        emit = out[c.fade_from * HOP : c.emit_to * HOP]
        if held is not None and crossfade:
            emit = fade_in_out(emit, held, window)
        pieces.append(emit)
        held = out[c.emit_to * HOP :] if c.emit_to * HOP < len(out) else None
    return torch.cat(pieces)
