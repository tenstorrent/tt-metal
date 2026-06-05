# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Long-audio handling for SeamlessM4T v2.

SeamlessM4T is an **utterance-level** model: it is trained on short clips and, given a long clip,
tends to *translate* (e.g. Hindi speech → English text) instead of transcribing, and to degenerate
— and the HF reference does this too (so it is a model property, not a TTNN bug). The standard
remedy, used by real SeamlessM4T pipelines, is to **segment long audio at silences into short
(≤~15-20 s) utterance chunks**, run the model per chunk, and join the results.

``segment_by_silence`` is a VAD-style splitter (ported from the ``yito`` branch's demo): it cuts only
at silences (``librosa.effects.split``) and groups non-silent regions into spans up to ``max_sec`` so
cuts land on natural pauses, hard-splitting any still-overlong span. Callers run ASR/S2TT on each
returned chunk (resetting the generation runtime between chunks) and concatenate the outputs.
"""

from __future__ import annotations

from typing import List

import numpy as np


def segment_by_silence(
    wav: np.ndarray,
    sr: int,
    max_sec: float = 15.0,
    top_db: float = 30.0,
    min_sec: float = 0.3,
) -> List[np.ndarray]:
    """Split ``wav`` into ≤ ``max_sec`` chunks, cutting at silences.

    Returns a list of 1-D float32 segments (in order). Falls back to a single hard-split-by-``max_sec``
    partition if no silences are found, and drops chunks shorter than ``min_sec``. A clip already
    shorter than ``max_sec`` is returned unchanged as a single segment.
    """
    import librosa

    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    max_len = max(1, int(max_sec * sr))
    min_len = int(min_sec * sr)
    if wav.size <= max_len:
        return [wav]

    intervals = librosa.effects.split(wav, top_db=top_db)
    if len(intervals) == 0:
        intervals = np.array([[0, wav.size]])

    # Group consecutive non-silent intervals into spans no longer than max_len (cut at silences).
    spans: List[tuple] = []
    seg_s, seg_e = int(intervals[0][0]), int(intervals[0][1])
    for s, e in intervals[1:]:
        if int(e) - seg_s <= max_len:
            seg_e = int(e)
        else:
            spans.append((seg_s, seg_e))
            seg_s, seg_e = int(s), int(e)
    spans.append((seg_s, seg_e))

    # Hard-split any span still longer than max_len (a single utterance with no internal pause).
    out: List[tuple] = []
    for s, e in spans:
        if e - s <= max_len:
            out.append((s, e))
        else:
            for c in range(s, e, max_len):
                out.append((c, min(c + max_len, e)))

    segs = [wav[s:e] for s, e in out if e - s >= min_len]
    return segs or [wav]
