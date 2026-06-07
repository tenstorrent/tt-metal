# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Audio input synthesizer — emits ``demo/audio_loader.py`` source.

Same shape as my manual ``models/demos/hf_seamless_m4t_medium/demo/
audio_loader.py``. Generalizes across ASR / S2TT / T2S task templates
(any task that consumes audio).

Usage from a task template:

    from scripts.tt_hw_planner.input_synthesizers import audio
    out.add("demo/audio_loader.py", audio.emit_source(ctx))
"""

from __future__ import annotations

from ..task_templates._base import TemplateContext


def emit_source(ctx: TemplateContext) -> str:
    """Return source code for ``demo/audio_loader.py``.

    The emitted module exposes:
      * ``load_audio_file(path, sample_rate)`` -> np.ndarray
      * ``synthesize_noise(seconds, sample_rate)`` -> np.ndarray (smoke fallback)
      * ``vad_split(audio, ...)`` -> list[(start, end, segment)]
      * ``extract_features(audio, sample_rate)`` -> torch.Tensor (1, T, feat_dim)
      * ``load_and_segment(path, ...)`` -> list[(start, end, features)]

    Task templates use ``load_and_segment`` in their demo, falling
    back to ``synthesize_noise + extract_features`` for the no-WAV
    smoke test.
    """
    model_id = ctx.model_id
    # The feature-extractor model_id can differ from the model_id
    # (some models have a separate processor). For SeamlessM4T-style
    # models the same id works for AutoFeatureExtractor.
    feature_extractor_id = ctx.feature_extractor_name or model_id

    return f'''# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Audio loading + VAD segmentation + fbank features for the generated demo.

Loads a WAV (or any librosa-compatible audio file) and splits long
clips at silence so the TT encoder sees bounded segment lengths.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


SAMPLE_RATE_HZ = 16000
DEFAULT_VAD_TOP_DB = 30
DEFAULT_VAD_FRAME = 2048
DEFAULT_VAD_HOP = 512
DEFAULT_MIN_SEGMENT_SEC = 0.3
DEFAULT_MAX_SEGMENT_SEC = 25.0

_FEATURE_EXTRACTOR_ID = {feature_extractor_id!r}


def _load_feature_extractor():
    import transformers
    return transformers.AutoFeatureExtractor.from_pretrained(_FEATURE_EXTRACTOR_ID)


def load_audio_file(path, sample_rate: int = SAMPLE_RATE_HZ) -> np.ndarray:
    """librosa.load -> mono float32 at target sample_rate."""
    import librosa
    audio, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return audio.astype(np.float32)


def synthesize_noise(seconds: float = 2.0, sample_rate: int = SAMPLE_RATE_HZ) -> np.ndarray:
    """Deterministic low-volume noise clip for smoke tests."""
    rng = np.random.default_rng(42)
    return (0.02 * rng.standard_normal(int(sample_rate * seconds))).astype(np.float32)


def vad_split(
    audio: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE_HZ,
    top_db: int = DEFAULT_VAD_TOP_DB,
    frame_length: int = DEFAULT_VAD_FRAME,
    hop_length: int = DEFAULT_VAD_HOP,
    min_segment_sec: float = DEFAULT_MIN_SEGMENT_SEC,
    max_segment_sec: float = DEFAULT_MAX_SEGMENT_SEC,
) -> List[Tuple[float, float, np.ndarray]]:
    """Silence-based segmentation. Returns [(start_sec, end_sec, audio_segment), ...]."""
    import librosa

    total_sec = len(audio) / sample_rate
    if total_sec < min_segment_sec:
        return [(0.0, total_sec, audio)]

    intervals = librosa.effects.split(
        audio,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    if len(intervals) == 0:
        return [(0.0, total_sec, audio)]

    segments: List[Tuple[float, float, np.ndarray]] = []
    for start_idx, end_idx in intervals:
        seg = audio[start_idx:end_idx]
        seg_sec = (end_idx - start_idx) / sample_rate
        if seg_sec < min_segment_sec:
            continue
        if seg_sec <= max_segment_sec:
            segments.append((start_idx / sample_rate, end_idx / sample_rate, seg))
            continue
        step = int(max_segment_sec * sample_rate)
        for chunk_start in range(0, len(seg), step):
            chunk = seg[chunk_start: chunk_start + step]
            if len(chunk) / sample_rate < min_segment_sec:
                continue
            segments.append(
                (
                    (start_idx + chunk_start) / sample_rate,
                    (start_idx + chunk_start + len(chunk)) / sample_rate,
                    chunk,
                )
            )
    if not segments:
        return [(0.0, total_sec, audio)]
    return segments


def extract_features(audio: np.ndarray, sample_rate: int = SAMPLE_RATE_HZ) -> torch.Tensor:
    """fbank features for one audio segment. Returns ``(1, T, feat_dim)``."""
    fe = _load_feature_extractor()
    out = fe(audio, sampling_rate=sample_rate, return_tensors="pt")
    return out["input_features"]


def load_and_segment(
    path,
    *,
    sample_rate: int = SAMPLE_RATE_HZ,
    top_db: int = DEFAULT_VAD_TOP_DB,
    min_segment_sec: float = DEFAULT_MIN_SEGMENT_SEC,
    max_segment_sec: float = DEFAULT_MAX_SEGMENT_SEC,
) -> List[Tuple[float, float, torch.Tensor]]:
    """End-to-end: WAV path -> list of ``(start_sec, end_sec, fbank_features)``."""
    audio = load_audio_file(path, sample_rate=sample_rate)
    segments = vad_split(
        audio,
        sample_rate=sample_rate,
        top_db=top_db,
        min_segment_sec=min_segment_sec,
        max_segment_sec=max_segment_sec,
    )
    return [(s, e, extract_features(seg, sample_rate)) for s, e, seg in segments]


__all__ = [
    "SAMPLE_RATE_HZ",
    "load_audio_file",
    "synthesize_noise",
    "vad_split",
    "extract_features",
    "load_and_segment",
]
'''
