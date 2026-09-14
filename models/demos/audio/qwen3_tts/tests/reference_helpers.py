# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministic inputs and live references for the Qwen3-TTS tests.

No golden files. Each helper builds its input from a seed and computes the reference
in-process from the checkpoint, so the suite needs only the checkpoint and (for the device
tests) a card. Results are cached per process: loading the encoder is the slow part.
"""

import functools
import math

import torch

from models.demos.audio.qwen3_tts import weights
from models.demos.audio.qwen3_tts.reference.qwen3_speaker_ref import SpeakerReference, speaker_mel

# A reference clip is 3 s in the model card's usage; the mel hop of 256 at 24 kHz puts that
# at 281 frames, which is what the device tests compile for.
CLIP_SECONDS = 3.0


# Two voices far enough apart that the encoder separates them: a low, dark, slow one and a
# high, bright, fast one. Keyed by name so a test can ask for a contrast without restating
# the parameters.
VOICES = {
    "low": {"f0": 110.0, "drift": 35.0, "formant": 4.0, "syllables": 3.1, "harmonics": 20},
    "high": {"f0": 215.0, "drift": 60.0, "formant": 11.0, "syllables": 5.4, "harmonics": 12},
}


def synthetic_voiced_clip(seconds=CLIP_SECONDS, seed=0, voice="low"):
    """A deterministic voiced-speech stand-in: harmonic stack, drifting f0, breath noise.

    Real speech would mean shipping an audio fixture. This keeps the input in the same
    territory the mel front-end expects (harmonic structure, a moving pitch, an amplitude
    envelope, peak below 1.0) without one. `voice` selects a timbre from VOICES.
    """
    settings = VOICES[voice]
    sample_rate = weights.SPEAKER_MEL["sampling_rate"]
    count = int(seconds * sample_rate)
    t = torch.arange(count, dtype=torch.float64) / sample_rate

    # f0 drifts around its centre the way a spoken phrase does.
    f0 = settings["f0"] + settings["drift"] * torch.sin(2 * math.pi * 0.45 * t)
    phase = 2 * math.pi * torch.cumsum(f0, dim=0) / sample_rate

    signal = torch.zeros_like(t)
    for harmonic in range(1, settings["harmonics"] + 1):
        # 1/k rolloff, with a broad formant-like bump that sets the timbre.
        weight = (1.0 / harmonic) * (1.0 + 0.6 * math.exp(-((harmonic - settings["formant"]) ** 2) / 8.0))
        signal += weight * torch.sin(harmonic * phase)

    # Syllable-rate envelope, never fully closing, plus a low noise floor.
    envelope = 0.55 + 0.45 * torch.sin(2 * math.pi * settings["syllables"] * t).abs()
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(count, generator=generator, dtype=torch.float64) * 0.005

    clip = signal * envelope + noise
    clip = clip / clip.abs().max() * 0.9
    return clip.to(torch.float32)


@functools.lru_cache(maxsize=None)
def speaker_reference(seconds=CLIP_SECONDS, seed=0):
    """Input mel, reference embedding and per-block intermediates, computed live.

    Returns a dict with `mel` [1, T, 128], `embedding` [1, 2048] and `intermediates`,
    the latter keyed by the names in `qwen3_speaker_ref.INTERMEDIATES` and laid out
    channel-first the way the reference works.
    """
    clip = synthetic_voiced_clip(seconds, seed)
    mel = speaker_mel(clip)
    embedding, intermediates = SpeakerReference()(mel, return_intermediates=True)
    return {"mel": mel, "embedding": embedding, "intermediates": intermediates}
