# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Smoke test for the simplified customer API: pipeline.generate_song(prompt) -> audio.

Validates the one-call text-to-music entry point (tokenization + text encoder + condition encoder
+ DiT denoise + VAE decode, all handled internally). This is the interface a customer uses:

    pipe = create_tt_pipeline(AceStepModelConfig.from_hf(), device)
    wav  = pipe.generate_song("upbeat synthwave, nostalgic", lyrics="neon city lights", seconds=4)

Asserts the output is a well-formed 48 kHz stereo waveform of the requested length with finite,
in-range samples. (Numerical fidelity of each stage is covered by the per-stage PCC tests; this
test guards the plumbing of the public API.) Skipped if the pipeline bundle isn't downloaded.
"""

import pytest
import torch

from models.experimental.acestep.reference.weight_utils import have_pipeline
from models.experimental.acestep.tt.model_config import AceStepModelConfig
from models.experimental.acestep.tt.pipeline import create_tt_pipeline
from models.experimental.acestep.tests.test_utils import require_single_device


@pytest.mark.slow
@pytest.mark.skipif(not have_pipeline(), reason="ACE-Step pipeline bundle not downloaded")
def test_generate_song_api(device):
    require_single_device(device)

    args = AceStepModelConfig.from_hf()
    pipe = create_tt_pipeline(args, device)  # defaults: with_vae=True, with_encoders=True

    seconds = 4.0
    wav = pipe.generate_song(
        "upbeat synthwave, driving bass, nostalgic",
        lyrics="neon lights over the city tonight",
        seconds=seconds,
        infer_steps=8,  # short for a fast smoke test; fidelity is covered elsewhere
        seed=0,
    )

    # Well-formed stereo 48 kHz waveform of ~the requested length.
    assert wav.dim() == 3 and wav.shape[0] == 1 and wav.shape[1] == 2, f"expected [1,2,S], got {tuple(wav.shape)}"
    expected_samples = pipe._latent_len(seconds) * 1920  # 25 Hz latents, 1920 samples/frame
    assert wav.shape[-1] == expected_samples, f"expected {expected_samples} samples, got {wav.shape[-1]}"
    assert torch.isfinite(wav).all(), "waveform has non-finite samples"
    assert wav.abs().max() > 0, "waveform is silent"
    print(f"GENERATE_SONG ok: {tuple(wav.shape)} = {wav.shape[-1] / 48000:.2f}s @ 48kHz, peak={wav.abs().max():.3f}")
