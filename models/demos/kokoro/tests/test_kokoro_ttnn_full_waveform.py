# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: full TTNN Kokoro produces waveform (non-hybrid)."""

from __future__ import annotations

import pytest
import torch


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_kokoro_ttnn_full_produces_audio(mesh_device):
    pytest.importorskip("kokoro")
    from kokoro import KPipeline

    from models.demos.kokoro.reference import KokoroConfig
    from models.demos.kokoro.tt.kokoro_tt_full_model import KokoroTtFull

    # Keep this very short to avoid large intermediate activations
    # that can trigger L1 circular-buffer overflows in conv1d.
    text = "Hi."
    voice = "af_heart"
    pipe = KPipeline(lang_code="a", model=False)
    results = list(pipe(text, voice=voice, speed=1.0))
    assert results, "pipeline produced no chunks"
    phonemes = results[0].phonemes
    assert phonemes

    pack = pipe.load_voice(voice)
    ref_s = pack[len(phonemes) - 1].to("cpu")
    if ref_s.dim() == 1:
        ref_s = ref_s.unsqueeze(0)

    model = KokoroTtFull(mesh_device, repo_id=KokoroConfig.repo_id)
    torch.manual_seed(0)
    out = model(phonemes=phonemes, ref_s=ref_s, speed=2.0)
    audio = out.audio
    assert audio.ndim == 1
    assert audio.numel() > 1000
    assert torch.isfinite(audio).all()
