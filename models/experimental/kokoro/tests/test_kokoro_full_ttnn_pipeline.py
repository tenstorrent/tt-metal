# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end: :class:`KokoroFullTtnn` vs :class:`KokoroFullReference` (short utterance, deterministic)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

ttnn = pytest.importorskip("ttnn")

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference import KokoroConfig, KokoroFullReference


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_kokoro_full_ttnn_matches_reference_waveform(mesh_device):
    pytest.importorskip("kokoro")
    from kokoro import KPipeline

    from models.experimental.kokoro.tt.ttnn_kokoro_full_pipeline import KokoroFullTtnn

    text = "Hi."
    voice = "af_heart"
    pipe = KPipeline(lang_code="a", model=False)
    results = list(pipe(text, voice=voice, speed=2.0))
    assert results, "pipeline produced no chunks"
    phonemes = results[0].phonemes
    assert phonemes

    pack = pipe.load_voice(voice)
    ref_s = pack[len(phonemes) - 1].to("cpu")
    if ref_s.dim() == 1:
        ref_s = ref_s.unsqueeze(0)

    def zeros_rand(*args, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k != "generator"}
        return torch.zeros(*args, **kwargs)

    def zeros_randn(*args, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k != "generator"}
        return torch.zeros(*args, **kwargs)

    def zeros_randn_like(t, **kwargs):
        return torch.zeros_like(t)

    ref = KokoroFullReference(repo_id=KokoroConfig.repo_id, device="cpu", disable_complex=True)
    tt_model = KokoroFullTtnn(mesh_device, repo_id=KokoroConfig.repo_id, disable_complex=True)

    with (
        mock.patch("torch.rand", side_effect=zeros_rand),
        mock.patch("torch.randn", side_effect=zeros_randn),
        mock.patch("torch.randn_like", side_effect=zeros_randn_like),
    ):
        with torch.no_grad():
            out_ref = ref(phonemes=phonemes, ref_s=ref_s, speed=2.0)

    with (
        mock.patch("torch.rand", side_effect=zeros_rand),
        mock.patch("torch.randn", side_effect=zeros_randn),
        mock.patch("torch.randn_like", side_effect=zeros_randn_like),
    ):
        with torch.no_grad():
            out_tt = tt_model(phonemes=phonemes, ref_s=ref_s, speed=2.0, deterministic=True)

    assert out_ref.audio.shape == out_tt.audio.shape, f"audio shape ref={out_ref.audio.shape} tt={out_tt.audio.shape}"
    assert torch.isfinite(out_tt.audio).all()
    ok, p = comp_pcc(out_ref.audio, out_tt.audio, pcc=0.50)
    print(f"full Kokoro TTNN vs ref PCC={p:.6f} pass={ok} (min 0.50)")
    assert ok, f"full pipeline PCC {p} expected >= 0.50"
