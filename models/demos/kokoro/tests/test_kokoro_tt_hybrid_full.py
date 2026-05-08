# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end PCC: `KokoroTtHybridFull` (TT `bert_encoder` linear) vs full PyTorch reference."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_tt_hybrid_full_matches_torch_reference(mesh_device):
    """
    Same phonemes + ref_s + RNG seed as `test_reference_full_model_matches_official_waveform`:
    waveform PCC should stay high with bf16 only on the projection matmul.
    """
    pytest.importorskip("kokoro")
    from kokoro import KPipeline

    from models.common.utility_functions import comp_pcc
    from models.demos.kokoro.reference import KokoroConfig, load_full_reference_from_huggingface
    from models.demos.kokoro.tt.kokoro_tt_hybrid_model import KokoroTtHybridFull

    device = "cpu"
    hf_repo_id = KokoroConfig.repo_id
    text = "Hello from Kokoro TT hybrid vs reference test."
    voice = "af_heart"

    pipe = KPipeline(lang_code="a", model=False)
    results = list(pipe(text, voice=voice, speed=1.0))
    assert results, "pipeline produced no chunks"
    phonemes = results[0].phonemes
    assert phonemes and len(phonemes) > 0

    pack = pipe.load_voice(voice)
    ref_s = pack[len(phonemes) - 1].to(device)
    if ref_s.dim() == 1:
        ref_s = ref_s.unsqueeze(0)

    reference_full = load_full_reference_from_huggingface(repo_id=hf_repo_id, device=device)
    tt_full = KokoroTtHybridFull(mesh_device, repo_id=hf_repo_id, torch_device=device)

    torch.manual_seed(0)
    out_reference = reference_full(phonemes=phonemes, ref_s=ref_s, speed=1.0).audio
    torch.manual_seed(0)
    out_tt = tt_full(phonemes=phonemes, ref_s=ref_s, speed=1.0).audio

    assert out_reference.shape == out_tt.shape, f"shape mismatch ref={out_reference.shape} tt={out_tt.shape}"

    ok, reported_pcc = comp_pcc(out_reference, out_tt, pcc=0.97)
    assert ok, f"waveform PCC too low: {reported_pcc}"
