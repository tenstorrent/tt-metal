# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Compare the tt-metal Kokoro **reference** implementation (HF checkpoint, `reference/*.py`)
against the **official** `kokoro` package (`KModel`).

Similar in spirit to `models/demos/qwen3_tts/tests/test_reference_vs_official.py`:
optional prints for debugging, plus assertions suitable for `pytest`.

Requires:
- `pip install kokoro soundfile`
- `espeak-ng` on PATH for G2P (end-to-end text tests)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# tt-metal root (models/demos/kokoro/tests -> parents[4])
_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation coefficient between flattened tensors."""
    a = a.flatten().float()
    b = b.flatten().float()
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    cov = (a_centered * b_centered).sum()
    std_a = torch.sqrt((a_centered**2).sum())
    std_b = torch.sqrt((b_centered**2).sum())
    if std_a == 0 or std_b == 0:
        return 0.0
    return (cov / (std_a * std_b)).item()


def test_official_kmodel_produces_finite_audio():
    """Sanity check: upstream `KModel` runs and returns finite waveform."""
    pytest.importorskip("kokoro")
    from models.demos.kokoro.reference import KokoroConfig, load_reference_model

    device = "cpu"
    hf_repo_id = KokoroConfig().repo_id
    km = load_reference_model(repo_id=hf_repo_id, device=device).kmodel

    input_ids = torch.zeros((1, 8), dtype=torch.long, device=device)
    ref_s = torch.randn(1, 256, dtype=torch.float32, device=device)

    with torch.no_grad():
        audio, pred_dur = km.forward_with_tokens(input_ids=input_ids, ref_s=ref_s, speed=1.0)

    assert torch.isfinite(audio).all(), "official audio has non-finite values"
    assert audio.numel() > 0
    assert pred_dur.numel() > 0
    print(f"  official audio shape={tuple(audio.shape)} pred_dur shape={tuple(pred_dur.shape)}")


def test_reference_plbert_matches_official_kmodel():
    """PLBERT + projection: reference `KokoroPlBert` vs upstream `km.bert` + `km.bert_encoder`."""
    pytest.importorskip("kokoro")
    from models.demos.kokoro.reference import KokoroConfig, load_plbert_from_huggingface, load_reference_model

    device = "cpu"
    hf_repo_id = KokoroConfig().repo_id
    reference_plbert = load_plbert_from_huggingface(repo_id=hf_repo_id, device=device)
    upstream = load_reference_model(repo_id=hf_repo_id, device=device).kmodel

    torch.manual_seed(0)
    input_ids = torch.zeros((1, 16), dtype=torch.long, device=device)

    reference_out = reference_plbert(input_ids)
    bert_dur_up = upstream.bert(input_ids, attention_mask=(~reference_out.text_mask).int())
    d_en_up = upstream.bert_encoder(bert_dur_up).transpose(-1, -2)

    match_dur = torch.allclose(reference_out.bert_dur, bert_dur_up, atol=1e-5, rtol=1e-5)
    match_den = torch.allclose(reference_out.d_en, d_en_up, atol=1e-5, rtol=1e-5)
    print(f"  PLBERT bert_dur match={match_dur} d_en match={match_den}")

    assert match_dur, "bert_dur mismatch vs official KModel.bert"
    assert match_den, "d_en mismatch vs official bert_encoder"


def test_reference_full_model_matches_official_waveform():
    """
    End-to-end: reference `KokoroFullReference` vs official `KModel` on the same phonemes + style.

    Uses `KPipeline(model=False)` for phonemes only. Decoder uses RNG — seed before each forward.
    """
    pytest.importorskip("kokoro")
    from kokoro import KPipeline

    from models.demos.kokoro.reference import KokoroConfig, load_full_reference_from_huggingface, load_reference_model

    device = "cpu"
    hf_repo_id = KokoroConfig().repo_id
    text = "Hello from Kokoro reference vs official test."
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
    upstream = load_reference_model(repo_id=hf_repo_id, device=device).kmodel

    torch.manual_seed(0)
    out_reference = reference_full(phonemes=phonemes, ref_s=ref_s, speed=1.0).audio
    torch.manual_seed(0)
    out_official = upstream(phonemes, ref_s, speed=1.0, return_output=False)

    pcc = compute_pcc(out_reference, out_official)
    close = torch.allclose(out_reference, out_official, atol=2e-3, rtol=1e-4)
    print(
        f"  E2E waveform allclose={close} PCC={pcc:.6f} "
        f"reference_len={out_reference.numel()} official_len={out_official.numel()}"
    )

    assert close, f"full-model waveform mismatch (PCC={pcc:.6f})"
    assert pcc > 0.999, f"unexpected low PCC={pcc:.6f}"


def test_reference_predictor_matches_official_submodules():
    """
    Predictor + text encoder path vs upstream `predictor` / `text_encoder` on shared PLBERT outputs.
    """
    pytest.importorskip("kokoro")
    from models.demos.kokoro.reference import (
        KokoroConfig,
        load_plbert_from_huggingface,
        load_predictor_from_huggingface,
        load_reference_model,
    )

    device = "cpu"
    hf_repo_id = KokoroConfig().repo_id
    torch.manual_seed(42)
    input_ids = torch.randint(2, 80, (1, 24), dtype=torch.long, device=device)
    ref_s = torch.randn(1, 256, dtype=torch.float32, device=device)

    reference_plbert = load_plbert_from_huggingface(repo_id=hf_repo_id, device=device)
    reference_predictor = load_predictor_from_huggingface(repo_id=hf_repo_id, device=device)
    pl_out = reference_plbert(input_ids)

    reference_pred_out = reference_predictor(
        d_en=pl_out.d_en,
        ref_s=ref_s,
        input_ids=input_ids,
        input_lengths=pl_out.input_lengths,
        text_mask=pl_out.text_mask,
        speed=1.0,
    )

    km = load_reference_model(repo_id=hf_repo_id, device=device).kmodel
    s = ref_s[:, 128:]
    speed = 1.0
    with torch.no_grad():
        d = km.predictor.text_encoder(pl_out.d_en, s, pl_out.input_lengths, pl_out.text_mask)
        x, _ = km.predictor.lstm(d)
        duration = km.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur_up = torch.round(duration).clamp(min=1).long().squeeze()

        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=input_ids.device), pred_dur_up)
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=input_ids.device)
        pred_aln_trg[indices, torch.arange(indices.shape[0], device=input_ids.device)] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(input_ids.device)

        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred_up, N_pred_up = km.predictor.F0Ntrain(en, s)
        t_en_up = km.text_encoder(input_ids, pl_out.input_lengths, pl_out.text_mask)
        asr_up = t_en_up @ pred_aln_trg

    assert torch.equal(reference_pred_out.pred_dur, pred_dur_up), "pred_dur mismatch vs official predictor path"
    assert torch.allclose(reference_pred_out.asr, asr_up, atol=1e-4, rtol=1e-4), "asr mismatch"
    assert torch.allclose(reference_pred_out.t_en, t_en_up, atol=1e-4, rtol=1e-4), "t_en mismatch"
    assert torch.allclose(reference_pred_out.F0_pred, F0_pred_up, atol=5e-3, rtol=1e-3), "F0_pred drift vs official"
    assert torch.allclose(reference_pred_out.N_pred, N_pred_up, atol=5e-3, rtol=1e-3), "N_pred drift vs official"
    print("  predictor path: pred_dur/asr/t_en/F0/N checks passed")


def test_reference_decoder_matches_official_istftnet():
    """ISTFTNet decoder: same ASR / F0 / N / style as upstream after identical predictor alignment."""
    pytest.importorskip("kokoro")
    from models.demos.kokoro.reference import (
        KokoroConfig,
        load_decoder_from_huggingface,
        load_plbert_from_huggingface,
        load_predictor_from_huggingface,
        load_reference_model,
    )

    device = "cpu"
    hf_repo_id = KokoroConfig().repo_id
    torch.manual_seed(0)
    input_ids = torch.randint(2, 80, (1, 20), dtype=torch.long, device=device)
    ref_s = torch.randn(1, 256, dtype=torch.float32, device=device)

    reference_plbert = load_plbert_from_huggingface(repo_id=hf_repo_id, device=device)
    reference_predictor = load_predictor_from_huggingface(repo_id=hf_repo_id, device=device)
    reference_decoder = load_decoder_from_huggingface(repo_id=hf_repo_id, device=device)

    pl_out = reference_plbert(input_ids)
    pred = reference_predictor(
        d_en=pl_out.d_en,
        ref_s=ref_s,
        input_ids=input_ids,
        input_lengths=pl_out.input_lengths,
        text_mask=pl_out.text_mask,
        speed=1.0,
    )

    km = load_reference_model(repo_id=hf_repo_id, device=device).kmodel

    torch.manual_seed(0)
    out_reference = reference_decoder(asr=pred.asr, F0_pred=pred.F0_pred, N_pred=pred.N_pred, ref_s=ref_s).audio
    torch.manual_seed(0)
    with torch.no_grad():
        out_official = km.decoder(pred.asr, pred.F0_pred, pred.N_pred, ref_s[:, :128]).squeeze()

    pcc = compute_pcc(out_reference.flatten(), out_official.flatten())
    ok = torch.allclose(out_reference.cpu().flatten(), out_official.cpu().flatten(), atol=2e-3, rtol=1e-4)
    print(f"  decoder allclose={ok} PCC={pcc:.6f}")

    assert ok, "decoder waveform mismatch vs official km.decoder"
    assert pcc > 0.9999


def main() -> None:
    """Run all sections manually (like qwen3_tts test_reference_vs_official.py)."""
    print("Kokoro reference vs official (kokoro KModel)")
    print("=" * 80)

    test_official_kmodel_produces_finite_audio()
    print()

    test_reference_plbert_matches_official_kmodel()
    print()

    test_reference_predictor_matches_official_submodules()
    print()

    test_reference_decoder_matches_official_istftnet()
    print()

    test_reference_full_model_matches_official_waveform()
    print()

    print("Done.")


if __name__ == "__main__":
    main()
