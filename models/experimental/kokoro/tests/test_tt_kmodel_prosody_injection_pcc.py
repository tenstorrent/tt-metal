# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Teacher-forced waveform PCC for the TTNN Kokoro port.

"Teacher forcing" here means feeding the reference (CPU float32) KModel's *ground-truth*
prosody outputs — ``asr``, ``F0``, ``N`` and the style vector — straight into the TT decoder /
vocoder, instead of letting the TT prosody stack predict them.  The rest of the vocoder
(decode stack + generator + (i)STFT) runs on device.  The resulting waveform PCC isolates the
on-device **vocoder** error from any prosody-prediction drift.

Contrast with the standard full-pipeline PCC (``test_tt_kmodel_pcc.py``), where the TT model
predicts its own prosody.  Because the prosody stack is already PCC > 0.998 on short text, the
teacher-forced number should be *very close* to the full-pipeline number — which is exactly the
point: it confirms the residual deficit lives in the harmonic-source / STFT vocoder path, not in
prosody prediction.

The reference is built ``disable_complex=False`` so its generator uses ``TorchSTFT``
(``torch.stft``), matching the TT ``use_torch_stft_fallback`` formulation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.m_source_rng import (
    deallocate_m_source_rng_tt,
    make_zero_m_source_rng,
    upload_m_source_rng,
)
from models.experimental.kokoro.tests.kokoro_checkpoint import _ref_prosody, _zero_noise
from models.experimental.kokoro.tests.test_tt_kmodel_pcc import (
    _find_checkpoint,
    _ref_audio,
    _setup,
    _tt_audio,
)
from models.experimental.kokoro.tt.tt_kmodel import TTKModel


def _audio_pcc(ref: torch.Tensor, tt: torch.Tensor) -> float:
    r = ref.detach().float().reshape(-1)
    t = tt.detach().float().reshape(-1)
    n = min(r.numel(), t.numel())
    _, pcc = comp_pcc(r[:n].unsqueeze(0), t[:n].unsqueeze(0), pcc=0.0)
    return float(pcc)


def _teacher_forced_audio(
    device,
    ref,
    params,
    phonemes: str,
    ref_s: torch.Tensor,
    *,
    use_torch_stft_fallback: bool,
    use_torch_phase_fallback: bool,
) -> tuple[torch.Tensor, int]:
    """Run the TT vocoder on *reference* prosody (asr/F0/N/style).  Returns (audio, T_mel)."""
    # Reference ground-truth prosody (CPU float32), computed with zero noise to match _ref_audio.
    with torch.no_grad(), _zero_noise():
        asr_bct, F0_curve, N_curve, s_style, _pred_dur = _ref_prosody(ref, phonemes, ref_s)

    t_mel = int(asr_bct.shape[-1])

    with _zero_noise():
        tt_model = TTKModel(
            device,
            ref,
            params,
            use_torch_stft_fallback=use_torch_stft_fallback,
            use_torch_phase_fallback=use_torch_phase_fallback,
        )

    mc = ttnn.DRAM_MEMORY_CONFIG
    decoder = tt_model._get_decoder(t_mel)
    gen = decoder._generator

    # Upload reference prosody.  asr: BCT [B,512,T_mel] -> NLC [B,T_mel,512].
    asr_nlc = ttnn.from_torch(
        asr_bct.transpose(1, 2).contiguous().float(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mc,
    )
    F0_tt = ttnn.from_torch(
        F0_curve.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
    )
    N_tt = ttnn.from_torch(
        N_curve.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
    )
    s_tt = ttnn.from_torch(
        s_style.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
    )

    # Deterministic (zero) harmonic-source noise, matching _ref_audio's _zero_noise.
    B_dec = int(F0_curve.shape[0])
    T_har = int(F0_curve.shape[1]) * int(gen.params.upsample_scale_full)
    dim = int(gen.params.m_source.sinegen.dim)
    rng_cpu = make_zero_m_source_rng(B_dec, T_har, dim)
    rng_tt = upload_m_source_rng(rng_cpu, device, memory_config=mc)

    audio_tt = decoder(
        asr_nlc,
        F0_tt,
        N_tt,
        s_tt,
        sinegen_rand_ini=rng_tt.rand_ini,
        sinegen_noise_raw=rng_tt.sinegen_noise,
        source_noise_raw=rng_tt.source_noise,
        memory_config=mc,
    )
    audio = ttnn.to_torch(audio_tt).float().squeeze()

    deallocate_m_source_rng_tt(rng_tt)
    for t in (audio_tt, asr_nlc, F0_tt, N_tt, s_tt):
        ttnn.deallocate(t)

    return audio, t_mel


@pytest.mark.timeout(600)
def test_tt_kmodel_teacher_forcing_waveform_pcc(device):
    """Teacher-force ref prosody into the TT vocoder; report waveform PCC vs full pipeline.

    Reports, for both the no-fallback and config-E (STFT+phase) configs:
      * teacher-forced PCC  — TT vocoder on reference asr/F0/N/style,
      * full-pipeline PCC   — TT predicts its own prosody (standard path).

    The two should track closely (prosody stack is already > 0.998), proving the residual
    vocoder deficit is harmonic-source / STFT, not prosody prediction.
    """
    ckpt = _find_checkpoint()
    if ckpt is None:
        pytest.skip("Kokoro-82M checkpoint not found locally.")

    # disable_complex=False -> reference uses TorchSTFT (torch.stft), matching the TT STFT fallback.
    ref, params, phonemes, ref_s = _setup(ckpt, device, disable_complex=False)
    y_ref = _ref_audio(ref, phonemes, ref_s)

    configs = {
        "none": dict(use_torch_stft_fallback=False, use_torch_phase_fallback=False),
        "stft+phase": dict(use_torch_stft_fallback=True, use_torch_phase_fallback=True),
    }

    rows: dict[str, dict[str, float]] = {}
    for name, kw in configs.items():
        y_tf, t_mel = _teacher_forced_audio(device, ref, params, phonemes, ref_s, **kw)
        assert torch.isfinite(y_tf).all(), f"{name}: teacher-forced audio has NaN/Inf"
        assert y_tf.abs().max().item() > 1e-3, f"{name}: teacher-forced audio is ~zero"
        pcc_tf = _audio_pcc(y_ref, y_tf)

        y_full = _tt_audio(device, ref, params, phonemes, ref_s, disable_complex=False, **kw)
        pcc_full = _audio_pcc(y_ref, y_full)

        rows[name] = {"teacher_forced": pcc_tf, "full_pipeline": pcc_full}

    print(f"\nTeacher-forcing waveform PCC  (phonemes={len(phonemes)}, ref disable_complex=False):")
    print(f"  {'config':<12} {'teacher_forced':>15} {'full_pipeline':>15} {'delta':>9}")
    for name in configs:
        tf = rows[name]["teacher_forced"]
        fp = rows[name]["full_pipeline"]
        print(f"  {name:<12} {tf:>15.6f} {fp:>15.6f} {tf - fp:>+9.4f}")

    # Teacher forcing must be at least as good as the full pipeline (prosody can only add error),
    # within run-to-run jitter.
    for name in configs:
        tf = rows[name]["teacher_forced"]
        fp = rows[name]["full_pipeline"]
        assert tf >= fp - 0.02, (
            f"{name}: teacher-forced PCC {tf:.4f} is well below full-pipeline {fp:.4f} — "
            "feeding ground-truth prosody should not hurt"
        )

    # Config E with ground-truth prosody should clear the production vocoder floor.
    assert (
        rows["stft+phase"]["teacher_forced"] > 0.84
    ), f"teacher-forced config-E PCC {rows['stft+phase']['teacher_forced']:.4f} below floor 0.84"
