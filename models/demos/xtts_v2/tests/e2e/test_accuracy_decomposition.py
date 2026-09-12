# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Accuracy decomposition for coqui/XTTS-v2 (PRINT-ONLY — no gates, no thresholds).

Re-derives, in one command, WHY the full-chain waveform number is what it is:

1. 2x2 vocoder ablation — swap TT/HF latents and d-vector independently through the
   reference HiFi-GAN: separates the d-vector term from the GPT-latents term.
2. Speaker-encoder front-end sub-stage PCCs (mel / log-mel / InstanceNorm /
   embedding) for BOTH input dtypes: bf16 (the old upload — reproduces the
   historical floor) and fp32 (the current pipeline). This is the A1 experiment
   that root-caused the d-vector term to the bf16 wav upload.
3. Phase-insensitive log-mel spectral metrics alongside raw-sample PCC.

Nothing here is gated: every number is printed, no assert compares against a
threshold. Run: pytest models/demos/xtts_v2/tests/e2e/test_accuracy_decomposition.py -s
"""

from __future__ import annotations

import os

import pytest
import torch

from models.common.utility_functions import comp_pcc
from models.demos.xtts_v2 import reference
from models.demos.xtts_v2.tt import pipeline as P

HF_MODEL_ID = "coqui/XTTS-v2"
_N = int(os.environ.get("XTTS_E2E_N", "40"))


def _load_reference():
    return reference.load_reference_model(HF_MODEL_ID)


def _pcc(a, b):
    return comp_pcc(a.float().reshape(-1), b.float().reshape(-1), 0.0)[1]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_accuracy_decomposition(device):
    torch.manual_seed(0)
    model = _load_reference()
    fo = P.forward_on_device(device, model, "hello world.", "en", None, _N, 5.0, True)
    ins = fo["ins"]

    g_tt = P._th(fo["g"])
    latents_tt = P._th(fo["latents"])
    codes_tt = P._th(fo["codes"]).round().to(torch.long)
    wav_tt = P._th(fo["waveform"]).reshape(-1)
    cond_latent_tt = P._th(fo["cond_lat"])

    g_hf = P._hf_speaker_embedding(model, ins["wav_16k"])
    latents_hf = P._hf_latents(model, ins["text_tokens"], fo["text_len"], codes_tt, fo["exp_len"], cond_latent_tt)

    # ── 1. 2x2 vocoder ablation ──────────────────────────────────────────────
    wav = {
        ("tt", "tt"): wav_tt,
        ("tt", "hf"): P._hf_vocode(model, latents_tt, g_hf).reshape(-1),
        ("hf", "tt"): P._hf_vocode(model, latents_hf, g_tt).reshape(-1),
        ("hf", "hf"): P._hf_vocode(model, latents_hf, g_hf).reshape(-1),
    }
    print("\n== 2x2 vocoder ablation (rows latents, cols d-vector; PCC vs wav_tt) ==")
    for (lat_src, g_src), w in wav.items():
        mm = min(w.shape[0], wav_tt.shape[0])
        print(f"  lat_{lat_src} g_{g_src}: pcc_vs_wav_tt={_pcc(w[:mm], wav_tt[:mm]):.7f}")
    w_tt_hf, w_hf_tt, w_hf_hf = wav[("tt", "hf")], wav[("hf", "tt")], wav[("hf", "hf")]
    mm = min(w_tt_hf.shape[0], w_hf_hf.shape[0])
    print(f"  d-vector term   vocode(lat_tt, g_tt) vs vocode(lat_tt, g_hf): {_pcc(wav_tt[:mm], w_tt_hf[:mm]):.7f}")
    print(f"  latents term    vocode(lat_tt, g_tt) vs vocode(lat_hf, g_tt): {_pcc(wav_tt[:mm], w_hf_tt[:mm]):.7f}")
    print(f"  full chain      vocode(lat_tt, g_tt) vs vocode(lat_hf, g_hf): {_pcc(wav_tt[:mm], w_hf_hf[:mm]):.7f}")
    print(f"  latents_pcc (unit-space) = {_pcc(latents_hf, latents_tt):.7f}")
    print(f"  g cosine = {torch.nn.functional.cosine_similarity(g_hf.reshape(1, -1), g_tt.reshape(1, -1)).item():.7f}")

    # ── 2. speaker front-end sub-stages, both input dtypes ───────────────────
    import ttnn

    se = P._resolve(model, "hifigan_decoder.speaker_encoder")
    wav = ins["wav_16k"]
    with torch.no_grad():
        mel_hf = se.torch_spec(wav)
        logmel_hf = torch.log(mel_hf + 1e-6)
        h_hf = se.instancenorm(logmel_hf)
        emb_hf = g_hf

    pre = P._build("pre_emphasis")(device, se.torch_spec[0])
    mel_fe = P._build("mel_spectrogram")(device, se.torch_spec[1])
    inorm = P._build("instance_norm1d")(device, se.instancenorm)
    se_fwd = P._build("res_net_speaker_encoder")(device, se)

    print("\n== speaker-encoder front-end sub-stages vs HF (fp32 torch) ==")
    print(f"  {'input':6s} {'mel':>10s} {'logmel':>10s} {'instnorm':>10s} {'emb_pcc':>10s} {'emb_cos':>10s}")
    for name, dt in [("bf16", ttnn.bfloat16), ("fp32", ttnn.float32)]:
        x = ttnn.as_tensor(
            wav.reshape(1, -1).contiguous().float(),
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if dt != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        mel_tt = mel_fe(pre(x))
        logmel_tt = ttnn.log(ttnn.add(mel_tt, 1e-6))
        h_tt = inorm(logmel_tt)
        wav16 = ttnn.as_tensor(
            wav.contiguous().float(),
            dtype=dt,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        emb_tt = P._th(P._l2norm_device(se_fwd(wav16))).reshape(emb_hf.shape)
        row = (
            _pcc(mel_hf, P._th(mel_tt).reshape(mel_hf.shape)),
            _pcc(logmel_hf, P._th(logmel_tt).reshape(logmel_hf.shape)),
            _pcc(h_hf, P._th(h_tt).reshape(h_hf.shape)),
            _pcc(emb_hf, emb_tt),
            torch.nn.functional.cosine_similarity(emb_hf.reshape(1, -1), emb_tt.reshape(1, -1)).item(),
        )
        print(f"  {name:6s} {row[0]:10.7f} {row[1]:10.7f} {row[2]:10.7f} {row[3]:10.7f} {row[4]:10.7f}")

    # ── 3. phase-insensitive full-chain metrics ──────────────────────────────
    m = min(wav_tt.shape[0], w_hf_hf.shape[0])
    lp, l1 = P._logmel_spectral_metrics(wav_tt[:m], w_hf_hf[:m])
    print("\n== phase-insensitive full-chain ==")
    print(f"  raw waveform PCC = {_pcc(wav_tt[:m], w_hf_hf[:m]):.7f}")
    print(f"  logmel_pcc = {lp:.7f}   logmel_l1 = {l1:.7f}")
