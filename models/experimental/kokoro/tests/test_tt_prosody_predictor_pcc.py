# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: :class:`~models.experimental.kokoro.tt.tt_prosody_predictor.TTProsodyPredictor`
vs reference :class:`~models.experimental.kokoro.reference.modules.ProsodyPredictor`."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference.modules import ProsodyPredictor
from models.experimental.kokoro.tt.tt_prosody_predictor import (
    TTProsodyPredictor,
    preprocess_tt_prosody_predictor,
)


# Config: small bring-up dims by default; ``KOKORO_PROSODY_DIMS=prod`` selects the real Kokoro-82M
# dims. Only the ``prod`` config (``d_hid=512`` -> shared/duration LSTM H=256) exercises the P1+P2
# L1-resident fp32 per-direction BiLSTM path, which is gated to H>64 (H=64 is already L1-resident);
# use ``prod`` to profile/validate that path. ``d_hid`` must be even; ``d_hid // 2 >= 48`` keeps the
# inner ``conv1d`` on the code path that PCC-matches PyTorch (see ``test_tt_adain_resblk_1d_pcc.py``).
if os.environ.get("KOKORO_PROSODY_DIMS", "small") == "prod":
    _STYLE_DIM = 128
    _D_HID = 512
    _NLAYERS = 3
    _MAX_DUR = 50
else:
    _STYLE_DIM = 64
    _D_HID = 128
    _NLAYERS = 2
    _MAX_DUR = 16


def _make_reference() -> ProsodyPredictor:
    torch.manual_seed(0)
    mod = ProsodyPredictor(style_dim=_STYLE_DIM, d_hid=_D_HID, nlayers=_NLAYERS, max_dur=_MAX_DUR, dropout=0.1)
    mod.eval()
    return mod


def _make_random_alignment(B: int, T: int, T_aligned: int) -> torch.Tensor:
    """Round-trip a roughly-uniform expansion: each ``T`` index repeats ``T_aligned // T`` times.

    Shape ``[B, T, T_aligned]``: row ``i`` of token ``t`` is 1 if that aligned column maps to ``t``.
    """
    rep = T_aligned // T
    align = torch.zeros(B, T, T_aligned)
    for b in range(B):
        for t in range(T):
            align[b, t, t * rep : (t + 1) * rep] = 1.0
    return align


def _upload_inputs(*, ref: ProsodyPredictor, device, B: int, T: int, T_aligned: int, seed: int):
    torch.manual_seed(seed)
    texts = torch.randn(B, _D_HID, T)
    style = torch.randn(B, _STYLE_DIM)
    text_lengths = torch.full((B,), T, dtype=torch.long)
    alignment = _make_random_alignment(B, T, T_aligned)
    text_mask = torch.zeros(B, T, dtype=torch.bool)

    texts_tt = ttnn.from_torch(texts, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    style_tt = ttnn.from_torch(style, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    alignment_tt = ttnn.from_torch(alignment, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return (
        dict(texts=texts, style=style, text_lengths=text_lengths, alignment=alignment, text_mask=text_mask),
        dict(texts_tt=texts_tt, style_tt=style_tt, alignment_tt=alignment_tt),
    )


def test_tt_prosody_predictor_forward_matches_torch(device):
    """``duration`` and ``en`` from :meth:`TTProsodyPredictor.forward` match the reference."""
    ref = _make_reference()
    params = preprocess_tt_prosody_predictor(ref, device)
    tt_mod = TTProsodyPredictor(device, params)

    B, T, T_aligned = 2, 32, 64
    refs, tts = _upload_inputs(ref=ref, device=device, B=B, T=T, T_aligned=T_aligned, seed=1)

    with torch.no_grad():
        ref_duration, ref_en = ref(
            refs["texts"], refs["style"], refs["text_lengths"], refs["alignment"], refs["text_mask"]
        )

    tt_duration, tt_en = tt_mod(
        tts["texts_tt"],
        tts["style_tt"],
        refs["text_lengths"],
        tts["alignment_tt"],
        refs["text_mask"],
    )
    tt_duration_torch = ttnn.to_torch(tt_duration).float()
    tt_en_torch = ttnn.to_torch(tt_en).float()
    while tt_duration_torch.dim() > ref_duration.dim():
        tt_duration_torch = tt_duration_torch.squeeze(0)
    # Reference ``en`` is BCT ``[B, C, T_aligned]``; ours is NLC. Transpose for compare.
    tt_en_bct = tt_en_torch.transpose(-1, -2)
    while tt_en_bct.dim() > ref_en.dim():
        tt_en_bct = tt_en_bct.squeeze(0)

    ttnn.deallocate(tt_duration)
    ttnn.deallocate(tt_en)

    assert tt_duration_torch.shape == ref_duration.shape, (tt_duration_torch.shape, ref_duration.shape)
    assert tt_en_bct.shape == ref_en.shape, (tt_en_bct.shape, ref_en.shape)

    _, pcc_dur = comp_pcc(ref_duration, tt_duration_torch, pcc=0.0)
    _, pcc_en = comp_pcc(ref_en, tt_en_bct, pcc=0.0)
    print(f"TTProsodyPredictor.forward duration PCC: {pcc_dur:.6f}, en PCC: {pcc_en:.6f}")
    assert pcc_dur > 0.99, f"duration PCC too low: {pcc_dur}"
    assert pcc_en > 0.99, f"en PCC too low: {pcc_en}"


def test_tt_prosody_predictor_F0Ntrain_matches_torch(device):
    """``F0Ntrain`` decoder branches match the reference."""
    ref = _make_reference()
    params = preprocess_tt_prosody_predictor(ref, device)
    tt_mod = TTProsodyPredictor(device, params)

    torch.manual_seed(2)
    B, T_aligned = 1, 64
    C = _D_HID + _STYLE_DIM
    en_bct = torch.randn(B, C, T_aligned)
    style = torch.randn(B, _STYLE_DIM)

    with torch.no_grad():
        ref_F0, ref_N = ref.F0Ntrain(en_bct, style)

    en_nlc = en_bct.transpose(-1, -2).contiguous()  # [B, T_aligned, C]
    en_tt = ttnn.from_torch(en_nlc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    style_tt = ttnn.from_torch(style, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_F0, tt_N = tt_mod.F0Ntrain(en_tt, style_tt)
    tt_F0_torch = ttnn.to_torch(tt_F0).float()
    tt_N_torch = ttnn.to_torch(tt_N).float()
    while tt_F0_torch.dim() > ref_F0.dim():
        tt_F0_torch = tt_F0_torch.squeeze(0)
    while tt_N_torch.dim() > ref_N.dim():
        tt_N_torch = tt_N_torch.squeeze(0)

    ttnn.deallocate(tt_F0)
    ttnn.deallocate(tt_N)
    ttnn.deallocate(en_tt)
    ttnn.deallocate(style_tt)

    assert tt_F0_torch.shape == ref_F0.shape, (tt_F0_torch.shape, ref_F0.shape)
    assert tt_N_torch.shape == ref_N.shape, (tt_N_torch.shape, ref_N.shape)

    _, pcc_f0 = comp_pcc(ref_F0, tt_F0_torch, pcc=0.0)
    _, pcc_n = comp_pcc(ref_N, tt_N_torch, pcc=0.0)
    print(f"TTProsodyPredictor.F0Ntrain F0 PCC: {pcc_f0:.6f}, N PCC: {pcc_n:.6f}")
    assert pcc_f0 > 0.99, f"F0 PCC too low: {pcc_f0}"
    assert pcc_n > 0.99, f"N PCC too low: {pcc_n}"
