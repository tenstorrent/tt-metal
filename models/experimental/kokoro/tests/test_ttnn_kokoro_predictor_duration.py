# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: TTNN Kokoro predictor duration/alignment path vs PyTorch reference."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference import KokoroConfig, load_predictor_from_huggingface
from models.experimental.kokoro.tt.ttnn_kokoro_predictor import TtKokoroPredictorDuration, preprocess_predictor_duration


def _make_fake_den(device, B=1, T=32, C=512):
    # d_en is [B,C,T]
    torch.manual_seed(0)
    den = torch.randn(B, C, T, dtype=torch.float32)
    return ttnn.from_torch(den, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def test_ttnn_predictor_duration_matches_torch(device):
    torch_ref = load_predictor_from_huggingface(repo_id=KokoroConfig.repo_id, device="cpu")
    params = preprocess_predictor_duration(torch_ref.predictor, device)
    tt = TtKokoroPredictorDuration(device, params)

    B, T = 1, 32
    input_ids = torch.randint(0, 50, (B, T), dtype=torch.long)
    input_lengths = torch.tensor([T], dtype=torch.long)
    text_mask = torch.zeros((B, T), dtype=torch.bool)
    ref_s = torch.randn(B, 256, dtype=torch.float32)

    d_en_tt = _make_fake_den(device, B=B, T=T, C=params.d_hid)
    with torch.no_grad():
        s = ref_s[:, 128:]
        d_ref = torch_ref.predictor.text_encoder(ttnn.to_torch(d_en_tt).to(torch.float32), s, input_lengths, text_mask)
        x_ref, _ = torch_ref.predictor.lstm(d_ref)
        dur_logits = torch_ref.predictor.duration_proj(x_ref)
        dur_ref = torch.sigmoid(dur_logits).sum(axis=-1)

    d_tt, dur_tt, pred_dur_tt, pred_aln_tt = tt(
        d_en_bct=d_en_tt,
        ref_s=ref_s,
        input_ids=input_ids,
        input_lengths=input_lengths,
        text_mask=text_mask,
        speed=1.0,
    )
    d_tt_torch = ttnn.to_torch(d_tt).to(torch.float32)
    dur_tt_torch = ttnn.to_torch(dur_tt).to(torch.float32)

    ok_d, pcc_d = comp_pcc(d_ref, d_tt_torch, pcc=0.99)
    assert ok_d, f"d PCC low: {pcc_d}"
    ok_u, pcc_u = comp_pcc(dur_ref, dur_tt_torch, pcc=0.99)
    assert ok_u, f"duration PCC low: {pcc_u}"
