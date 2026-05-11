# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: full TTNN Kokoro predictor vs PyTorch reference (with host alignment creation)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import ttnn

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference import KokoroConfig, load_predictor_from_huggingface
from models.experimental.kokoro.tt.ttnn_kokoro_predictor import TtKokoroPredictor, preprocess_predictor_full


def _to_torch(x):
    if isinstance(x, ttnn.Tensor):
        return ttnn.to_torch(x).to(torch.float32)
    return x


def _match_last_dim(a, b) -> tuple[torch.Tensor, torch.Tensor]:
    a = _to_torch(a)
    b = _to_torch(b)
    # Predictor heads can be [B,L] in torch but [B,1,L] in TTNN.
    if a.ndim == 2 and b.ndim == 3 and b.shape[1] == 1:
        b = b.squeeze(1)
    if b.ndim == 2 and a.ndim == 3 and a.shape[1] == 1:
        a = a.squeeze(1)
    assert a.ndim == b.ndim
    min_len = min(a.shape[-1], b.shape[-1])
    if min_len != a.shape[-1]:
        a = a[..., :min_len]
    if min_len != b.shape[-1]:
        b = b[..., :min_len]
    return a, b


def test_ttnn_predictor_full_matches_torch(device):
    ref = load_predictor_from_huggingface(repo_id=KokoroConfig.repo_id, device="cpu")
    params = preprocess_predictor_full(ref, device)
    tt = TtKokoroPredictor(device, params)

    torch.manual_seed(0)
    B, T = 1, 32
    d_en = torch.randn(B, params.d_hid, T, dtype=torch.float32)
    d_en_tt = ttnn.from_torch(d_en, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ref_s = torch.randn(B, 256, dtype=torch.float32)
    input_ids = torch.randint(0, 50, (B, T), dtype=torch.long)
    input_lengths = torch.tensor([T], dtype=torch.long)
    text_mask = torch.zeros((B, T), dtype=torch.bool)

    with torch.no_grad():
        out_ref = ref(
            d_en=d_en,
            ref_s=ref_s,
            input_ids=input_ids,
            input_lengths=input_lengths,
            text_mask=text_mask,
            speed=1.0,
        )

    out_tt = tt(
        d_en_bct=d_en_tt,
        ref_s=ref_s,
        input_ids=input_ids,
        input_lengths=input_lengths,
        text_mask=text_mask,
        speed=1.0,
    )

    en_ref, en_tt = _match_last_dim(out_ref.en, out_tt["en"])
    ok_en, pcc_en = comp_pcc(en_ref, en_tt, pcc=0.90)
    assert ok_en, f"en PCC low: {pcc_en}"
    f0_ref, f0_tt = _match_last_dim(out_ref.F0_pred, out_tt["F0_pred"])
    # F0 head is slightly more bf16-sensitive than en/N/asr (device stem is bfloat16).
    ok_f0, pcc_f0 = comp_pcc(f0_ref, f0_tt, pcc=0.875)
    assert ok_f0, f"F0 PCC low: {pcc_f0}"
    n_ref, n_tt = _match_last_dim(out_ref.N_pred, out_tt["N_pred"])
    ok_n, pcc_n = comp_pcc(n_ref, n_tt, pcc=0.90)
    assert ok_n, f"N PCC low: {pcc_n}"
    asr_ref, asr_tt = _match_last_dim(out_ref.asr, out_tt["asr"])
    ok_asr, pcc_asr = comp_pcc(asr_ref, asr_tt, pcc=0.90)
    assert ok_asr, f"asr PCC low: {pcc_asr}"
