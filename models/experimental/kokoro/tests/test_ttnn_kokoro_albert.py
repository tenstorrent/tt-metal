# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: TTNN Kokoro Albert vs HuggingFace `AlbertModel` (last_hidden_state + d_en)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

import ttnn

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference import KokoroConfig, load_plbert_from_huggingface
from models.experimental.kokoro.tt.preprocess_kokoro_albert import preprocess_kokoro_albert_for_ttnn
from models.experimental.kokoro.tt.preprocessing import preprocess_bert_encoder_linear
from models.experimental.kokoro.tt.ttnn_kokoro_albert import TtKokoroAlbert
from models.experimental.kokoro.tt.ttnn_kokoro_plbert_projection import TtKokoroPlBertProjection


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_ttnn_albert_and_projection_match_torch(mesh_device):
    torch_ref = load_plbert_from_huggingface(repo_id=KokoroConfig.repo_id, device="cpu")
    params = preprocess_kokoro_albert_for_ttnn(torch_ref.albert, mesh_device)
    tt_albert = TtKokoroAlbert(mesh_device, torch_ref.albert.config, params)
    enc_params = preprocess_bert_encoder_linear(torch_ref.bert_encoder, device=mesh_device)
    proj = TtKokoroPlBertProjection(mesh_device, parameters=enc_params)

    torch.manual_seed(0)
    input_ids = torch.randint(0, 50, (1, 32), dtype=torch.long)

    text_mask = torch.zeros((1, 32), dtype=torch.bool)
    attn = (~text_mask).int()

    with torch.no_grad():
        ref_out = torch_ref.albert(input_ids=input_ids, attention_mask=attn)
        bert_ref = ref_out.last_hidden_state
        d_en_ref = torch_ref.bert_encoder(bert_ref).transpose(-1, -2)

    bert_tt = tt_albert(input_ids, attn.float())
    d_en_tt = proj(bert_tt)
    bert_tt_cpu = ttnn.to_torch(bert_tt).to(torch.float32)
    d_en_cpu = ttnn.to_torch(d_en_tt).to(torch.float32)
    ttnn.deallocate(bert_tt)
    ttnn.deallocate(d_en_tt)

    ok_b, pcc_b = comp_pcc(bert_ref, bert_tt_cpu, pcc=0.96)
    assert ok_b, f"bert_dur PCC low: {pcc_b}"
    ok_d, pcc_d = comp_pcc(d_en_ref, d_en_cpu, pcc=0.94)
    assert ok_d, f"d_en PCC low: {pcc_d}"
