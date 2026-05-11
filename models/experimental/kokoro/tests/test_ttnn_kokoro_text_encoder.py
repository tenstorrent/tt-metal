# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: TTNN Kokoro TextEncoder vs PyTorch reference TextEncoder."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference import KokoroConfig, load_predictor_from_huggingface
from models.experimental.kokoro.tt.ttnn_kokoro_text_encoder import TtKokoroTextEncoder, preprocess_text_encoder


def test_ttnn_text_encoder_matches_torch(device):
    torch_ref = load_predictor_from_huggingface(repo_id=KokoroConfig.repo_id, device="cpu")
    ref_enc = torch_ref.text_encoder

    params = preprocess_text_encoder(ref_enc, device)
    tt_enc = TtKokoroTextEncoder(device, params)

    torch.manual_seed(0)
    input_ids = torch.randint(0, 50, (1, 32), dtype=torch.long)
    input_lengths = torch.tensor([32], dtype=torch.long)
    text_mask = torch.zeros((1, 32), dtype=torch.bool)

    with torch.no_grad():
        ref_out = ref_enc(input_ids, input_lengths, text_mask)

    tt_out = tt_enc(input_ids, input_lengths, text_mask)
    import ttnn

    tt_out_torch = ttnn.to_torch(tt_out).to(torch.float32)
    ttnn.deallocate(tt_out)

    assert ref_out.shape == tt_out_torch.shape
    ok, pcc = comp_pcc(ref_out, tt_out_torch, pcc=0.93)
    assert ok, f"text_encoder PCC too low: {pcc}"
