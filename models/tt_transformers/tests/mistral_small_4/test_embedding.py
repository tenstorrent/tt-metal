# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn

from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tt.mistral_small_4.embedding import embedding_lookup_bf16, embedding_lookup_reference_torch


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_mistral_small_4_embedding_matches_torch_and_hf(mesh_device, reset_seeds):
    """Token embedding: ttnn gather vs ``F.embedding`` / ``nn.Embedding`` (Mistral4-style)."""
    torch.manual_seed(0)
    vocab_size, hidden_size = 128, 128
    b, seq = 2, 8

    emb = nn.Embedding(vocab_size, hidden_size, dtype=torch.bfloat16)
    input_ids = torch.randint(0, vocab_size, (b, seq), dtype=torch.long)

    with torch.no_grad():
        expected_hf = emb(input_ids)
        expected_ref = embedding_lookup_reference_torch(input_ids, emb.weight)

    assert torch.allclose(expected_hf, expected_ref)

    out = embedding_lookup_bf16(mesh_device, input_ids, emb.weight.data)

    ok, msg = comp_pcc(expected_ref, out, pcc=0.99)
    assert ok, msg
    close, amsg = comp_allclose(expected_ref, out, rtol=0.08, atol=0.08)
    assert close, amsg
