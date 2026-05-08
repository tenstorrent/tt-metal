# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn

from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tt.mistral_small_4.lm_head import lm_head_logits_bf16, lm_head_logits_reference_torch


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_lm_head_logits_match_torch_linear(mesh_device, reset_seeds):
    """LM head as bias-free linear: ttnn vs ``nn.Linear`` / ``F.linear`` (Mistral4-style)."""
    torch.manual_seed(0)
    hidden_size, vocab_size = 128, 128
    b, seq = 2, 8

    head = nn.Linear(hidden_size, vocab_size, bias=False, dtype=torch.bfloat16)
    x = torch.randn(b, seq, hidden_size, dtype=torch.bfloat16)

    with torch.no_grad():
        expected_hf = head(x)
        expected_ref = lm_head_logits_reference_torch(x, head.weight.data)

    assert torch.allclose(expected_hf, expected_ref)

    out = lm_head_logits_bf16(mesh_device, x, head.weight.data)

    ok, msg = comp_pcc(expected_ref, out, pcc=0.99)
    assert ok, msg
    close, amsg = comp_allclose(expected_ref, out, rtol=0.08, atol=0.08)
    assert close, amsg
