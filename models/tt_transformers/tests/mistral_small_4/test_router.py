# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tt.mistral_small_4.router import router_logits_bf16


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_mistral_small_4_router_logits_match_hf(mesh_device, reset_seeds):
    from transformers.models.mistral4.configuration_mistral4 import Mistral4Config
    from transformers.models.mistral4.modeling_mistral4 import Mistral4TopkRouter

    torch.manual_seed(0)
    hidden_size = 128
    n_routed_experts = 64
    b, seq = 2, 8

    cfg = Mistral4Config(
        vocab_size=128,
        hidden_size=hidden_size,
        n_routed_experts=n_routed_experts,
        num_hidden_layers=1,
        num_attention_heads=4,
    )
    gate = Mistral4TopkRouter(cfg).to(torch.bfloat16).eval()
    torch.nn.init.normal_(gate.weight, std=0.02)

    x = torch.randn(b, seq, hidden_size, dtype=torch.bfloat16)
    with torch.no_grad():
        expected = gate(x)

    out = router_logits_bf16(mesh_device, x, gate.weight.data)

    ok, msg = comp_pcc(expected, out, pcc=0.99)
    assert ok, msg
    close, amsg = comp_allclose(expected, out, rtol=0.08, atol=0.08)
    assert close, amsg
