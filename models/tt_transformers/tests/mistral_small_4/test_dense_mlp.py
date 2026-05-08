# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tt.mistral_small_4.dense_mlp import dense_mlp_bf16


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_mistral_small_4_dense_mlp_matches_hf(mesh_device, reset_seeds):
    from transformers.models.mistral4.configuration_mistral4 import Mistral4Config
    from transformers.models.mistral4.modeling_mistral4 import Mistral4MLP

    torch.manual_seed(0)
    hidden_size = 128
    intermediate_size = 256
    b, seq = 1, 16

    cfg = Mistral4Config(
        vocab_size=128,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=4,
    )
    mlp = Mistral4MLP(cfg).to(torch.bfloat16).eval()
    x = torch.randn(b, seq, hidden_size, dtype=torch.bfloat16)

    with torch.no_grad():
        expected = mlp(x)

    out = dense_mlp_bf16(
        mesh_device,
        x,
        mlp.gate_proj.weight,
        mlp.up_proj.weight,
        mlp.down_proj.weight,
    )

    ok, msg = comp_pcc(expected, out, pcc=0.97)
    assert ok, msg
    close, amsg = comp_allclose(expected, out, rtol=0.12, atol=0.15)
    assert close, amsg
