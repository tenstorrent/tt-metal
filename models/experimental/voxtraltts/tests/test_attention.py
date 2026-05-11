# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.voxtraltts.reference.functional import (
    VoxtralTextConfig,
    text_attention as reference_text_attention,
)
from models.experimental.voxtraltts.tt.attention import VoxtralTTAttention
from models.experimental.voxtraltts.tt.rope import compute_rope_frequencies


@torch.no_grad()
def test_voxtral_text_attention_pcc(device, reset_seeds):
    batch = 1
    seq_len = 64
    hidden = 512
    n_heads = 8
    n_kv_heads = 2
    head_dim = hidden // n_heads

    config = VoxtralTextConfig(
        hidden_size=hidden,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        head_dim=head_dim,
    )

    torch_input = torch.randn(batch, 1, seq_len, hidden, dtype=torch.bfloat16)
    wq = torch.randn(hidden, hidden, dtype=torch.bfloat16)
    wk = torch.randn(n_kv_heads * head_dim, hidden, dtype=torch.bfloat16)
    wv = torch.randn(n_kv_heads * head_dim, hidden, dtype=torch.bfloat16)
    wo = torch.randn(hidden, hidden, dtype=torch.bfloat16)
    layer_weights = {
        "attention.wq.weight": wq,
        "attention.wk.weight": wk,
        "attention.wv.weight": wv,
        "attention.wo.weight": wo,
    }

    cos, sin = compute_rope_frequencies(
        head_dim=head_dim,
        max_seq_len=seq_len,
        theta=config.rope_theta,
        device=torch.device("cpu"),
    )
    reference_output = reference_text_attention(
        hidden_states=torch_input.squeeze(1),
        layer_weights=layer_weights,
        cos=cos,
        sin=sin,
        config=config,
        attention_mask=None,
    )

    tt_model = VoxtralTTAttention(
        device=device,
        hidden_size=hidden,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        head_dim=head_dim,
        state_dict=layer_weights,
        weight_prefix="attention",
    )

    tt_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = tt_model(tt_input, cos=cos, sin=sin)
    tt_output_torch = ttnn.to_torch(tt_output).squeeze(1)

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc=0.985)
    assert passing, f"Voxtral text attention PCC failed: {pcc_message}"
