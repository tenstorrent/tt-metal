# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Phase 2 PCC: TtConformerSelfAttention vs HF SeamlessM4Tv2ConformerSelfAttention.

Covers both the encoder variant (Shaw relative_key position bias) and the adapter
variant (use_position_embeddings=False). Randomly-initialized reference, no
checkpoint needed.
"""

import pytest
import torch
from transformers import SeamlessM4Tv2Config
from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import (
    SeamlessM4Tv2ConformerSelfAttention,
)

import ttnn
from models.demos.seamless_m4t_v2.tt.conformer_attention import TtConformerSelfAttention
from models.demos.seamless_m4t_v2.tt.model_config import SeamlessS2TTConfig
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("use_pos", [True, False])
@pytest.mark.parametrize("seq_len", [128, 384])
def test_conformer_attention_pcc(device, seq_len, use_pos):
    torch.manual_seed(0)
    hf_config = SeamlessM4Tv2Config()
    s2tt = SeamlessS2TTConfig()
    C = s2tt.hidden_size

    ref = SeamlessM4Tv2ConformerSelfAttention(hf_config, use_position_embeddings=use_pos).eval()
    x = torch.randn(1, seq_len, C)
    with torch.no_grad():
        golden, _ = ref(x, attention_mask=None, output_attentions=False)

    tt_mod = TtConformerSelfAttention(
        ref.state_dict(), s2tt, device, use_position_embeddings=use_pos, dtype=ttnn.bfloat16
    )
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(tt_mod(x_tt))

    assert out.shape == golden.shape
    assert_with_pcc(golden, out, pcc=0.99)
