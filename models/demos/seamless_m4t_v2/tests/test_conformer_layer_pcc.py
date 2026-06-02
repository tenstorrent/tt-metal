# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Phase 3a PCC: TtConformerEncoderLayer vs HF SeamlessM4Tv2ConformerEncoderLayer.
Randomly-initialized reference (no checkpoint).
"""

import pytest
import torch
from transformers import SeamlessM4Tv2Config
from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import (
    SeamlessM4Tv2ConformerEncoderLayer,
)

import ttnn
from models.demos.seamless_m4t_v2.tt.conformer_encoder import TtConformerEncoderLayer
from models.demos.seamless_m4t_v2.tt.model_config import SeamlessS2TTConfig
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("seq_len", [128, 384])
def test_conformer_layer_pcc(device, seq_len):
    torch.manual_seed(0)
    hf_config = SeamlessM4Tv2Config()
    s2tt = SeamlessS2TTConfig()
    C = s2tt.hidden_size

    ref = SeamlessM4Tv2ConformerEncoderLayer(hf_config).eval()
    with torch.no_grad():
        for name, p in ref.named_parameters():
            if name.endswith("layer_norm.weight"):
                p.copy_(1.0 + 0.05 * torch.randn_like(p))
            elif name.endswith("layer_norm.bias"):
                p.copy_(0.05 * torch.randn_like(p))

    x = torch.randn(1, seq_len, C)
    with torch.no_grad():
        golden, _ = ref(x, attention_mask=None, output_attentions=False, conv_attention_mask=None)

    tt_mod = TtConformerEncoderLayer(ref.state_dict(), s2tt, device, dtype=ttnn.bfloat16)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(tt_mod(x_tt))

    assert out.shape == golden.shape
    assert_with_pcc(golden, out, pcc=0.99)
