# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The whole attention block, one-shot, at real dims (96 q / 8 kv heads, head_dim 128, hidden 12288):
QKV proj -> head split -> YaRN RoPE -> causal ring-joint SDPA over SP -> o_proj + TP reduce-scatter.
Same random weights on both sides; the device rope tables are built from the same reference cos/sin,
so the test measures attention rather than the RoPE constants (those are pinned in test_rope_vs_ref)."""

import pytest
import torch

from models.demos.mistral_medium_3_5_128b.reference.model import (
    ReferenceAttention,
    random_layer_state_dict,
    rope_cos_sin,
)
from models.demos.mistral_medium_3_5_128b.tt.attention import Attention
from models.demos.mistral_medium_3_5_128b.tt.rope import RopeSetup

from .common import CFG, assert_pcc, randn, residual_to_torch, spec_dtypes, to_full


def attention_weights(seed):
    return {
        k[len("self_attn.") :]: v
        for k, v in random_layer_state_dict(CFG, seed=seed).items()
        if k.startswith("self_attn.")
    }


@pytest.mark.timeout(900)
@pytest.mark.parametrize("seq_len", [10240])
def test_attention_prefill_vs_ref(galaxy_mesh, mesh_config, ccl_manager, seq_len):
    sd = attention_weights(31)
    ref_attn = ReferenceAttention(CFG).to(torch.bfloat16).eval()
    ref_attn.load_state_dict(sd)
    x = randn(1, 1, seq_len, CFG.hidden_size, seed=32)
    pos = torch.arange(seq_len)
    cos, sin = rope_cos_sin(CFG, pos)
    with torch.no_grad():
        ref, _, _ = ref_attn(x[0], cos, sin, pos)

    rope = RopeSetup(galaxy_mesh, mesh_config, CFG, max_seq_len=seq_len, chunk_size=seq_len)
    attn = Attention(galaxy_mesh, mesh_config, ccl_manager, CFG, sd, weight_dtype=spec_dtypes()["attention"])
    out = residual_to_torch(attn(to_full(x, galaxy_mesh, mesh_config), rope), galaxy_mesh, mesh_config)
    assert_pcc("attention_one_shot", ref[None], out)
