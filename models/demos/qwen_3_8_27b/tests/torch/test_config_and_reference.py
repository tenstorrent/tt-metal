# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""D1 (host only): config constants vs config.json, and the vendored reference vs upstream HF math.

Pattern: deepseek_v3_d_p/tests/torch/test_kimi_k3_mla_reference.py.
"""

import os
from pathlib import Path

import pytest
import torch

from models.demos.qwen_3_8_27b.config import QWEN38, VENDORED_CONFIG, Qwen38Config
from models.demos.qwen_3_8_27b.reference import qwen3_8_ref as ref

hf_mod = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@pytest.mark.parametrize(
    "path",
    [
        VENDORED_CONFIG,
        Path(os.environ.get("PREFILL_HF_MODEL", os.environ.get("HF_MODEL", "/nonexistent"))) / "config.json",
    ],
    ids=["vendored", "checkpoint"],
)
def test_constants_match_config_json(path):
    if not Path(path).exists():
        pytest.skip(f"{path} not present")
    parsed = Qwen38Config.from_hf_json(path)
    for f in Qwen38Config.__dataclass_fields__:
        assert getattr(parsed, f) == getattr(
            QWEN38, f
        ), f"{f}: config.json {getattr(parsed, f)} != {getattr(QWEN38, f)}"
    # derived values the implementation relies on
    assert QWEN38.rotary_dim == 64
    assert QWEN38.conv_dim == 10240
    assert QWEN38.full_attention_layers == list(range(3, 64, 4))
    assert sum(QWEN38.mrope_section) * 2 == QWEN38.rotary_dim


# ------------------------------------------------------------------------------------------------
# reduced config used to check the vendored reference against upstream
# ------------------------------------------------------------------------------------------------
SMALL = QWEN38.reduced(
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=4,
    vocab_size=512,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    mrope_section=(3, 3, 2),
    linear_num_key_heads=2,
    linear_num_value_heads=6,
    linear_key_head_dim=32,
    linear_value_head_dim=32,
)


def hf_config(cfg: Qwen38Config):
    c = Qwen3_5TextConfig(
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        vocab_size=cfg.vocab_size,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        rms_norm_eps=cfg.rms_norm_eps,
        linear_num_key_heads=cfg.linear_num_key_heads,
        linear_num_value_heads=cfg.linear_num_value_heads,
        linear_key_head_dim=cfg.linear_key_head_dim,
        linear_value_head_dim=cfg.linear_value_head_dim,
        linear_conv_kernel_dim=cfg.linear_conv_kernel_dim,
        layer_types=list(cfg.layer_types),
        full_attention_interval=cfg.full_attention_interval,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": cfg.rope_theta,
            "partial_rotary_factor": cfg.partial_rotary_factor,
            "mrope_section": list(cfg.mrope_section),
            "mrope_interleaved": True,
        },
    )
    c._attn_implementation = "eager"
    return c


def build_pair(seed=0):
    ours = ref.init_random_(ref.TextModel(SMALL), seed=seed).float()
    theirs = hf_mod.Qwen3_5TextModel(hf_config(SMALL)).float()
    missing, unexpected = theirs.load_state_dict(
        {k: v for k, v in ours.state_dict().items() if not k.startswith("lm_head")}, strict=False
    )
    assert not unexpected, unexpected
    assert all("rotary" in m for m in missing), missing
    return ours.eval(), theirs.eval()


def test_rope_matches_upstream_interleaved_mrope():
    cfg = QWEN38
    rot = hf_mod.Qwen3_5TextRotaryEmbedding(hf_config(cfg))
    pos = torch.arange(0, 10240, 37)
    x = torch.zeros(1, 1, dtype=torch.float32)
    cos_hf, sin_hf = rot(x, pos[None, :])
    cos, sin = ref.rope_cos_sin(cfg, pos, dtype=torch.float32)
    assert torch.allclose(cos_hf[0], cos, atol=1e-5) and torch.allclose(sin_hf[0], sin, atol=1e-5)


def test_delta_rule_matches_upstream():
    torch.manual_seed(0)
    B, T, H, D = 1, 200, 3, 32
    q, k, v = (torch.randn(B, T, H, D) for _ in range(3))
    g = -torch.rand(B, T, H)
    beta = torch.rand(B, T, H)
    s0 = torch.randn(B, H, D, D) * 0.1
    o_hf, s_hf = hf_mod.torch_chunk_gated_delta_rule(
        q, k, v, g, beta, initial_state=s0, output_final_state=True, use_qk_l2norm_in_kernel=True
    )
    o, s = ref.chunk_gated_delta_rule(q, k, v, g, beta, initial_state=s0)
    assert pcc(o, o_hf) > 0.99999 and pcc(s, s_hf) > 0.99999
    o_r, s_r = ref.recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=s0)
    assert pcc(o, o_r) > 0.9999 and pcc(s, s_r) > 0.9999


def test_reference_model_matches_upstream_hf():
    """Whole reduced model (3 GDN + 1 attention layer) vs upstream Qwen3_5TextModel, incl. cache contents."""
    ours, theirs = build_pair()
    torch.manual_seed(1)
    ids = torch.randint(0, SMALL.vocab_size, (1, 96))
    with torch.no_grad():
        out = theirs(input_ids=ids, use_cache=True)
        h, states = ours(ids)
    assert pcc(h, out.last_hidden_state) > 0.99999
    cache = out.past_key_values
    for i, st in enumerate(states):
        layer = cache.layers[i]
        if SMALL.is_full_attention(i):
            assert pcc(st["k"], layer.keys) > 0.99999 and pcc(st["v"], layer.values) > 0.99999
        else:
            assert pcc(st["recurrent_state"], layer.recurrent_states) > 0.99999
            # upstream keeps a K-wide conv window; the last K-1 columns are the carried state
            assert torch.allclose(st["conv_state"], layer.conv_states[..., 1:])
