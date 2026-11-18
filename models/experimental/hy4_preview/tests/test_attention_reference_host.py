# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Anchor: the standalone attention reference must track HF transformers' HYV4Attention.

reference/attention.py is what the device PCC tests validate against; this pins it to the HF
module run with the eager core. The HF layer is built as a ``"shared"`` indexer layer and handed
top-k indices that cover every position, so its DSA mask is empty and both sides compute dense
causal attention; a second case hands it a real subset to pin how the sparse mask composes.
Pure CPU (torch vs torch), no device. Skipped when the installed transformers predates ``hy_v4``.
"""

import pytest
import torch

from models.experimental.hy4_preview.reference.attention import AttentionConfig, Hy4Attention, causal_mask, rope_cos_sin

hy_v4 = pytest.importorskip("transformers.models.hy_v4.modeling_hy_v4")
from transformers.models.hy_v4.configuration_hy_v4 import HYV4Config  # noqa: E402

# checkpoint name -> HF module attribute
HF_NAMES = {
    "q_a_proj": "q_a_proj.weight",
    "q_a_layernorm": "q_a_layernorm.weight",
    "q_b_proj": "q_b_proj.weight",
    "kv_a_proj_with_mqa": "kv_a_proj_with_mqa.weight",
    "kv_a_layernorm": "kv_a_layernorm.weight",
    "kv_b_proj": "kv_b_proj.weight",
    "o_proj": "o_proj.weight",
    "linear_gate": "gate_proj.weight",
    "learnable_sink_param": "sinks",
}

SMALL = AttentionConfig(
    hidden_size=96,
    num_heads=4,
    q_lora_rank=48,
    kv_lora_rank=32,
    qk_nope_head_dim=24,
    qk_rope_head_dim=8,
    v_head_dim=16,
)


def _hf_config(cfg: AttentionConfig) -> HYV4Config:
    return HYV4Config(
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_heads,
        num_key_value_heads=cfg.num_heads,
        head_dim=cfg.qk_rope_head_dim,
        num_hidden_layers=2,
        q_lora_rank=cfg.q_lora_rank,
        kv_lora_rank=cfg.kv_lora_rank,
        qk_nope_head_dim=cfg.qk_nope_head_dim,
        qk_rope_head_dim=cfg.qk_rope_head_dim,
        v_head_dim=cfg.v_head_dim,
        indexer_types=["full", "shared"],
        mlp_layer_types=["dense", "sparse"],
        max_position_embeddings=4096,
        rope_parameters={"rope_type": "default", "rope_theta": cfg.rope_theta},
        attn_implementation="eager",
    )


def _pair(cfg: AttentionConfig, seed: int):
    """An HF shared-indexer attention layer and the reference, holding the same random parameters."""
    hf_cfg = _hf_config(cfg)
    hf = hy_v4.HYV4Attention(hf_cfg, layer_idx=1).eval()
    ref = Hy4Attention(cfg).eval()
    g = torch.Generator().manual_seed(seed)
    hf_state = dict(hf.named_parameters())
    with torch.no_grad():
        for name, hf_name in HF_NAMES.items():
            p = getattr(ref, name)
            if p.dim() == 2:
                value = torch.randn(p.shape, generator=g) * p.shape[1] ** -0.5
            else:  # norm weights and sinks: away from their trivial init
                value = 1.0 + 0.5 * torch.randn(p.shape, generator=g)
            p.copy_(value)
            assert hf_state[hf_name].shape == p.shape, (name, hf_state[hf_name].shape, p.shape)
            hf_state[hf_name].copy_(value)
    return hf_cfg, hf, ref


@pytest.mark.parametrize("seq_len, topk", [(12, None), (40, None), (40, 9)], ids=["S12", "S40", "S40-top9"])
def test_attention_matches_hf(seq_len, topk):
    cfg = SMALL
    hf_cfg, hf, ref = _pair(cfg, seed=3)
    g = torch.Generator().manual_seed(4)
    x = torch.randn(seq_len, cfg.hidden_size, generator=g)
    positions = torch.arange(seq_len) + 5  # a non-zero start, so RoPE is exercised off the origin

    hf_mask = causal_mask(seq_len)[None, None]
    if topk is None:  # every position selected: the DSA mask is empty
        indices = torch.arange(seq_len).expand(1, seq_len, seq_len)
        mask = causal_mask(seq_len)
    else:  # an arbitrary subset per query; causality still comes from the causal mask
        indices = torch.stack([torch.randperm(seq_len, generator=g)[:topk] for _ in range(seq_len)])[None]
        keep = torch.zeros(seq_len, seq_len, dtype=torch.bool).scatter(-1, indices[0], True)
        mask = causal_mask(seq_len).masked_fill(~keep, torch.finfo(torch.float32).min)

    with torch.no_grad():
        position_embeddings = hy_v4.HYV4RotaryEmbedding(hf_cfg)(x[None], positions[None])
        hf_out, _, _ = hf(
            x[None],
            position_embeddings=position_embeddings,
            attention_mask=hf_mask,
            prev_topk_indices=indices.to(torch.int32),
        )
        cos, sin = rope_cos_sin(positions, cfg)
        out = ref(x, cos, sin, mask)

    assert torch.equal(cos, position_embeddings[0][0]) and torch.equal(sin, position_embeddings[1][0])
    max_diff = (out - hf_out[0]).abs().max().item()
    assert max_diff <= 1e-5 * hf_out.abs().max().item(), f"max|Δ|={max_diff:.3e}"


def test_sink_takes_probability_mass():
    """A large sink logit must shrink the output towards zero; without it the test above could pass
    with the sink wired to nothing on both sides."""
    cfg = SMALL
    _, _, ref = _pair(cfg, seed=5)
    x = torch.randn(8, cfg.hidden_size, generator=torch.Generator().manual_seed(6))
    cos, sin = rope_cos_sin(torch.arange(8), cfg)
    with torch.no_grad():
        base = ref(x, cos, sin, causal_mask(8))
        ref.learnable_sink_param.fill_(30.0)
        sunk = ref(x, cos, sin, causal_mask(8))
    assert sunk.abs().max() < 1e-6 * base.abs().max()
