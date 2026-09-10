# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Host-only unit test for LFM2.5-VL HF -> meta key conversion (no device / ttnn needed)."""

import torch

from models.demos.multimodal.lfm25_vl.tt.load_checkpoints import convert_lfm_hf_to_meta


def test_convert_lfm_hf_to_meta_key_renames():
    head_dim = 64
    hidden = 128
    sd = {
        "model.language_model.embed_tokens.weight": torch.randn(32, hidden),
        "model.language_model.embedding_norm.weight": torch.randn(hidden),
        "model.language_model.layers.0.operator_norm.weight": torch.randn(hidden),
        "model.language_model.layers.0.ffn_norm.weight": torch.randn(hidden),
        "model.language_model.layers.0.conv.in_proj.weight": torch.randn(3 * hidden, hidden),
        "model.language_model.layers.0.conv.out_proj.weight": torch.randn(hidden, hidden),
        "model.language_model.layers.0.conv.conv.weight": torch.randn(hidden, 1, 3),
        "model.language_model.layers.0.feed_forward.w1.weight": torch.randn(256, hidden),
        "model.language_model.layers.0.feed_forward.w2.weight": torch.randn(hidden, 256),
        "model.language_model.layers.0.feed_forward.w3.weight": torch.randn(256, hidden),
        "model.language_model.layers.2.self_attn.q_proj.weight": torch.randn(hidden, hidden),
        "model.language_model.layers.2.self_attn.k_proj.weight": torch.randn(hidden, hidden),
        "model.language_model.layers.2.self_attn.v_proj.weight": torch.randn(hidden, hidden),
        "model.language_model.layers.2.self_attn.out_proj.weight": torch.randn(hidden, hidden),
        "model.language_model.layers.2.self_attn.q_layernorm.weight": torch.randn(head_dim),
        "model.language_model.layers.2.self_attn.k_layernorm.weight": torch.randn(head_dim),
        "model.language_model.layers.2.operator_norm.weight": torch.randn(hidden),
        "model.language_model.layers.2.ffn_norm.weight": torch.randn(hidden),
        "model.language_model.layers.2.feed_forward.w1.weight": torch.randn(256, hidden),
        "model.language_model.layers.2.feed_forward.w2.weight": torch.randn(hidden, 256),
        "model.language_model.layers.2.feed_forward.w3.weight": torch.randn(256, hidden),
        "model.vision_tower.embeddings.patch_embedding.weight": torch.randn(16, 48),
        "model.vision_tower.embeddings.position_embedding.weight": torch.randn(4, 16),
        "model.multi_modal_projector.linear_1.weight": torch.randn(32, 64),
        "model.multi_modal_projector.linear_2.weight": torch.randn(hidden, 32),
        "lm_head.weight": torch.randn(32, hidden),
    }

    out = convert_lfm_hf_to_meta(sd, head_dim=head_dim)

    assert "tok_embeddings.weight" in out
    assert "norm.weight" in out
    assert "output.weight" in out
    assert "layers.0.attention_norm.weight" in out
    assert "layers.0.conv.in_proj.weight" in out
    assert "layers.0.conv.out_proj.weight" in out
    assert "layers.0.conv.conv.weight" in out
    assert "layers.0.feed_forward.w1.weight" in out
    assert "layers.2.attention.wq.weight" in out
    assert "layers.2.attention.wo.weight" in out
    assert "layers.2.attention.q_norm.weight" in out
    assert "layers.2.attention.k_norm.weight" in out
    # Vision tower should have vision_model level re-inserted and patch_embedding._linear mapped.
    assert any("vision_tower.vision_model.embeddings" in k for k in out)
    assert "model.multi_modal_projector.linear_1.weight" in out
    # Conv out_proj must NOT be remapped to wo.
    assert "layers.0.conv.out_proj.weight" in out
    assert "layers.0.attention.wo.weight" not in out
