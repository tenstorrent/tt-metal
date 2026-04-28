# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for CosyVoice model configuration and weight remapping.

These tests validate that:
  1. CosyVoiceModelConfig has correct Qwen2-0.5B parameters
  2. Weight key remapping correctly transforms CosyVoice → tt_transformers format
  3. All expected keys are present after remapping
"""

import torch

from models.demos.wormhole.cosy_voice.tt.model_config import CosyVoiceModelConfig, remap_cosyvoice_llm_state_dict


class TestCosyVoiceModelConfig:
    """Test CosyVoiceModelConfig initialization."""

    def test_qwen2_params(self):
        """Verify Qwen2-0.5B architecture parameters."""
        config = CosyVoiceModelConfig(mesh_device=None)
        assert config.dim == 896
        assert config.n_layers == 24
        assert config.n_heads == 14
        assert config.n_kv_heads == 2
        assert config.head_dim == 64
        assert config.hidden_dim == 4864
        assert config.vocab_size == 151936
        assert config.norm_eps == 1e-6

    def test_speech_params(self):
        """Verify CosyVoice speech-specific parameters."""
        config = CosyVoiceModelConfig(mesh_device=None)
        assert config.speech_token_size == 6561
        assert config.speech_vocab_size == 6561 + 200
        assert config.sos_token == 6561
        assert config.eos_token == 6562
        assert config.task_id_token == 6563
        assert config.fill_token == 6564

    def test_derived_params(self):
        """Verify derived parameters are computed correctly."""
        config = CosyVoiceModelConfig(mesh_device=None)
        # head_dim=64, TILE_SIZE=32, so padded_head_dim should be 64
        assert config.padded_head_dim == 64
        # qkv_size = padded_head_dim * (2 * n_kv_heads + n_heads) = 64 * (4 + 14) = 1152
        assert config.qkv_size == 64 * (2 * 2 + 14)


class TestWeightRemapping:
    """Test CosyVoice → tt_transformers weight key remapping."""

    @staticmethod
    def _make_dummy_state_dict():
        """Create a minimal dummy state dict with CosyVoice key structure."""
        sd = {}
        # Embedding
        sd["llm.model.model.embed_tokens.weight"] = torch.randn(151936, 896)
        # Layer 0
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            sd[f"llm.model.model.layers.0.self_attn.{proj}.weight"] = torch.randn(896, 896)
            if proj in ["q_proj", "k_proj", "v_proj"]:
                sd[f"llm.model.model.layers.0.self_attn.{proj}.bias"] = torch.randn(896)
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            sd[f"llm.model.model.layers.0.mlp.{proj}.weight"] = torch.randn(4864, 896)
        sd["llm.model.model.layers.0.input_layernorm.weight"] = torch.randn(896)
        sd["llm.model.model.layers.0.post_attention_layernorm.weight"] = torch.randn(896)
        # Final norm
        sd["llm.model.model.norm.weight"] = torch.randn(896)
        # LM head (Qwen2's text head — we won't use this, but it's in the checkpoint)
        sd["llm.model.lm_head.weight"] = torch.randn(151936, 896)
        # CosyVoice-specific
        sd["speech_embedding.weight"] = torch.randn(6761, 896)
        sd["llm_decoder.weight"] = torch.randn(6761, 896)
        return sd

    def test_embedding_remapped(self):
        sd = self._make_dummy_state_dict()
        remapped, _ = remap_cosyvoice_llm_state_dict(sd)
        assert "tok_embeddings.weight" in remapped

    def test_attention_keys_remapped(self):
        sd = self._make_dummy_state_dict()
        remapped, _ = remap_cosyvoice_llm_state_dict(sd)
        assert "layers.0.attention.wq.weight" in remapped
        assert "layers.0.attention.wk.weight" in remapped
        assert "layers.0.attention.wv.weight" in remapped
        assert "layers.0.attention.wo.weight" in remapped
        # Qwen2 has bias on q, k, v (not o)
        assert "layers.0.attention.wq.bias" in remapped
        assert "layers.0.attention.wk.bias" in remapped
        assert "layers.0.attention.wv.bias" in remapped

    def test_mlp_keys_remapped(self):
        sd = self._make_dummy_state_dict()
        remapped, _ = remap_cosyvoice_llm_state_dict(sd)
        assert "layers.0.feed_forward.w1.weight" in remapped  # gate_proj
        assert "layers.0.feed_forward.w3.weight" in remapped  # up_proj
        assert "layers.0.feed_forward.w2.weight" in remapped  # down_proj

    def test_norm_keys_remapped(self):
        sd = self._make_dummy_state_dict()
        remapped, _ = remap_cosyvoice_llm_state_dict(sd)
        assert "layers.0.attention_norm.weight" in remapped
        assert "layers.0.ffn_norm.weight" in remapped
        assert "norm.weight" in remapped

    def test_lm_head_remapped(self):
        sd = self._make_dummy_state_dict()
        remapped, _ = remap_cosyvoice_llm_state_dict(sd)
        assert "output.weight" in remapped

    def test_cosyvoice_keys_extracted(self):
        sd = self._make_dummy_state_dict()
        _, cosyvoice_keys = remap_cosyvoice_llm_state_dict(sd)
        assert "speech_embedding.weight" in cosyvoice_keys
        assert "llm_decoder.weight" in cosyvoice_keys

    def test_no_original_prefixes_remain(self):
        sd = self._make_dummy_state_dict()
        remapped, _ = remap_cosyvoice_llm_state_dict(sd)
        for key in remapped:
            assert not key.startswith("llm."), f"Unmapped key: {key}"
            assert "self_attn" not in key, f"Unmapped attention key: {key}"
            assert "gate_proj" not in key, f"Unmapped MLP key: {key}"

    def test_weight_shapes_preserved(self):
        """Ensure remapping doesn't alter tensor shapes."""
        sd = self._make_dummy_state_dict()
        remapped, cosyvoice_keys = remap_cosyvoice_llm_state_dict(sd)

        # Embedding shape should be preserved
        assert remapped["tok_embeddings.weight"].shape == (151936, 896)
        # Attention weight shape preserved
        assert remapped["layers.0.attention.wq.weight"].shape == (896, 896)
        # Speech embedding shape preserved
        assert cosyvoice_keys["speech_embedding.weight"].shape == (6761, 896)
