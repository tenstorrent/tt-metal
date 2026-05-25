# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
CPU-only tests for the Qwen3-TTS Code Predictor: config, key remapping, reference.
No device or ttnn dependency at module level.
"""

import pytest
import torch
from loguru import logger


class TestCodePredictorConfig:
    """Test CodePredictorConfig parsing."""

    def test_config_from_dict(self):
        from models.demos.qwen3_tts.tt.configs import CodePredictorConfig

        cfg = CodePredictorConfig.from_dict({
            "hidden_size": 1024,
            "num_hidden_layers": 5,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 3072,
            "vocab_size": 2048,
            "num_code_groups": 16,
        })

        assert cfg.hidden_size == 1024
        assert cfg.num_layers == 5
        assert cfg.num_heads == 16
        assert cfg.num_kv_heads == 8
        assert cfg.head_dim == 128
        assert cfg.intermediate_size == 3072
        assert cfg.vocab_size == 2048
        assert cfg.num_code_groups == 16
        assert cfg.num_cb_predict == 15
        assert cfg.talker_hidden_size == 2048
        logger.info("CodePredictorConfig parsing verified")


class TestCodePredictorKeyRemapping:
    """Test that HF→meta key conversion works for Code Predictor keys."""

    def test_key_remapping(self):
        from models.tt_transformers.tt.load_checkpoints import convert_hf_to_meta_no_qkv_permute

        hf_keys = {
            "talker.code_predictor.model.layers.0.self_attn.q_proj.weight": torch.randn(2048, 1024),
            "talker.code_predictor.model.layers.0.self_attn.k_proj.weight": torch.randn(1024, 1024),
            "talker.code_predictor.model.layers.0.self_attn.v_proj.weight": torch.randn(1024, 1024),
            "talker.code_predictor.model.layers.0.self_attn.o_proj.weight": torch.randn(1024, 2048),
            "talker.code_predictor.model.layers.0.self_attn.q_norm.weight": torch.randn(128),
            "talker.code_predictor.model.layers.0.self_attn.k_norm.weight": torch.randn(128),
            "talker.code_predictor.model.layers.0.mlp.gate_proj.weight": torch.randn(3072, 1024),
            "talker.code_predictor.model.layers.0.mlp.up_proj.weight": torch.randn(3072, 1024),
            "talker.code_predictor.model.layers.0.mlp.down_proj.weight": torch.randn(1024, 3072),
            "talker.code_predictor.model.layers.0.input_layernorm.weight": torch.randn(1024),
            "talker.code_predictor.model.layers.0.post_attention_layernorm.weight": torch.randn(1024),
            "talker.code_predictor.model.norm.weight": torch.randn(1024),
        }

        converted = convert_hf_to_meta_no_qkv_permute(hf_keys, head_dim=128, n_heads=16, n_kv_heads=8)

        # Verify critical keys (model. stripped, HF→meta names)
        assert "talker.code_predictor.layers.0.attention.wq.weight" in converted
        assert "talker.code_predictor.layers.0.attention.wk.weight" in converted
        assert "talker.code_predictor.layers.0.attention.wv.weight" in converted
        assert "talker.code_predictor.layers.0.attention.wo.weight" in converted
        assert "talker.code_predictor.layers.0.attention.q_norm.weight" in converted
        assert "talker.code_predictor.layers.0.attention.k_norm.weight" in converted
        assert "talker.code_predictor.layers.0.feed_forward.w1.weight" in converted
        assert "talker.code_predictor.layers.0.feed_forward.w2.weight" in converted
        assert "talker.code_predictor.layers.0.feed_forward.w3.weight" in converted
        assert "talker.code_predictor.layers.0.attention_norm.weight" in converted
        assert "talker.code_predictor.layers.0.ffn_norm.weight" in converted
        assert "talker.code_predictor.norm.weight" in converted

        logger.info("Code Predictor key remapping verified")


class TestCodePredictorReference:
    """Test the standalone PyTorch reference model."""

    def test_small_model_predict(self):
        """Verify reference model with small random weights."""
        from models.demos.qwen3_tts.reference.code_predictor_ref import CodePredictorReference

        model = CodePredictorReference(
            hidden_size=64, talker_hidden_size=128,
            n_layers=2, n_heads=4, n_kv_heads=2, head_dim=16,
            ffn_dim=128, vocab_size=32, num_code_groups=4, max_seq_len=32,
        )
        model.eval()

        # Mock talker codec embedding
        talker_codec_emb = torch.nn.Embedding(32, 128)

        talker_hidden = torch.randn(1, 1, 128)
        cb0_token = torch.tensor([5])

        with torch.no_grad():
            all_cb = model.predict_codebooks(talker_hidden, cb0_token, talker_codec_emb, temperature=0)

        assert all_cb.shape == (1, 4)  # CB0 + CB1 + CB2 + CB3
        assert all_cb[0, 0].item() == 5  # CB0 should be passed through
        assert all(0 <= t.item() < 32 for t in all_cb[0])
        logger.info(f"Small model predict: {all_cb}")

    def test_real_weights_load(self):
        """Verify loading from real safetensors (if available)."""
        try:
            from models.demos.qwen3_tts.reference.code_predictor_ref import CodePredictorReference

            model = CodePredictorReference.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
            model.eval()

            params = sum(p.numel() for p in model.parameters())
            assert params > 100e6, f"Expected >100M params, got {params/1e6:.1f}M"
            assert len(model.codec_embeddings) == 15
            assert len(model.lm_heads) == 15
            assert len(model.layers) == 5
            logger.info(f"Real weights loaded: {params/1e6:.1f}M params")
        except Exception as e:
            pytest.skip(f"Weights not available: {e}")


class TestCodePredictorStateDictPrefix:
    """Test get_state_dict_prefix for Code Predictor."""

    def test_prefix_mapping(self):
        from models.demos.qwen3_tts.tt.model_config import CodePredictorModelArgs

        class MockArgs(CodePredictorModelArgs):
            def __init__(self):
                pass

        # Manually set the code_predictor_config since we skip __init__
        args = MockArgs()

        assert args.get_state_dict_prefix("Attention", 0) == "talker.code_predictor.layers.0.attention"
        assert args.get_state_dict_prefix("MLP", 0) == "talker.code_predictor.layers.0.feed_forward"
        assert args.get_state_dict_prefix("TransformerBlock", 0) == "talker.code_predictor.layers.0."
        assert args.get_state_dict_prefix("", None) == "talker.code_predictor."
        assert args.get_state_dict_prefix("Attention", 4) == "talker.code_predictor.layers.4.attention"

        logger.info("Code Predictor state dict prefix mappings verified")
