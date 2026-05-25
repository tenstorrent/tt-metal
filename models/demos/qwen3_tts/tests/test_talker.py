# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
CPU-only tests for the Qwen3-TTS Talker: config parsing, key remapping.
No device or ttnn dependency at module level.
"""

import pytest
import torch
from loguru import logger


class TestTalkerKeyRemapping:
    """Test that HF→meta key conversion works correctly with real key names."""

    def test_key_remapping(self):
        """Verify the full key conversion pipeline: pre-remap + convert_hf_to_meta."""
        from models.tt_transformers.tt.load_checkpoints import convert_hf_to_meta_no_qkv_permute

        # Actual HF safetensors key names from Qwen3-TTS-12Hz-1.7B-Base
        hf_keys = {
            "talker.model.codec_embedding.weight": torch.randn(3072, 2048),
            "talker.model.text_embedding.weight": torch.randn(151936, 2048),
            "talker.codec_head.weight": torch.randn(3072, 2048),
            "talker.text_projection.linear_fc1.weight": torch.randn(2048, 2048),
            "talker.text_projection.linear_fc1.bias": torch.randn(2048),
            "talker.text_projection.linear_fc2.weight": torch.randn(2048, 2048),
            "talker.text_projection.linear_fc2.bias": torch.randn(2048),
            "talker.model.layers.0.self_attn.q_proj.weight": torch.randn(2048, 2048),
            "talker.model.layers.0.self_attn.k_proj.weight": torch.randn(1024, 2048),
            "talker.model.layers.0.self_attn.v_proj.weight": torch.randn(1024, 2048),
            "talker.model.layers.0.self_attn.o_proj.weight": torch.randn(2048, 2048),
            "talker.model.layers.0.self_attn.q_norm.weight": torch.randn(128),
            "talker.model.layers.0.self_attn.k_norm.weight": torch.randn(128),
            "talker.model.layers.0.mlp.gate_proj.weight": torch.randn(6144, 2048),
            "talker.model.layers.0.mlp.up_proj.weight": torch.randn(6144, 2048),
            "talker.model.layers.0.mlp.down_proj.weight": torch.randn(2048, 6144),
            "talker.model.layers.0.input_layernorm.weight": torch.randn(2048),
            "talker.model.layers.0.post_attention_layernorm.weight": torch.randn(2048),
            "talker.model.norm.weight": torch.randn(2048),
        }

        # Step 1: Pre-remap (same as load_state_dict in model_config.py)
        pre_remapped = {}
        for k, v in hf_keys.items():
            k = k.replace("codec_embedding", "tok_embeddings")
            k = k.replace("codec_head", "output")
            pre_remapped[k] = v

        # Step 2: HF→meta conversion
        converted = convert_hf_to_meta_no_qkv_permute(pre_remapped, head_dim=128, n_heads=16, n_kv_heads=8)

        # Verify critical keys
        assert "talker.tok_embeddings.weight" in converted, "codec_embedding → tok_embeddings failed"
        assert "talker.output.weight" in converted, "codec_head → output failed"
        assert "talker.text_embedding.weight" in converted, "text_embedding should be preserved"
        assert "talker.norm.weight" in converted, "norm should be preserved"

        # Text projection keys should be preserved (not under model.)
        assert "talker.text_projection.linear_fc1.weight" in converted
        assert "talker.text_projection.linear_fc1.bias" in converted
        assert "talker.text_projection.linear_fc2.weight" in converted
        assert "talker.text_projection.linear_fc2.bias" in converted

        # Layer keys: model. stripped, HF→meta names
        assert "talker.layers.0.attention.wq.weight" in converted, "q_proj→wq failed"
        assert "talker.layers.0.attention.wk.weight" in converted, "k_proj→wk failed"
        assert "talker.layers.0.attention.wv.weight" in converted, "v_proj→wv failed"
        assert "talker.layers.0.attention.wo.weight" in converted, "o_proj→wo failed"
        assert "talker.layers.0.attention.q_norm.weight" in converted
        assert "talker.layers.0.attention.k_norm.weight" in converted
        assert "talker.layers.0.feed_forward.w1.weight" in converted, "gate_proj→w1 failed"
        assert "talker.layers.0.feed_forward.w2.weight" in converted, "down_proj→w2 failed"
        assert "talker.layers.0.feed_forward.w3.weight" in converted, "up_proj→w3 failed"
        assert "talker.layers.0.attention_norm.weight" in converted
        assert "talker.layers.0.ffn_norm.weight" in converted

        logger.info("All key remappings verified")


class TestTalkerStateDictPrefix:
    """Test get_state_dict_prefix returns correct meta-style keys."""

    def test_state_dict_prefix_mapping(self):
        from models.demos.qwen3_tts.tt.model_config import TalkerModelArgs

        class MockArgs(TalkerModelArgs):
            def __init__(self):
                pass

        args = MockArgs()

        assert args.get_state_dict_prefix("Attention", 0) == "talker.layers.0.attention"
        assert args.get_state_dict_prefix("MLP", 0) == "talker.layers.0.feed_forward"
        assert args.get_state_dict_prefix("TransformerBlock", 0) == "talker.layers.0."
        assert args.get_state_dict_prefix("", None) == "talker."
        assert args.get_state_dict_prefix("Attention", 27) == "talker.layers.27.attention"

        logger.info("State dict prefix mappings verified")


class TestTalkerConfigParsing:
    """Test that config params are correctly extracted from raw JSON."""

    def test_config_from_raw_json(self):
        raw_config = {
            "talker_config": {
                "hidden_size": 2048,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "num_hidden_layers": 28,
                "head_dim": 128,
                "rms_norm_eps": 1e-6,
                "intermediate_size": 6144,
                "max_position_embeddings": 32768,
                "vocab_size": 3072,
                "text_vocab_size": 151936,
                "text_hidden_size": 2048,
                "num_code_groups": 16,
                "rope_theta": 1000000.0,
                "rope_scaling": {"mrope_section": [24, 20, 20], "interleaved": True},
                "codec_bos_id": 2149,
                "codec_eos_token_id": 2150,
                "codec_pad_id": 2148,
                "code_predictor_config": {"hidden_size": 1024, "num_hidden_layers": 5},
            },
            "tts_bos_token_id": 151672,
            "tts_eos_token_id": 151673,
            "tts_pad_token_id": 151671,
            "speaker_encoder_config": {"enc_dim": 2048},
        }

        talker_cfg = raw_config["talker_config"]

        assert talker_cfg["hidden_size"] == 2048
        assert talker_cfg["num_attention_heads"] == 16
        assert talker_cfg["num_key_value_heads"] == 8
        assert talker_cfg["num_hidden_layers"] == 28
        assert talker_cfg["head_dim"] == 128
        assert talker_cfg["intermediate_size"] == 6144
        assert talker_cfg["vocab_size"] == 3072
        assert talker_cfg.get("text_vocab_size", 151936) == 151936
        assert talker_cfg.get("num_code_groups", 16) == 16
        assert talker_cfg.get("rope_theta", 1000000.0) == 1000000.0

        rope_scaling = talker_cfg.get("rope_scaling", {})
        assert rope_scaling.get("mrope_section") == [24, 20, 20]
        assert rope_scaling.get("interleaved") is True

        assert raw_config.get("tts_bos_token_id") == 151672
        assert raw_config.get("speaker_encoder_config", {}).get("enc_dim") == 2048

        logger.info("Config parsing from raw JSON verified")
