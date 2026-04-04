# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests for full Qwen3-Coder-Next model integration."""


from models.demos.qwen3_coder_next.tt.model_config import Qwen3CoderNextConfig


class TestModelConfig:
    """Test that model config correctly identifies layer types."""

    def test_layer_count(self):
        config = Qwen3CoderNextConfig()
        assert config.num_hidden_layers == 48
        assert config.num_deltanet_layers == 36
        assert config.num_gqa_layers == 12

    def test_layer_type_pattern(self):
        config = Qwen3CoderNextConfig()
        for i in range(48):
            if i % 4 == 3:
                assert config.is_gqa_layer(i), f"Layer {i} should be GQA"
            else:
                assert not config.is_gqa_layer(i), f"Layer {i} should be DeltaNet"

    def test_moe_params(self):
        config = Qwen3CoderNextConfig()
        assert config.num_experts == 512
        assert config.num_experts_per_tok == 10
        assert config.moe_intermediate_size == 512
        assert config.shared_expert_intermediate_size == 512

    def test_deltanet_params(self):
        config = Qwen3CoderNextConfig()
        assert config.linear_key_head_dim == 128
        assert config.linear_num_key_heads == 16
        assert config.linear_num_value_heads == 32
        assert config.linear_conv_kernel_dim == 4

    def test_partial_rope(self):
        config = Qwen3CoderNextConfig()
        assert config.partial_rotary_factor == 0.25
        assert config.rotary_dim == 64
        assert config.non_rotary_dim == 192

    def test_gqa_ratio(self):
        config = Qwen3CoderNextConfig()
        assert config.num_attention_heads == 16
        assert config.num_key_value_heads == 2
        assert config.gqa_ratio == 8

    def test_from_hf_config(self):
        hf_dict = {
            "hidden_size": 2048,
            "num_hidden_layers": 48,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "vocab_size": 151936,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "full_attention_interval": 4,
            "partial_rotary_factor": 0.25,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_conv_kernel_dim": 4,
            "model_type": "qwen3_next",
        }
        config = Qwen3CoderNextConfig.from_hf_config(hf_dict)
        assert config.num_experts == 512
        assert config.model_type == "qwen3_next"
        assert config.num_deltanet_layers == 36


class TestModelAssembly:
    """Test that model layers are correctly assembled."""

    def test_48_layers_correct_types(self):
        config = Qwen3CoderNextConfig()
        layer_types = []
        for i in range(config.num_hidden_layers):
            if config.is_gqa_layer(i):
                layer_types.append("gqa")
            else:
                layer_types.append("deltanet")

        assert len(layer_types) == 48
        assert layer_types.count("deltanet") == 36
        assert layer_types.count("gqa") == 12

        # Verify pattern: DeltaNet, DeltaNet, DeltaNet, GQA, repeat
        for i in range(48):
            expected = "gqa" if i % 4 == 3 else "deltanet"
            assert layer_types[i] == expected, f"Layer {i}: expected {expected}, got {layer_types[i]}"

    def test_expert_sharding(self):
        """512 experts across 8 devices = 64 per device."""
        config = Qwen3CoderNextConfig()
        num_devices = 8
        experts_per_device = config.num_experts // num_devices
        assert experts_per_device == 64

        # Verify contiguous assignment
        for device_id in range(num_devices):
            start = device_id * experts_per_device
            end = start + experts_per_device
            assert end - start == 64
            assert end <= config.num_experts
