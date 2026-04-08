# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Quick test to verify Gemma 4 weight loading and config parsing work correctly.
Run without TT device - CPU only.
"""

import os
import sys

# Add tt-metal to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def test_config_parsing():
    """Test that we can parse the Gemma 4 config correctly."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained("google/gemma-4-31B-it").to_dict()
    text_config = config.get("text_config", config)

    # Verify key architecture params
    assert text_config["hidden_size"] == 5376, f"Expected hidden_size=5376, got {text_config['hidden_size']}"
    assert text_config["num_hidden_layers"] == 60, f"Expected 60 layers, got {text_config['num_hidden_layers']}"
    assert text_config["num_attention_heads"] == 32, f"Expected 32 heads, got {text_config['num_attention_heads']}"
    assert text_config["num_key_value_heads"] == 16, f"Expected 16 kv_heads, got {text_config['num_key_value_heads']}"
    assert text_config["head_dim"] == 256, f"Expected head_dim=256, got {text_config['head_dim']}"
    assert text_config["global_head_dim"] == 512, f"Expected global_head_dim=512, got {text_config['global_head_dim']}"
    assert text_config["num_global_key_value_heads"] == 4
    assert text_config["attention_k_eq_v"] == True
    assert text_config["intermediate_size"] == 21504
    assert text_config["sliding_window"] == 1024

    # Verify layer types
    layer_types = text_config["layer_types"]
    assert len(layer_types) == 60
    sliding_count = sum(1 for lt in layer_types if lt == "sliding_attention")
    full_count = sum(1 for lt in layer_types if lt == "full_attention")
    assert sliding_count == 50, f"Expected 50 sliding layers, got {sliding_count}"
    assert full_count == 10, f"Expected 10 full layers, got {full_count}"

    # Verify full attention layers are at expected positions (every 6th, starting from 5)
    full_indices = [i for i, lt in enumerate(layer_types) if lt == "full_attention"]
    expected_full = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59]
    assert full_indices == expected_full, f"Expected full layers at {expected_full}, got {full_indices}"

    # Verify dual RoPE
    rope_params = text_config["rope_parameters"]
    assert rope_params["sliding_attention"]["rope_theta"] == 10000.0
    assert rope_params["full_attention"]["rope_theta"] == 1000000.0
    assert rope_params["full_attention"]["partial_rotary_factor"] == 0.25

    print("PASSED: Config parsing")


def test_weight_shapes():
    """Test that weight shapes match expected architecture."""
    import safetensors.torch

    snapshot = "/home/ttuser/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/419b2efe421994fdfd3394e621983d4cc511cd4f"

    # Load a small subset to check shapes
    with safetensors.torch.safe_open(os.path.join(snapshot, "model-00001-of-00002.safetensors"), framework="pt") as f:
        # Layer 0 (sliding attention)
        q0 = f.get_tensor("model.language_model.layers.0.self_attn.q_proj.weight")
        k0 = f.get_tensor("model.language_model.layers.0.self_attn.k_proj.weight")
        v0 = f.get_tensor("model.language_model.layers.0.self_attn.v_proj.weight")
        o0 = f.get_tensor("model.language_model.layers.0.self_attn.o_proj.weight")
        scalar0 = f.get_tensor("model.language_model.layers.0.layer_scalar")

        assert q0.shape == (8192, 5376), f"Layer 0 Q shape: {q0.shape}"  # 32*256
        assert k0.shape == (4096, 5376), f"Layer 0 K shape: {k0.shape}"  # 16*256
        assert v0.shape == (4096, 5376), f"Layer 0 V shape: {v0.shape}"  # 16*256
        assert o0.shape == (5376, 8192), f"Layer 0 O shape: {o0.shape}"
        assert scalar0.shape == (1,), f"Layer scalar shape: {scalar0.shape}"

    with safetensors.torch.safe_open(os.path.join(snapshot, "model-00002-of-00002.safetensors"), framework="pt") as f:
        # Layer 5 (full attention, K=V)
        q5 = f.get_tensor("model.language_model.layers.5.self_attn.q_proj.weight")
        k5 = f.get_tensor("model.language_model.layers.5.self_attn.k_proj.weight")
        o5 = f.get_tensor("model.language_model.layers.5.self_attn.o_proj.weight")

        assert q5.shape == (16384, 5376), f"Layer 5 Q shape: {q5.shape}"  # 32*512
        assert k5.shape == (2048, 5376), f"Layer 5 K shape: {k5.shape}"  # 4*512
        assert o5.shape == (5376, 16384), f"Layer 5 O shape: {o5.shape}"

        # Verify layer 5 does NOT have v_proj
        keys = list(f.keys())
        layer5_keys = [k for k in keys if "layers.5." in k]
        has_v_proj = any("v_proj" in k for k in layer5_keys)
        assert not has_v_proj, f"Layer 5 should NOT have v_proj, but found it in keys"

    print("PASSED: Weight shapes")


def test_key_mapping():
    """Test the HF to meta key conversion for Gemma 4."""
    import torch

    from models.tt_transformers.tt.load_checkpoints import (
        convert_hf_to_meta_no_qkv_permute,
        standardize_hf_keys_multimodal,
    )

    # Create a minimal fake state dict to test key mapping
    fake_sd = {
        "model.language_model.embed_tokens.weight": torch.randn(10, 5376),
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(8192, 5376),
        "model.language_model.layers.0.self_attn.k_proj.weight": torch.randn(4096, 5376),
        "model.language_model.layers.0.self_attn.v_proj.weight": torch.randn(4096, 5376),
        "model.language_model.layers.0.self_attn.o_proj.weight": torch.randn(5376, 8192),
        "model.language_model.layers.0.self_attn.q_norm.weight": torch.randn(256),
        "model.language_model.layers.0.self_attn.k_norm.weight": torch.randn(256),
        "model.language_model.layers.0.input_layernorm.weight": torch.randn(5376),
        "model.language_model.layers.0.post_attention_layernorm.weight": torch.randn(5376),
        "model.language_model.layers.0.pre_feedforward_layernorm.weight": torch.randn(5376),
        "model.language_model.layers.0.post_feedforward_layernorm.weight": torch.randn(5376),
        "model.language_model.layers.0.mlp.gate_proj.weight": torch.randn(21504, 5376),
        "model.language_model.layers.0.mlp.up_proj.weight": torch.randn(21504, 5376),
        "model.language_model.layers.0.mlp.down_proj.weight": torch.randn(5376, 21504),
        "model.language_model.layers.0.layer_scalar": torch.tensor([1.0]),
        "model.language_model.norm.weight": torch.randn(5376),
    }

    sd = standardize_hf_keys_multimodal(fake_sd)
    sd = convert_hf_to_meta_no_qkv_permute(sd, head_dim=256)

    # Verify key mappings
    expected_keys = [
        "output.weight",  # embed_tokens → lm_head → output (tied weights)
        "layers.0.attention.wq.weight",
        "layers.0.attention.wk.weight",
        "layers.0.attention.wv.weight",
        "layers.0.attention.wo.weight",
        "layers.0.attention.q_norm.weight",
        "layers.0.attention.k_norm.weight",
        "layers.0.attention_norm.weight",
        "layers.0.ffn_norm.weight",
        "layers.0.pre_feedforward_layernorm.weight",
        "layers.0.post_feedforward_layernorm.weight",
        "layers.0.feed_forward.w1.weight",
        "layers.0.feed_forward.w3.weight",
        "layers.0.feed_forward.w2.weight",
        "layers.0.layer_scalar",
        "norm.weight",
    ]

    for ek in expected_keys:
        assert ek in sd, f"Missing expected key: {ek}. Available: {sorted(sd.keys())}"

    print("PASSED: Key mapping")


if __name__ == "__main__":
    test_config_parsing()
    test_weight_shapes()
    test_key_mapping()
    print("\nAll tests passed!")
