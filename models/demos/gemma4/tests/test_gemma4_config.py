# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test Gemma 4 E4B model configuration and weight loading.

This test validates:
1. ModelArgs correctly parses HF config
2. State dict keys are properly converted
3. Weight shapes match expected dimensions
4. Per-layer head_dim is correct
5. KV sharing map is correct
"""

import os

import pytest
import torch

# Skip if weights not available
GEMMA4_WEIGHTS = os.environ.get(
    "GEMMA4_WEIGHTS",
    os.path.expanduser(
        "~/gemma4_weights/models--google--gemma-4-E4B-it/snapshots/292a7e278a400932df35f9fd4b1501edd04133a5"
    ),
)


def test_gemma4_config_parsing():
    """Test that Gemma 4 HF config is correctly parsed."""
    os.environ["HF_MODEL"] = GEMMA4_WEIGHTS
    os.environ["USER"] = os.environ.get("USER", "node")

    from models.demos.gemma4.tt.model_config import ModelArgs

    # Create ModelArgs with no device (CPU-only config test)
    args = ModelArgs(mesh_device=None, instruct=True, dummy_weights=False, max_batch_size=1)

    # Basic dimensions
    assert args.dim == 2560, f"Expected dim=2560, got {args.dim}"
    assert args.n_layers == 42, f"Expected n_layers=42, got {args.n_layers}"
    assert args.n_heads == 8, f"Expected n_heads=8, got {args.n_heads}"
    assert args.n_kv_heads == 2, f"Expected n_kv_heads=2, got {args.n_kv_heads}"
    assert args.head_dim == 256, f"Expected head_dim=256, got {args.head_dim}"
    assert args.hidden_dim == 10240, f"Expected hidden_dim=10240, got {args.hidden_dim}"
    assert args.vocab_size == 262144, f"Expected vocab_size=262144, got {args.vocab_size}"

    # Gemma 4 specific
    assert args.global_head_dim == 512, f"Expected global_head_dim=512, got {args.global_head_dim}"
    assert args.final_logit_softcapping == 30.0, f"Expected softcapping=30.0, got {args.final_logit_softcapping}"
    assert args.num_kv_shared_layers == 18, f"Expected num_kv_shared_layers=18, got {args.num_kv_shared_layers}"
    assert args.first_kv_shared_layer_idx == 24, f"Expected first_kv_shared=24, got {args.first_kv_shared_layer_idx}"
    assert (
        args.hidden_size_per_layer_input == 256
    ), f"Expected per_layer_input=256, got {args.hidden_size_per_layer_input}"
    assert args.partial_rotary_factor == 0.25, f"Expected partial_rotary=0.25, got {args.partial_rotary_factor}"

    # Gemma-family params
    assert args.rms_norm_add_unit_offset == True
    assert args.embed_scale is not None

    # Layer types
    assert args.layer_types is not None
    assert len(args.layer_types) == 42
    assert args.layer_types[0] == "sliding_attention"
    assert args.layer_types[5] == "full_attention"
    assert args.layer_types[41] == "full_attention"

    # Per-layer head dims
    assert len(args.layer_head_dims) == 42
    assert args.layer_head_dims[0] == 256  # sliding
    assert args.layer_head_dims[5] == 512  # global
    assert args.layer_head_dims[41] == 512  # global

    # Sliding window
    assert args.sliding_window == 512

    # RoPE
    assert args.rope_theta_local == 10000.0  # sliding
    assert args.rope_theta == 1000000.0  # global

    # KV sharing map
    assert len(args.kv_sharing_map) > 0
    # Layer 24 should map to a non-shared layer of same type
    for shared_idx, source_idx in args.kv_sharing_map.items():
        assert shared_idx >= 24, f"Shared layer {shared_idx} should be >= 24"
        assert source_idx < 24, f"Source layer {source_idx} should be < 24"
        assert (
            args.layer_types[shared_idx] == args.layer_types[source_idx]
        ), f"Type mismatch: layer {shared_idx}={args.layer_types[shared_idx]}, source {source_idx}={args.layer_types[source_idx]}"

    print("\n✓ All config parsing tests passed!")
    print(f"  Model: {args.model_name}")
    print(
        f"  Layers: {args.n_layers} ({sum(1 for lt in args.layer_types if lt == 'sliding_attention')} sliding, {sum(1 for lt in args.layer_types if lt == 'full_attention')} global)"
    )
    print(f"  KV sharing: {len(args.kv_sharing_map)} shared layers")


@pytest.mark.skipif(
    not os.path.exists(GEMMA4_WEIGHTS),
    reason="Gemma 4 weights not found. Set GEMMA4_WEIGHTS env var.",
)
def test_gemma4_weight_loading():
    """Test that Gemma 4 weights are loaded and converted correctly."""
    os.environ["HF_MODEL"] = GEMMA4_WEIGHTS
    os.environ["USER"] = os.environ.get("USER", "node")

    from models.demos.gemma4.tt.model_config import ModelArgs

    args = ModelArgs(mesh_device=None, instruct=True, dummy_weights=False, max_batch_size=1)
    state_dict = args.load_state_dict()

    print(f"\nLoaded {len(state_dict)} keys")

    # Check essential keys exist
    essential_keys = [
        "tok_embeddings.weight",
        "output.weight",
        "norm.weight",
    ]
    for key in essential_keys:
        assert key in state_dict, f"Missing key: {key}"

    # Check layer 0 (sliding) keys
    sliding_layer_keys = [
        "layers.0.attention.wq.weight",
        "layers.0.attention.wk.weight",
        "layers.0.attention.wv.weight",
        "layers.0.attention.wo.weight",
        "layers.0.attention.q_norm.weight",
        "layers.0.attention.k_norm.weight",
        "layers.0.feed_forward.w1.weight",
        "layers.0.feed_forward.w2.weight",
        "layers.0.feed_forward.w3.weight",
        "layers.0.attention_norm.weight",
        "layers.0.ffn_norm.weight",
    ]
    for key in sliding_layer_keys:
        assert key in state_dict, f"Missing sliding layer key: {key}"

    # Check layer 5 (global) keys
    global_layer_keys = [
        "layers.5.attention.wq.weight",
        "layers.5.attention.wk.weight",
        "layers.5.attention.wv.weight",
        "layers.5.attention.wo.weight",
    ]
    for key in global_layer_keys:
        assert key in state_dict, f"Missing global layer key: {key}"

    # Validate sliding layer weight shapes (layer 0, head_dim=256)
    wq_0 = state_dict["layers.0.attention.wq.weight"]
    wk_0 = state_dict["layers.0.attention.wk.weight"]
    wv_0 = state_dict["layers.0.attention.wv.weight"]
    wo_0 = state_dict["layers.0.attention.wo.weight"]
    print(f"\nSliding layer 0 shapes:")
    print(f"  wq: {wq_0.shape}")  # Expected: [8*256, 2560] = [2048, 2560]
    print(f"  wk: {wk_0.shape}")  # Expected: [2*256, 2560] = [512, 2560]
    print(f"  wv: {wv_0.shape}")  # Expected: [2*256, 2560] = [512, 2560]
    print(f"  wo: {wo_0.shape}")  # Expected: [2560, 8*256] = [2560, 2048]
    assert wq_0.shape == torch.Size([2048, 2560])
    assert wk_0.shape == torch.Size([512, 2560])
    assert wv_0.shape == torch.Size([512, 2560])
    assert wo_0.shape == torch.Size([2560, 2048])

    # Validate global layer weight shapes (layer 5, head_dim=512)
    wq_5 = state_dict["layers.5.attention.wq.weight"]
    wk_5 = state_dict["layers.5.attention.wk.weight"]
    wv_5 = state_dict["layers.5.attention.wv.weight"]
    wo_5 = state_dict["layers.5.attention.wo.weight"]
    print(f"\nGlobal layer 5 shapes:")
    print(f"  wq: {wq_5.shape}")  # Expected: [8*512, 2560] = [4096, 2560]
    print(f"  wk: {wk_5.shape}")  # Expected: [2*512, 2560] = [1024, 2560]
    print(f"  wv: {wv_5.shape}")  # Expected: [2*512, 2560] = [1024, 2560]
    print(f"  wo: {wo_5.shape}")  # Expected: [2560, 8*512] = [2560, 4096]
    assert wq_5.shape == torch.Size([4096, 2560])
    assert wk_5.shape == torch.Size([1024, 2560])
    assert wv_5.shape == torch.Size([1024, 2560])
    assert wo_5.shape == torch.Size([2560, 4096])

    # Check Gemma 4 specific keys
    gemma4_keys_found = [k for k in state_dict if "per_layer_input_gate" in k or "layer_scalar" in k]
    print(f"\nGemma 4 specific keys: {len(gemma4_keys_found)}")
    for k in gemma4_keys_found[:5]:
        print(f"  {k}: {state_dict[k].shape}")

    # Check pre/post feedforward norms
    pre_ff_keys = [k for k in state_dict if "pre_feedforward_layernorm" in k]
    post_ff_keys = [k for k in state_dict if "post_feedforward_layernorm" in k]
    print(f"\nPre-feedforward norm keys: {len(pre_ff_keys)}")
    print(f"Post-feedforward norm keys: {len(post_ff_keys)}")

    # Check embed_tokens_per_layer
    per_layer_embed_keys = [k for k in state_dict if "embed_tokens_per_layer" in k]
    print(f"\nPer-layer embedding keys: {len(per_layer_embed_keys)}")
    for k in per_layer_embed_keys:
        print(f"  {k}: {state_dict[k].shape}")

    print("\n✓ All weight loading tests passed!")


if __name__ == "__main__":
    test_gemma4_config_parsing()
    if os.path.exists(GEMMA4_WEIGHTS):
        test_gemma4_weight_loading()
    else:
        print(f"\nSkipping weight loading test (weights not found at {GEMMA4_WEIGHTS})")
