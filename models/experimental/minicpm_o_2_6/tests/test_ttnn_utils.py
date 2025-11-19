# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Test TTNN utilities for MiniCPM-o-2_6.

Tests PCC computation, weight generation, and basic tensor operations.
"""

import torch
import sys
from pathlib import Path

# Add tt directory to path
tt_path = Path(__file__).parent.parent / "tt"
if str(tt_path) not in sys.path:
    sys.path.insert(0, str(tt_path))

from test_utils import compute_pcc, validate_pcc, compute_relative_error, compute_mean_absolute_error
from weight_generator import (
    generate_resampler_weights,
    generate_audio_projector_weights,
    generate_cross_attention_weights,
    generate_conditional_chattts_weights,
    generate_dvae_weights,
    generate_all_component_weights,
)


def test_pcc_computation():
    """Test PCC computation with known correlations."""
    # Perfect correlation
    tensor1 = torch.randn(10, 20)
    pcc = compute_pcc(tensor1, tensor1)
    assert abs(pcc - 1.0) < 1e-6, f"Perfect correlation should be 1.0, got {pcc}"

    # No correlation (orthogonal random tensors should be close to 0)
    tensor2 = torch.randn(10, 20)
    tensor3 = torch.randn(10, 20)
    pcc = compute_pcc(tensor2, tensor3)
    # Random tensors typically have low correlation
    assert -0.5 < pcc < 0.5, f"Random tensors should have low correlation, got {pcc}"

    # Near-perfect correlation with small noise
    tensor4 = torch.randn(100, 100)
    tensor5 = tensor4 + torch.randn(100, 100) * 0.01  # Add 1% noise
    pcc = compute_pcc(tensor4, tensor5)
    assert pcc > 0.99, f"Near-identical tensors should have PCC > 0.99, got {pcc}"

    print("✅ PCC computation tests passed")


def test_pcc_validation():
    """Test PCC validation with threshold."""
    # Should pass
    result = validate_pcc(0.95, threshold=0.90, component_name="Test Component", raise_on_fail=False)
    assert result is True

    # Should fail
    result = validate_pcc(0.85, threshold=0.90, component_name="Test Component", raise_on_fail=False)
    assert result is False

    # Should raise - test with try/except for non-pytest runs
    try:
        validate_pcc(0.85, threshold=0.90, component_name="Test Component", raise_on_fail=True)
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        if "Should have raised" in str(e):
            raise
        pass  # Expected assertion error

    print("✅ PCC validation tests passed")


def test_relative_error():
    """Test relative error computation."""
    tensor1 = torch.ones(10, 10)
    tensor2 = torch.ones(10, 10) * 1.01  # 1% difference

    rel_error = compute_relative_error(tensor1, tensor2)
    # Relative error should be approximately 0.01
    assert 0.009 < rel_error < 0.011, f"Expected ~0.01, got {rel_error}"

    print("✅ Relative error tests passed")


def test_mae_computation():
    """Test mean absolute error computation."""
    tensor1 = torch.zeros(10, 10)
    tensor2 = torch.ones(10, 10)

    mae = compute_mean_absolute_error(tensor1, tensor2)
    assert abs(mae - 1.0) < 1e-6, f"Expected MAE of 1.0, got {mae}"

    print("✅ MAE computation tests passed")


def test_resampler_weight_generation():
    """Test resampler weight generation."""
    weights = generate_resampler_weights(seed=42)

    # Check key tensors exist
    assert "query" in weights
    assert "attn.q_proj.weight" in weights
    assert "mlp.fc1.weight" in weights

    # Check shapes
    assert weights["query"].shape == (64, 3584)
    assert weights["attn.q_proj.weight"].shape == (3584, 3584)
    assert weights["mlp.fc1.weight"].shape == (14336, 3584)

    # Check reproducibility
    weights2 = generate_resampler_weights(seed=42)
    pcc = compute_pcc(weights["query"], weights2["query"])
    assert abs(pcc - 1.0) < 1e-6, f"Same seed should produce identical weights, got PCC={pcc}"

    print("✅ Resampler weight generation tests passed")


def test_audio_projector_weight_generation():
    """Test audio projector weight generation."""
    weights = generate_audio_projector_weights(seed=42)

    # Check key tensors exist (with audio_projection_layer prefix)
    assert "audio_projection_layer.linear1.weight" in weights
    assert "audio_projection_layer.linear1.bias" in weights
    assert "audio_projection_layer.linear2.weight" in weights
    assert "audio_projection_layer.linear2.bias" in weights

    # Check shapes (updated for correct input_dim=1024)
    assert weights["audio_projection_layer.linear1.weight"].shape == (3584, 1024)
    assert weights["audio_projection_layer.linear2.weight"].shape == (3584, 3584)

    print("✅ Audio projector weight generation tests passed")


def test_cross_attention_weight_generation():
    """Test cross-attention weight generation."""
    weights = generate_cross_attention_weights(seed=42)

    # Check key tensors exist
    assert "q_proj.weight" in weights
    assert "k_proj.weight" in weights
    assert "v_proj.weight" in weights
    assert "o_proj.weight" in weights
    assert "q_norm.weight" in weights
    assert "k_norm.weight" in weights

    # Check shapes (GQA: 28 query heads, 4 KV heads)
    hidden_size = 3584
    head_dim = hidden_size // 28
    assert weights["q_proj.weight"].shape == (28 * head_dim, hidden_size)
    assert weights["k_proj.weight"].shape == (4 * head_dim, hidden_size)
    # Both q_norm and k_norm are per-head (head_dim)
    assert weights["q_norm.weight"].shape == (head_dim,)
    assert weights["k_norm.weight"].shape == (head_dim,)

    print("✅ Cross-attention weight generation tests passed")


def test_conditional_chattts_weight_generation():
    """Test ConditionalChatTTS weight generation."""
    weights = generate_conditional_chattts_weights(seed=42)

    # Check key tensors exist
    assert "projector.linear1.weight" in weights
    assert "emb_text.weight" in weights
    assert "emb_code.0.weight" in weights  # First codebook
    assert "emb_code.3.weight" in weights  # Fourth codebook
    assert "model.layers.0.self_attn.q_proj.weight" in weights
    assert "model.layers.19.self_attn.q_proj.weight" in weights  # Layer 19 (last)
    assert "head_code.0.weight" in weights

    # Check shapes
    assert weights["emb_text.weight"].shape == (21178, 768)
    assert weights["emb_code.0.weight"].shape == (626, 768)
    assert weights["model.layers.0.self_attn.q_proj.weight"].shape == (768, 768)
    assert weights["head_code.0.weight"].shape == (626, 768)

    print("✅ ConditionalChatTTS weight generation tests passed")


def test_dvae_weight_generation():
    """Test DVAE weight generation."""
    weights = generate_dvae_weights(seed=42)

    # Check key tensors exist
    assert "coef" in weights
    assert "downsample_conv.0.weight" in weights
    assert "encoder.decoder_block.0.dwconv.weight" in weights
    assert "encoder.decoder_block.1.dwconv.weight" in weights  # Last encoder block (num_encoder_layers=2)
    assert "decoder.decoder_block.0.dwconv.weight" in weights
    assert "decoder.decoder_block.1.dwconv.weight" in weights  # Last decoder block (num_decoder_layers=2)

    # Check shapes - 2D conv format [out_channels, in_channels, kernel_h, kernel_w]
    assert weights["coef"].shape == (1, 100, 1)
    assert weights["encoder.decoder_block.0.dwconv.weight"].shape == (256, 1, 1, 7)  # Conv2D format

    print("✅ DVAE weight generation tests passed")


def test_all_component_weights_generation():
    """Test generation of all component weights."""
    all_weights = generate_all_component_weights(seed=42)

    # Check all components present
    expected_components = [
        "vision_resampler",
        "audio_resampler",
        "audio_projector",
        "cross_attention_layer8",
        "cross_attention_layer16",
        "cross_attention_layer24",
        "conditional_chattts",
        "dvae",
    ]

    for component in expected_components:
        assert component in all_weights, f"Missing component: {component}"
        assert len(all_weights[component]) > 0, f"Empty weights for: {component}"

    print("✅ All component weight generation tests passed")


if __name__ == "__main__":
    print("Testing TTNN utilities for MiniCPM-o-2_6")
    print("=" * 60)

    test_pcc_computation()
    test_pcc_validation()
    test_relative_error()
    test_mae_computation()
    test_resampler_weight_generation()
    test_audio_projector_weight_generation()
    test_cross_attention_weight_generation()
    test_conditional_chattts_weight_generation()
    test_dvae_weight_generation()
    test_all_component_weights_generation()

    print("=" * 60)
    print("✅ All utility tests passed!")
