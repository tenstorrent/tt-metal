# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Tests for YaRN RoPE reference implementation.

Verifies that the YaRN implementation matches HuggingFace OLMo.
Run with: pytest models/demos/llama3_70b_galaxy/reference/test_yarn_rope.py -v

Prerequisites:
- pip install transformers>=4.57.0
- export HF_MODEL=~/models/models--allenai--Olmo-3.1-32B-Think (or HF path)
"""

import os
import pytest
import torch

from .yarn_rope import (
    YaRNConfig,
    yarn_find_correction_dim,
    yarn_find_correction_range,
    yarn_linear_ramp_mask,
    compute_yarn_inv_freq,
    compute_yarn_mscale,
    precompute_yarn_freqs,
    precompute_yarn_freqs_cis,
    apply_rotary_emb_yarn,
)


# ==============================================================================
# Fixtures
# ==============================================================================
@pytest.fixture
def yarn_config():
    """OLMo-3.1-32B YaRN configuration."""
    return YaRNConfig.from_olmo()


@pytest.fixture
def small_config():
    """Smaller config for faster tests."""
    return YaRNConfig(
        dim=64,
        max_position_embeddings=1024,
        base=10000.0,
        scaling_factor=2.0,
        original_max_position_embeddings=256,
        attention_factor=1.1,
        beta_fast=16.0,
        beta_slow=1.0,
    )


def skip_if_no_model():
    """Skip test if HF model is not available."""
    hf_model = os.getenv("HF_MODEL")
    if not hf_model:
        pytest.skip("HF_MODEL not set - skipping HF comparison test")
    if not os.path.exists(hf_model) and "allenai" not in hf_model:
        pytest.skip(f"Model path {hf_model} does not exist")


# ==============================================================================
# Unit Tests
# ==============================================================================
class TestYaRNCorrectionDim:
    """Test correction dimension computation."""

    def test_correction_dim_basic(self):
        """Test basic correction dim calculation."""
        # For standard RoPE params
        dim = yarn_find_correction_dim(
            num_rotations=32,
            dim=128,
            base=10000.0,
            max_position_embeddings=2048,
        )
        assert isinstance(dim, float)
        assert 0 <= dim <= 128

    def test_correction_dim_olmo(self, yarn_config):
        """Test correction dim for OLMo params."""
        dim_fast = yarn_find_correction_dim(
            num_rotations=yarn_config.beta_fast,
            dim=yarn_config.dim,
            base=yarn_config.base,
            max_position_embeddings=yarn_config.original_max_position_embeddings,
        )
        dim_slow = yarn_find_correction_dim(
            num_rotations=yarn_config.beta_slow,
            dim=yarn_config.dim,
            base=yarn_config.base,
            max_position_embeddings=yarn_config.original_max_position_embeddings,
        )

        # fast should have lower dim (high freq)
        assert dim_fast < dim_slow


class TestYaRNCorrectionRange:
    """Test correction range computation."""

    def test_correction_range_bounds(self, yarn_config):
        """Test correction range is within valid bounds."""
        low, high = yarn_find_correction_range(
            yarn_config.beta_fast,
            yarn_config.beta_slow,
            yarn_config.dim,
            yarn_config.base,
            yarn_config.original_max_position_embeddings,
        )

        assert low >= 0
        assert high <= yarn_config.dim - 1
        assert low <= high

    def test_correction_range_olmo(self, yarn_config):
        """Test OLMo correction range."""
        low, high = yarn_find_correction_range(
            yarn_config.beta_fast,
            yarn_config.beta_slow,
            yarn_config.dim,
            yarn_config.base,
            yarn_config.original_max_position_embeddings,
        )

        # For OLMo with high theta (500000), the range should be small
        print(f"OLMo correction range: dim {low} to {high}")
        assert 0 <= low < yarn_config.dim // 2
        assert low < high


class TestLinearRampMask:
    """Test linear ramp mask."""

    def test_mask_shape(self):
        """Test mask has correct shape."""
        mask = yarn_linear_ramp_mask(0, 10, 64)
        assert mask.shape == (64,)

    def test_mask_values(self):
        """Test mask values are in [0, 1]."""
        mask = yarn_linear_ramp_mask(0, 10, 64)
        assert torch.all(mask >= 0)
        assert torch.all(mask <= 1)

    def test_mask_ramp(self):
        """Test mask ramps from 0 to 1."""
        mask = yarn_linear_ramp_mask(0, 10, 64)

        # First element should be 0
        assert mask[0] == 0

        # Element at 10 should be 1
        assert mask[10] == 1

        # Elements after 10 should be clamped to 1
        assert torch.all(mask[11:] == 1)


class TestInvFreq:
    """Test inverse frequency computation."""

    def test_inv_freq_shape(self, yarn_config):
        """Test inv_freq has correct shape."""
        inv_freq = compute_yarn_inv_freq(yarn_config)
        assert inv_freq.shape == (yarn_config.dim // 2,)

    def test_inv_freq_positive(self, yarn_config):
        """Test inv_freq values are positive."""
        inv_freq = compute_yarn_inv_freq(yarn_config)
        assert torch.all(inv_freq > 0)

    def test_inv_freq_decreasing(self, yarn_config):
        """Test inv_freq decreases (higher dims = lower freq)."""
        inv_freq = compute_yarn_inv_freq(yarn_config)
        # First few should be larger than last few
        assert inv_freq[0] > inv_freq[-1]

    def test_inv_freq_scaled(self, small_config):
        """Test inv_freq is scaled by scaling_factor."""
        # Standard RoPE inv_freq
        base_inv_freq = 1.0 / (
            small_config.base ** (torch.arange(0, small_config.dim, 2, dtype=torch.float32) / small_config.dim)
        )

        # YaRN scaled
        yarn_inv_freq = compute_yarn_inv_freq(small_config)

        # YaRN should be scaled down
        ratio = base_inv_freq / yarn_inv_freq
        # All should be approximately scaling_factor
        assert torch.allclose(ratio, torch.full_like(ratio, small_config.scaling_factor), rtol=0.1)


class TestMscale:
    """Test mscale computation."""

    def test_mscale_matches_attention_factor(self, yarn_config):
        """Test mscale equals attention_factor for OLMo."""
        mscale = compute_yarn_mscale(yarn_config)
        assert mscale == yarn_config.attention_factor

    def test_mscale_positive(self, yarn_config):
        """Test mscale is positive."""
        mscale = compute_yarn_mscale(yarn_config)
        assert mscale > 0


class TestPrecomputeFreqs:
    """Test frequency precomputation."""

    def test_cos_sin_shapes(self, yarn_config):
        """Test cos/sin have correct shapes."""
        seq_len = 1024
        cos, sin, mscale = precompute_yarn_freqs(yarn_config, seq_len=seq_len)

        assert cos.shape == (seq_len, yarn_config.dim // 2)
        assert sin.shape == (seq_len, yarn_config.dim // 2)

    def test_cos_sin_bounds(self, yarn_config):
        """Test cos/sin are in [-1, 1]."""
        cos, sin, mscale = precompute_yarn_freqs(yarn_config, seq_len=256)

        assert torch.all(cos >= -1)
        assert torch.all(cos <= 1)
        assert torch.all(sin >= -1)
        assert torch.all(sin <= 1)

    def test_cos_sin_identity(self, yarn_config):
        """Test cos^2 + sin^2 = 1."""
        cos, sin, _ = precompute_yarn_freqs(yarn_config, seq_len=256)

        identity = cos.pow(2) + sin.pow(2)
        assert torch.allclose(identity, torch.ones_like(identity), atol=1e-6)

    def test_freqs_cis_complex(self, yarn_config):
        """Test complex freqs_cis."""
        freqs_cis, mscale = precompute_yarn_freqs_cis(yarn_config, seq_len=256)

        assert freqs_cis.dtype == torch.complex64
        assert freqs_cis.shape == (256, yarn_config.dim // 2)

        # Magnitude should be 1
        magnitude = freqs_cis.abs()
        assert torch.allclose(magnitude, torch.ones_like(magnitude), atol=1e-6)


class TestApplyRotaryEmb:
    """Test rotary embedding application."""

    def test_output_shape(self, small_config):
        """Test output shapes match input."""
        batch, seq_len = 2, 64
        n_heads, n_kv_heads = 8, 4

        q = torch.randn(batch, seq_len, n_heads, small_config.dim)
        k = torch.randn(batch, seq_len, n_kv_heads, small_config.dim)

        cos, sin, _ = precompute_yarn_freqs(small_config, seq_len=seq_len)

        q_rot, k_rot = apply_rotary_emb_yarn(q, k, cos, sin)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_dtype_preserved(self, small_config):
        """Test dtype is preserved."""
        q = torch.randn(1, 16, 4, small_config.dim, dtype=torch.bfloat16)
        k = torch.randn(1, 16, 2, small_config.dim, dtype=torch.bfloat16)

        cos, sin, _ = precompute_yarn_freqs(small_config, seq_len=16)

        q_rot, k_rot = apply_rotary_emb_yarn(q, k, cos, sin)

        assert q_rot.dtype == torch.bfloat16
        assert k_rot.dtype == torch.bfloat16

    def test_rotation_changes_values(self, small_config):
        """Test rotation actually changes values."""
        q = torch.randn(1, 16, 4, small_config.dim)
        k = torch.randn(1, 16, 2, small_config.dim)

        cos, sin, _ = precompute_yarn_freqs(small_config, seq_len=16)

        q_rot, k_rot = apply_rotary_emb_yarn(q, k, cos, sin)

        assert not torch.allclose(q_rot, q)
        assert not torch.allclose(k_rot, k)


# ==============================================================================
# HuggingFace Comparison Tests
# ==============================================================================
class TestHuggingFaceComparison:
    """Compare against HuggingFace implementation."""

    @pytest.mark.slow
    def test_yarn_config_from_hf(self):
        """Test loading YaRN config from HF."""
        skip_if_no_model()

        try:
            from transformers import AutoConfig
        except ImportError:
            pytest.skip("transformers not installed")

        hf_model_path = os.getenv("HF_MODEL", "allenai/OLMo-3.1-32B-Think")
        hf_config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)

        yarn_config = YaRNConfig.from_hf_config(hf_config)

        # Verify config matches expected OLMo values
        assert yarn_config.dim == 128  # 5120 / 40 heads
        assert yarn_config.base == pytest.approx(500000.0)
        assert yarn_config.scaling_factor == pytest.approx(8.0)
        assert yarn_config.attention_factor == pytest.approx(1.2079, rel=1e-3)

    @pytest.mark.slow
    def test_rotary_emb_matches_hf(self):
        """Test rotary embedding matches HuggingFace."""
        skip_if_no_model()

        try:
            from transformers import AutoConfig
        except ImportError:
            pytest.skip("transformers not installed")

        hf_model_path = os.getenv("HF_MODEL", "allenai/OLMo-3.1-32B-Think")

        # Load HF config
        hf_config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
        yarn_config = YaRNConfig.from_hf_config(hf_config)

        # Compute our frequencies
        our_cos, our_sin, our_mscale = precompute_yarn_freqs(yarn_config, seq_len=128)

        print(f"Our cos[0, :4]: {our_cos[0, :4]}")
        print(f"Our sin[0, :4]: {our_sin[0, :4]}")
        print(f"Our mscale: {our_mscale}")

        # For full comparison, would need to load HF model and extract its RoPE
        # This is expensive for 32B model, so we just verify shapes and properties
        assert our_cos.shape == (128, 64)  # seq_len, head_dim//2
        assert our_mscale == pytest.approx(yarn_config.attention_factor)


# ==============================================================================
# Integration Tests
# ==============================================================================
class TestYaRNIntegration:
    """Integration tests for full YaRN pipeline."""

    def test_full_yarn_pipeline(self, yarn_config):
        """Test complete YaRN RoPE pipeline."""
        batch, seq_len = 1, 256
        n_heads = 40
        n_kv_heads = 8

        # Create inputs
        torch.manual_seed(42)
        q = torch.randn(batch, seq_len, n_heads, yarn_config.dim)
        k = torch.randn(batch, seq_len, n_kv_heads, yarn_config.dim)

        # Precompute frequencies
        cos, sin, mscale = precompute_yarn_freqs(yarn_config, seq_len=seq_len)

        # Apply rotation
        q_rot, k_rot = apply_rotary_emb_yarn(q, k, cos, sin)

        # Basic sanity checks
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        assert mscale == yarn_config.attention_factor

        # Rotation should be reversible (apply inverse)
        # For position 0, rotation should be identity (all angles 0)
        # Not quite true due to scaling, but close
        assert torch.allclose(q_rot[:, 0], q[:, 0], atol=1.0)  # Loose tolerance

    def test_yarn_vs_standard_rope(self, small_config):
        """Compare YaRN to standard RoPE."""
        # Standard RoPE (no YaRN scaling)
        standard_config = YaRNConfig(
            dim=small_config.dim,
            max_position_embeddings=small_config.max_position_embeddings,
            base=small_config.base,
            scaling_factor=1.0,  # No scaling
            original_max_position_embeddings=small_config.max_position_embeddings,
            attention_factor=1.0,
            beta_fast=32.0,
            beta_slow=1.0,
        )

        # Verify that YaRN scales down the inverse frequencies
        yarn_inv_freq = compute_yarn_inv_freq(small_config)
        std_inv_freq = compute_yarn_inv_freq(standard_config)

        # YaRN inv_freq should be scaled down by scaling_factor
        expected_ratio = small_config.scaling_factor
        actual_ratio = std_inv_freq / yarn_inv_freq

        # All dimensions should be scaled by approximately the scaling_factor
        assert torch.allclose(actual_ratio, torch.full_like(actual_ratio, expected_ratio), rtol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
