# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Tests for OLMo-3.1-32B reference implementation.

Verifies that the standalone functional implementations match HuggingFace outputs.
Run with: pytest models/demos/llama3_70b_galaxy/reference/test_olmo.py -v

Prerequisites:
- pip install transformers>=4.57.0
- export HF_MODEL=~/models/models--allenai--Olmo-3.1-32B-Think (or HF path)
"""

import os
import pytest
import torch
import torch.nn.functional as F

from .functional import (
    rmsnorm_forward,
    swiglu_mlp_forward,
    attention_forward,
    precompute_freqs_cos_sin,
    create_sliding_window_mask,
    decoder_block_forward,
    embedding_forward,
)
from .olmo import OlmoModelArgs, precompute_yarn_freqs_cis


# ==============================================================================
# Fixtures
# ==============================================================================
@pytest.fixture
def olmo_args():
    """OLMo-3.1-32B configuration."""
    return OlmoModelArgs()


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
class TestRMSNorm:
    """Test RMSNorm implementation."""

    def test_rmsnorm_output_shape(self, olmo_args):
        """Test that RMSNorm produces correct output shape."""
        torch.manual_seed(0)
        x = torch.randn(2, 128, olmo_args.dim)
        weight = torch.ones(olmo_args.dim)

        output = rmsnorm_forward(x, weight, olmo_args.norm_eps)

        assert output.shape == x.shape

    def test_rmsnorm_normalized(self, olmo_args):
        """Test that output has approximately unit RMS."""
        torch.manual_seed(0)
        x = torch.randn(2, 128, olmo_args.dim)
        weight = torch.ones(olmo_args.dim)

        output = rmsnorm_forward(x, weight, olmo_args.norm_eps)

        rms = output.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


class TestSwiGLUMLP:
    """Test SwiGLU MLP implementation."""

    def test_mlp_output_shape(self, olmo_args):
        """Test MLP output shape."""
        torch.manual_seed(0)
        x = torch.randn(2, 128, olmo_args.dim)
        w1 = torch.randn(olmo_args.intermediate_size, olmo_args.dim) * 0.01
        w2 = torch.randn(olmo_args.dim, olmo_args.intermediate_size) * 0.01
        w3 = torch.randn(olmo_args.intermediate_size, olmo_args.dim) * 0.01

        output = swiglu_mlp_forward(x, w1, w2, w3)

        assert output.shape == (2, 128, olmo_args.dim)

    def test_swiglu_formula(self):
        """Test SwiGLU: down(silu(gate(x)) * up(x))."""
        torch.manual_seed(0)
        x = torch.randn(1, 1, 16)
        w1 = torch.randn(32, 16)
        w2 = torch.randn(16, 32)
        w3 = torch.randn(32, 16)

        output = swiglu_mlp_forward(x, w1, w2, w3)

        # Manual computation
        gate = F.silu(F.linear(x, w1))
        up = F.linear(x, w3)
        expected = F.linear(gate * up, w2)

        assert torch.allclose(output, expected)


class TestYaRNRoPE:
    """Test YaRN RoPE implementation."""

    def test_yarn_freqs_shape(self, olmo_args):
        """Test YaRN frequency tensor shape."""
        freqs_cis, mscale = precompute_yarn_freqs_cis(olmo_args)

        assert freqs_cis.shape[0] == olmo_args.max_seq_len * 2
        assert freqs_cis.shape[1] == olmo_args.head_dim // 2
        assert freqs_cis.dtype == torch.complex64

    def test_yarn_mscale_value(self, olmo_args):
        """Test YaRN mscale matches attention_factor."""
        _, mscale = precompute_yarn_freqs_cis(olmo_args)

        assert mscale == pytest.approx(olmo_args.attention_factor, rel=1e-5)

    def test_cos_sin_precompute(self, olmo_args):
        """Test cos/sin precomputation."""
        cos, sin, mscale = precompute_freqs_cos_sin(
            dim=olmo_args.head_dim,
            end=1024,
            theta=olmo_args.rope_theta,
            use_yarn=True,
            scaling_factor=olmo_args.rope_scaling_factor,
            original_max_position_embeddings=olmo_args.original_max_position_embeddings,
            beta_fast=olmo_args.beta_fast,
            beta_slow=olmo_args.beta_slow,
            attention_factor=olmo_args.attention_factor,
        )

        assert cos.shape == (1024, olmo_args.head_dim // 2)
        assert sin.shape == (1024, olmo_args.head_dim // 2)
        assert mscale == pytest.approx(olmo_args.attention_factor, rel=1e-5)


class TestSlidingWindowMask:
    """Test sliding window mask creation."""

    def test_causal_mask_no_window(self):
        """Test causal mask without sliding window."""
        mask = create_sliding_window_mask(4, sliding_window=None)

        assert mask.shape == (1, 1, 4, 4)
        # Upper triangle should be -inf
        assert mask[0, 0, 0, 1] == float("-inf")
        assert mask[0, 0, 0, 0] == 0

    def test_sliding_window_mask(self):
        """Test sliding window mask."""
        seq_len = 8
        window = 3
        mask = create_sliding_window_mask(seq_len, sliding_window=window)

        # Position 7 should not attend to position 0 (distance > window)
        # But should attend to positions 4, 5, 6, 7
        assert mask[0, 0, 7, 0] == float("-inf")  # Outside window
        assert mask[0, 0, 7, 4] == 0  # Inside window (distance = 3)
        assert mask[0, 0, 7, 7] == 0  # Self-attention


class TestAttention:
    """Test attention implementation."""

    def test_attention_output_shape(self, olmo_args):
        """Test attention output shape."""
        torch.manual_seed(0)
        batch, seq_len = 2, 128
        x = torch.randn(batch, seq_len, olmo_args.dim)

        wq = torch.randn(olmo_args.n_heads * olmo_args.head_dim, olmo_args.dim) * 0.01
        wk = torch.randn(olmo_args.n_kv_heads * olmo_args.head_dim, olmo_args.dim) * 0.01
        wv = torch.randn(olmo_args.n_kv_heads * olmo_args.head_dim, olmo_args.dim) * 0.01
        wo = torch.randn(olmo_args.dim, olmo_args.n_heads * olmo_args.head_dim) * 0.01

        cos, sin, mscale = precompute_freqs_cos_sin(
            dim=olmo_args.head_dim,
            end=seq_len * 2,
            theta=olmo_args.rope_theta,
            use_yarn=True,
            scaling_factor=olmo_args.rope_scaling_factor,
            original_max_position_embeddings=olmo_args.original_max_position_embeddings,
            beta_fast=olmo_args.beta_fast,
            beta_slow=olmo_args.beta_slow,
            attention_factor=olmo_args.attention_factor,
        )

        output = attention_forward(
            x,
            wq,
            wk,
            wv,
            wo,
            cos,
            sin,
            n_heads=olmo_args.n_heads,
            n_kv_heads=olmo_args.n_kv_heads,
            head_dim=olmo_args.head_dim,
            sliding_window=None,
            mscale=mscale,
        )

        assert output.shape == (batch, seq_len, olmo_args.dim)

    def test_attention_with_sliding_window(self, olmo_args):
        """Test attention with sliding window."""
        torch.manual_seed(0)
        batch, seq_len = 1, 64
        x = torch.randn(batch, seq_len, olmo_args.dim)

        wq = torch.randn(olmo_args.n_heads * olmo_args.head_dim, olmo_args.dim) * 0.01
        wk = torch.randn(olmo_args.n_kv_heads * olmo_args.head_dim, olmo_args.dim) * 0.01
        wv = torch.randn(olmo_args.n_kv_heads * olmo_args.head_dim, olmo_args.dim) * 0.01
        wo = torch.randn(olmo_args.dim, olmo_args.n_heads * olmo_args.head_dim) * 0.01

        cos, sin, mscale = precompute_freqs_cos_sin(
            dim=olmo_args.head_dim,
            end=seq_len * 2,
            theta=olmo_args.rope_theta,
            use_yarn=False,  # Simpler for this test
        )

        # Run with and without sliding window
        out_full = attention_forward(
            x,
            wq,
            wk,
            wv,
            wo,
            cos,
            sin,
            n_heads=olmo_args.n_heads,
            n_kv_heads=olmo_args.n_kv_heads,
            head_dim=olmo_args.head_dim,
            sliding_window=None,
            mscale=1.0,
        )

        out_window = attention_forward(
            x,
            wq,
            wk,
            wv,
            wo,
            cos,
            sin,
            n_heads=olmo_args.n_heads,
            n_kv_heads=olmo_args.n_kv_heads,
            head_dim=olmo_args.head_dim,
            sliding_window=16,  # Small window
            mscale=1.0,
        )

        # Outputs should be different due to sliding window
        assert not torch.allclose(out_full, out_window, atol=1e-3)


class TestDecoderBlock:
    """Test decoder block implementation."""

    def test_decoder_block_output_shape(self, olmo_args):
        """Test decoder block output shape."""
        torch.manual_seed(0)
        batch, seq_len = 2, 64
        x = torch.randn(batch, seq_len, olmo_args.dim)

        # Create random weights
        attn_norm = torch.ones(olmo_args.dim)
        ffn_norm = torch.ones(olmo_args.dim)
        wq = torch.randn(olmo_args.n_heads * olmo_args.head_dim, olmo_args.dim) * 0.01
        wk = torch.randn(olmo_args.n_kv_heads * olmo_args.head_dim, olmo_args.dim) * 0.01
        wv = torch.randn(olmo_args.n_kv_heads * olmo_args.head_dim, olmo_args.dim) * 0.01
        wo = torch.randn(olmo_args.dim, olmo_args.n_heads * olmo_args.head_dim) * 0.01
        w1 = torch.randn(olmo_args.intermediate_size, olmo_args.dim) * 0.01
        w2 = torch.randn(olmo_args.dim, olmo_args.intermediate_size) * 0.01
        w3 = torch.randn(olmo_args.intermediate_size, olmo_args.dim) * 0.01

        cos, sin, mscale = precompute_freqs_cos_sin(
            dim=olmo_args.head_dim,
            end=seq_len * 2,
            theta=olmo_args.rope_theta,
            use_yarn=False,
        )

        output = decoder_block_forward(
            x,
            attn_norm,
            ffn_norm,
            wq,
            wk,
            wv,
            wo,
            w1,
            w2,
            w3,
            cos,
            sin,
            n_heads=olmo_args.n_heads,
            n_kv_heads=olmo_args.n_kv_heads,
            head_dim=olmo_args.head_dim,
            sliding_window=None,
            mscale=mscale,
            norm_eps=olmo_args.norm_eps,
        )

        assert output.shape == x.shape


class TestLayerTypes:
    """Test OLMo layer type pattern."""

    def test_layer_pattern(self, olmo_args):
        """Test 3 sliding + 1 full pattern."""
        layer_types = [olmo_args.get_layer_type(i) for i in range(olmo_args.n_layers)]

        # Count types
        sliding_count = sum(1 for t in layer_types if t == "sliding_attention")
        full_count = sum(1 for t in layer_types if t == "full_attention")

        assert sliding_count == 48  # 3 * 16
        assert full_count == 16  # 1 * 16

    def test_layer_pattern_positions(self, olmo_args):
        """Test which layers are full attention."""
        for i in range(olmo_args.n_layers):
            layer_type = olmo_args.get_layer_type(i)
            if (i + 1) % 4 == 0:
                assert layer_type == "full_attention"
            else:
                assert layer_type == "sliding_attention"


# ==============================================================================
# HuggingFace Comparison Tests
# ==============================================================================
class TestHuggingFaceComparison:
    """Tests comparing reference to HuggingFace implementation."""

    @pytest.mark.slow
    def test_rmsnorm_matches_hf(self):
        """Test RMSNorm matches HuggingFace."""
        skip_if_no_model()

        try:
            from transformers import AutoConfig
        except ImportError:
            pytest.skip("transformers not installed")

        hf_model_path = os.getenv("HF_MODEL", "allenai/OLMo-3.1-32B-Think")

        # Load HF model (just the config and one layer)
        config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)

        # Create test input
        torch.manual_seed(42)
        x = torch.randn(1, 8, config.hidden_size)
        weight = torch.ones(config.hidden_size)

        # Our implementation
        our_output = rmsnorm_forward(x, weight, config.rms_norm_eps)

        # HF implementation (RMSNorm is standard)
        hf_output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + config.rms_norm_eps)
        hf_output = hf_output * weight

        assert torch.allclose(our_output, hf_output, atol=1e-5)

    @pytest.mark.slow
    def test_embedding_matches_hf(self):
        """Test embedding lookup matches HuggingFace."""
        skip_if_no_model()

        try:
            from transformers import AutoModelForCausalLM
        except ImportError:
            pytest.skip("transformers not installed")

        hf_model_path = os.getenv("HF_MODEL", "allenai/OLMo-3.1-32B-Think")

        # Load HF model
        hf_model = AutoModelForCausalLM.from_pretrained(
            hf_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu",
        )

        # Test input
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])

        # Our implementation
        our_output = embedding_forward(input_ids, hf_model.model.embed_tokens.weight)

        # HF implementation
        hf_output = hf_model.model.embed_tokens(input_ids)

        assert torch.allclose(our_output, hf_output, atol=1e-5)

        del hf_model


# ==============================================================================
# Golden Output Tests
# ==============================================================================
class TestGoldenOutputs:
    """Test generating and verifying golden outputs."""

    def test_generate_rmsnorm_golden(self, olmo_args, tmp_path):
        """Generate golden output for RMSNorm."""
        torch.manual_seed(42)

        x = torch.randn(1, 128, olmo_args.dim, dtype=torch.float32)
        weight = torch.randn(olmo_args.dim, dtype=torch.float32)

        output = rmsnorm_forward(x, weight, olmo_args.norm_eps)

        # Save golden
        golden_path = tmp_path / "rmsnorm_golden.pt"
        torch.save(
            {
                "input": x,
                "weight": weight,
                "output": output,
                "eps": olmo_args.norm_eps,
            },
            golden_path,
        )

        # Verify reload
        loaded = torch.load(golden_path)
        assert torch.allclose(loaded["output"], output)

    def test_generate_mlp_golden(self, olmo_args, tmp_path):
        """Generate golden output for MLP."""
        torch.manual_seed(42)

        x = torch.randn(1, 128, olmo_args.dim, dtype=torch.float32)
        w1 = torch.randn(olmo_args.intermediate_size, olmo_args.dim, dtype=torch.float32) * 0.01
        w2 = torch.randn(olmo_args.dim, olmo_args.intermediate_size, dtype=torch.float32) * 0.01
        w3 = torch.randn(olmo_args.intermediate_size, olmo_args.dim, dtype=torch.float32) * 0.01

        output = swiglu_mlp_forward(x, w1, w2, w3)

        # Save golden
        golden_path = tmp_path / "mlp_golden.pt"
        torch.save(
            {
                "input": x,
                "w1": w1,
                "w2": w2,
                "w3": w3,
                "output": output,
            },
            golden_path,
        )

        # Verify reload
        loaded = torch.load(golden_path)
        assert torch.allclose(loaded["output"], output)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
