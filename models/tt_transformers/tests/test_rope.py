# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch

from models.tt_transformers.tt.common import gather_cos_sin, precompute_freqs, rope_scaling_model_factory
from models.tt_transformers.tt.rope import RotaryEmbedding, rotary_embedding_factory


class TestRope:
    """Test suite to compare different RoPE implementations for consistency."""

    def test_basic_rope_vs_precompute_freqs(self):
        """
        Test that compares sin/cos matrices computed by RotaryEmbedding class
        vs precompute_freqs function to check for discrepancies.
        """
        # Test parameters
        dim = 128
        max_seq_len = 1024
        base = 10000.0
        device = torch.device("cpu")

        # Create RotaryEmbedding instance
        rope = RotaryEmbedding(dim=dim, max_position_embeddings=max_seq_len, base=base, device=device)

        # Get cos/sin from RotaryEmbedding
        rope_cos, rope_sin = rope.cos_cached, rope.sin_cached

        # Get cos/sin from precompute_freqs
        precompute_cos, precompute_sin = precompute_freqs(
            dim=dim, end=2 * max_seq_len, theta=base, scale_factor=None, orig_context_len=None
        )
        precompute_cos, precompute_sin = gather_cos_sin(torch.arange(max_seq_len), precompute_cos, precompute_sin)

        print(f"RotaryEmbedding cos shape: {rope_cos.shape}")
        print(f"RotaryEmbedding sin shape: {rope_sin.shape}")
        print(f"precompute_freqs cos shape: {precompute_cos.shape}")
        print(f"precompute_freqs sin shape: {precompute_sin.shape}")

        # Compare shapes
        assert (
            rope_cos.shape == precompute_cos.shape
        ), f"Cos shapes don't match: {rope_cos.shape} vs {precompute_cos.shape}"
        assert (
            rope_sin.shape == precompute_sin.shape
        ), f"Sin shapes don't match: {rope_sin.shape} vs {precompute_sin.shape}"

        # Compare values with tolerance
        cos_diff = torch.abs(rope_cos - precompute_cos)
        sin_diff = torch.abs(rope_sin - precompute_sin)

        max_cos_diff = torch.max(cos_diff)
        max_sin_diff = torch.max(sin_diff)

        print(f"Max cos difference: {max_cos_diff}")
        print(f"Max sin difference: {max_sin_diff}")
        print(f"Mean cos difference: {torch.mean(cos_diff)}")
        print(f"Mean sin difference: {torch.mean(sin_diff)}")

        # Allow for small numerical differences
        tolerance = 1e-6
        assert max_cos_diff < tolerance, f"Cos values differ by more than {tolerance}: {max_cos_diff}"
        assert max_sin_diff < tolerance, f"Sin values differ by more than {tolerance}: {max_sin_diff}"

    def test_rope_llama3_scaling(self):
        """
        Test that the shape of the cos/sin matrices is correct for yarn scaling.
        """
        dim = 128
        max_seq_len = 1024
        base = 10000.0
        device = torch.device("cpu")

        rope = RotaryEmbedding(dim=dim, max_position_embeddings=max_seq_len, base=base, device=device)
        rope_cos, rope_sin = rope.cos_cached, rope.sin_cached

        rope_llama_model = rope_scaling_model_factory(
            {"rope_type": "llama3", "factor": 32, "original_max_position_embeddings": 8192}
        )
        rope_llama_scaled = rotary_embedding_factory(
            dim=dim, max_position_embeddings=max_seq_len, base=base, rope_scaling=rope_llama_model
        )
        rope_llama_scaled_cos, rope_llama_scaled_sin = rope_llama_scaled.cos_cached, rope_llama_scaled.sin_cached

        assert rope_llama_scaled_cos.shape == rope_cos.shape == (1, 1, max_seq_len, dim)
        assert rope_llama_scaled_sin.shape == rope_sin.shape == (1, 1, max_seq_len, dim)

        cos_diff = torch.abs(rope_cos - rope_llama_scaled_cos)
        sin_diff = torch.abs(rope_sin - rope_llama_scaled_sin)

        max_cos_diff = torch.max(cos_diff)
        max_sin_diff = torch.max(sin_diff)

        print(f"Max cos difference: {max_cos_diff}")
        print(f"Max sin difference: {max_sin_diff}")
        print(f"Mean cos difference: {torch.mean(cos_diff)}")
        print(f"Mean sin difference: {torch.mean(sin_diff)}")

        # Make sure we actually ran the scaling
        assert max_cos_diff > 1e-6, f"Cos values are the same as non scaled. Max diff = {max_cos_diff}"
        assert max_sin_diff > 1e-6, f"Sin values are the same as non scaled. Max diff = {max_sin_diff}"

    def test_rope_yarn_scaling(self):
        """
        Test that the shape of the cos/sin matrices is correct for yarn scaling.
        """
        dim = 128
        max_seq_len = 1024
        base = 10000.0
        device = torch.device("cpu")

        rope = RotaryEmbedding(dim=dim, max_position_embeddings=max_seq_len, base=base, device=device)
        rope_cos, rope_sin = rope.cos_cached, rope.sin_cached

        rope_yarn_model = rope_scaling_model_factory(
            {"rope_type": "yarn", "factor": 32, "original_max_position_embeddings": 8192}
        )
        rope_yarn_scaled = rotary_embedding_factory(
            dim=dim, max_position_embeddings=max_seq_len, base=base, rope_scaling=rope_yarn_model
        )
        rope_yarn_scaled_cos, rope_yarn_scaled_sin = rope_yarn_scaled.cos_cached, rope_yarn_scaled.sin_cached

        assert rope_yarn_scaled_cos.shape == rope_cos.shape == (1, 1, max_seq_len, dim)
        assert rope_yarn_scaled_sin.shape == rope_sin.shape == (1, 1, max_seq_len, dim)

        cos_diff = torch.abs(rope_cos - rope_yarn_scaled_cos)
        sin_diff = torch.abs(rope_sin - rope_yarn_scaled_sin)

        max_cos_diff = torch.max(cos_diff)
        max_sin_diff = torch.max(sin_diff)

        print(f"Max cos difference: {max_cos_diff}")
        print(f"Max sin difference: {max_sin_diff}")
        print(f"Mean cos difference: {torch.mean(cos_diff)}")
        print(f"Mean sin difference: {torch.mean(sin_diff)}")

        # Make sure we actually ran the scaling
        assert max_cos_diff > 1e-6, f"Cos values are the same as non scaled. Max diff = {max_cos_diff}"
        assert max_sin_diff > 1e-6, f"Sin values are the same as non scaled. Max diff = {max_sin_diff}"


if __name__ == "__main__":
    # Run a quick test if executed directly
    test_instance = TestRope()
    test_instance.test_basic_rope_vs_precompute_freqs()
    test_instance.test_rope_llama3_scaling_shape()
    test_instance.test_rope_yarn_scaling_shape()
    print("All tests passed!")
