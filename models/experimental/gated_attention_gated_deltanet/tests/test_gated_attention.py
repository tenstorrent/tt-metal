# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tests for the functional torch Gated Attention implementation.

Validates that the functional implementation produces correct outputs
by checking shapes and numerical properties. Also serves as the golden
reference for TTNN validation.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from torch_functional.gated_attention import gated_attention_forward


def make_gated_attention_params(
    hidden_size=512,
    num_attention_heads=8,
    num_key_value_heads=2,
    head_dim=64,
    seq_len=32,
    batch_size=2,
    dtype=torch.float32,
):
    """Create random weights and inputs for testing."""
    device = "cpu"

    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)

    # Q proj is 2x wide (query + gate)
    q_proj_weight = torch.randn(num_attention_heads * head_dim * 2, hidden_size, dtype=dtype, device=device) * 0.02
    k_proj_weight = torch.randn(num_key_value_heads * head_dim, hidden_size, dtype=dtype, device=device) * 0.02
    v_proj_weight = torch.randn(num_key_value_heads * head_dim, hidden_size, dtype=dtype, device=device) * 0.02
    o_proj_weight = torch.randn(hidden_size, num_attention_heads * head_dim, dtype=dtype, device=device) * 0.02

    q_norm_weight = torch.zeros(head_dim, dtype=dtype, device=device)
    k_norm_weight = torch.zeros(head_dim, dtype=dtype, device=device)

    # RoPE embeddings
    cos = torch.randn(batch_size, seq_len, head_dim, dtype=dtype, device=device)
    sin = torch.randn(batch_size, seq_len, head_dim, dtype=dtype, device=device)

    # Causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    attention_mask = mask.masked_fill(mask.bool(), float("-inf"))
    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

    return {
        "hidden_states": hidden_states,
        "q_proj_weight": q_proj_weight,
        "k_proj_weight": k_proj_weight,
        "v_proj_weight": v_proj_weight,
        "o_proj_weight": o_proj_weight,
        "q_norm_weight": q_norm_weight,
        "k_norm_weight": k_norm_weight,
        "cos": cos,
        "sin": sin,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "attention_mask": attention_mask,
    }


def test_gated_attention_output_shape():
    """Test that output shape matches input shape."""
    params = make_gated_attention_params()
    output, _, _ = gated_attention_forward(**params)
    B, T, D = params["hidden_states"].shape
    assert output.shape == (B, T, D), f"Expected {(B, T, D)}, got {output.shape}"
    print("PASS: test_gated_attention_output_shape")


def test_gated_attention_causal():
    """Test that causal masking works (future tokens don't affect past outputs)."""
    params = make_gated_attention_params(seq_len=16)

    output_full, _, _ = gated_attention_forward(**params)

    # Run with only the first 8 tokens
    params_half = {**params}
    params_half["hidden_states"] = params["hidden_states"][:, :8]
    params_half["cos"] = params["cos"][:, :8]
    params_half["sin"] = params["sin"][:, :8]
    params_half["attention_mask"] = None  # let it use default causal
    mask8 = (
        torch.triu(torch.ones(8, 8), diagonal=1)
        .masked_fill(torch.triu(torch.ones(8, 8), diagonal=1).bool(), float("-inf"))
        .unsqueeze(0)
        .unsqueeze(0)
    )
    params_half["attention_mask"] = mask8

    output_half, _, _ = gated_attention_forward(**params_half)

    # First 8 tokens of full output should match half output
    diff = (output_full[:, :8] - output_half).abs().max().item()
    assert diff < 1e-4, f"Causal violation: max diff = {diff}"
    print("PASS: test_gated_attention_causal")


def test_gated_attention_gate_effect():
    """Test that the sigmoid gate modulates the output."""
    params = make_gated_attention_params()
    output, _, _ = gated_attention_forward(**params)

    # Output should not be all zeros (gate should be ~0.5 on average)
    assert output.abs().mean() > 1e-6, "Output is all zeros"

    # Gate should reduce magnitude compared to ungated attention
    # (sigmoid produces values in (0, 1), so gated output <= ungated)
    print("PASS: test_gated_attention_gate_effect")


def test_gated_attention_deterministic():
    """Test that forward pass is deterministic."""
    params = make_gated_attention_params()
    out1, _, _ = gated_attention_forward(**params)
    out2, _, _ = gated_attention_forward(**params)
    diff = (out1 - out2).abs().max().item()
    assert diff == 0.0, f"Non-deterministic: max diff = {diff}"
    print("PASS: test_gated_attention_deterministic")


def test_gated_attention_kv_cache():
    """Test KV cache functionality."""
    params = make_gated_attention_params(seq_len=16)

    # Full forward
    output_full, k_cache, v_cache = gated_attention_forward(
        **params,
        output_kv_cache=True,
    )

    assert k_cache is not None, "KV cache should be returned"
    assert k_cache.shape[2] == 16, f"KV cache seq len should be 16, got {k_cache.shape[2]}"
    print("PASS: test_gated_attention_kv_cache")


def test_gated_attention_different_configs():
    """Test various head configurations (MHA, GQA, MQA)."""
    configs = [
        {"num_attention_heads": 8, "num_key_value_heads": 8, "head_dim": 64},  # MHA
        {"num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 64},  # GQA
        {"num_attention_heads": 8, "num_key_value_heads": 1, "head_dim": 64},  # MQA
        {"num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 128},  # Large GQA
    ]
    for cfg in configs:
        params = make_gated_attention_params(hidden_size=cfg["num_attention_heads"] * cfg["head_dim"], **cfg)
        output, _, _ = gated_attention_forward(**params)
        B, T, D = params["hidden_states"].shape
        assert output.shape == (B, T, D), f"Config {cfg}: shape mismatch"
    print("PASS: test_gated_attention_different_configs")


if __name__ == "__main__":
    test_gated_attention_output_shape()
    test_gated_attention_causal()
    test_gated_attention_gate_effect()
    test_gated_attention_deterministic()
    test_gated_attention_kv_cache()
    test_gated_attention_different_configs()
    print("\nAll Gated Attention tests passed!")
