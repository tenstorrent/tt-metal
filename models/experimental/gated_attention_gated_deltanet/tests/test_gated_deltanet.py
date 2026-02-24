"""
Tests for the functional torch Gated DeltaNet implementation.

Validates that the functional implementation produces correct outputs.
Tests both the recurrent and chunked modes, and verifies they produce
equivalent results.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from torch_functional.delta_rule_ops import (
    recurrent_gated_delta_rule,
    chunk_gated_delta_rule,
)
from torch_functional.gated_deltanet import gated_deltanet_forward


def make_delta_rule_inputs(
    batch_size=2,
    seq_len=128,
    num_heads=4,
    head_k_dim=64,
    head_v_dim=128,
    dtype=torch.float32,
):
    """Create random inputs for delta rule testing."""
    q = torch.randn(batch_size, seq_len, num_heads, head_k_dim, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_k_dim, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_v_dim, dtype=dtype)
    beta = torch.rand(batch_size, seq_len, num_heads, dtype=dtype)
    g = -torch.rand(batch_size, seq_len, num_heads, dtype=dtype) * 2  # negative log-decay
    return q, k, v, beta, g


def make_gated_deltanet_params(
    hidden_size=512,
    num_heads=4,
    num_v_heads=None,
    head_k_dim=128,
    head_v_dim=256,
    conv_kernel_size=4,
    use_gate=True,
    seq_len=64,
    batch_size=2,
    dtype=torch.float32,
):
    """Create random weights and inputs for full layer testing."""
    if num_v_heads is None:
        num_v_heads = num_heads

    key_dim = num_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim

    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)

    params = {
        "hidden_states": hidden_states,
        "q_proj_weight": torch.randn(key_dim, hidden_size, dtype=dtype) * 0.02,
        "k_proj_weight": torch.randn(key_dim, hidden_size, dtype=dtype) * 0.02,
        "v_proj_weight": torch.randn(value_dim, hidden_size, dtype=dtype) * 0.02,
        "a_proj_weight": torch.randn(num_v_heads, hidden_size, dtype=dtype) * 0.02,
        "b_proj_weight": torch.randn(num_v_heads, hidden_size, dtype=dtype) * 0.02,
        "o_proj_weight": torch.randn(hidden_size, value_dim, dtype=dtype) * 0.02,
        "q_conv_weight": torch.randn(key_dim, 1, conv_kernel_size, dtype=dtype) * 0.1,
        "k_conv_weight": torch.randn(key_dim, 1, conv_kernel_size, dtype=dtype) * 0.1,
        "v_conv_weight": torch.randn(value_dim, 1, conv_kernel_size, dtype=dtype) * 0.1,
        "q_conv_bias": None,
        "k_conv_bias": None,
        "v_conv_bias": None,
        "A_log": torch.rand(num_v_heads, dtype=dtype).uniform_(0, 16).log(),
        "dt_bias": torch.randn(num_v_heads, dtype=dtype),
        "o_norm_weight": torch.ones(head_v_dim, dtype=dtype),
        "g_proj_weight": torch.randn(value_dim, hidden_size, dtype=dtype) * 0.02 if use_gate else None,
        "num_heads": num_heads,
        "num_v_heads": num_v_heads,
        "head_k_dim": head_k_dim,
        "head_v_dim": head_v_dim,
        "conv_kernel_size": conv_kernel_size,
        "use_gate": use_gate,
    }
    return params


def test_recurrent_delta_rule_shape():
    """Test output shapes of recurrent delta rule."""
    q, k, v, beta, g = make_delta_rule_inputs()
    o, state = recurrent_gated_delta_rule(q, k, v, beta, g, output_final_state=True, use_qk_l2norm=True)

    B, T, H, V = q.shape[0], q.shape[1], q.shape[2], v.shape[3]
    K = q.shape[3]
    assert o.shape == (B, T, H, V), f"Output shape: expected {(B, T, H, V)}, got {o.shape}"
    assert state.shape == (B, H, K, V), f"State shape: expected {(B, H, K, V)}, got {state.shape}"
    print("PASS: test_recurrent_delta_rule_shape")


def test_chunk_delta_rule_shape():
    """Test output shapes of chunked delta rule."""
    q, k, v, beta, g = make_delta_rule_inputs()
    o, state = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=64, output_final_state=True, use_qk_l2norm=True)

    B, T, H, V = q.shape[0], q.shape[1], q.shape[2], v.shape[3]
    K = q.shape[3]
    assert o.shape == (B, T, H, V), f"Output shape: expected {(B, T, H, V)}, got {o.shape}"
    assert state.shape == (B, H, K, V), f"State shape: expected {(B, H, K, V)}, got {state.shape}"
    print("PASS: test_chunk_delta_rule_shape")


def test_recurrent_vs_chunk_equivalence():
    """Test that recurrent and chunked modes produce equivalent results."""
    q, k, v, beta, g = make_delta_rule_inputs(seq_len=128, num_heads=2, head_k_dim=32, head_v_dim=64)

    o_rec, s_rec = recurrent_gated_delta_rule(q, k, v, beta, g, output_final_state=True, use_qk_l2norm=True)
    o_chunk, s_chunk = chunk_gated_delta_rule(
        q, k, v, g, beta, chunk_size=64, output_final_state=True, use_qk_l2norm=True
    )

    output_diff = (o_rec - o_chunk).abs().max().item()
    state_diff = (s_rec - s_chunk).abs().max().item()

    # Allow for float32 accumulation differences
    assert output_diff < 1e-2, f"Output diff too large: {output_diff}"
    assert state_diff < 1e-2, f"State diff too large: {state_diff}"
    print(f"PASS: test_recurrent_vs_chunk_equivalence (output_diff={output_diff:.6f}, state_diff={state_diff:.6f})")


def test_recurrent_single_step():
    """Test single-step recurrent (T=1) for decode."""
    q, k, v, beta, g = make_delta_rule_inputs(seq_len=1, num_heads=4, head_k_dim=32, head_v_dim=64)

    initial_state = torch.randn(2, 4, 32, 64)
    o, new_state = recurrent_gated_delta_rule(
        q,
        k,
        v,
        beta,
        g,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm=True,
    )

    assert o.shape == (2, 1, 4, 64), f"Output shape: {o.shape}"
    assert new_state.shape == (2, 4, 32, 64), f"State shape: {new_state.shape}"
    print("PASS: test_recurrent_single_step")


def test_state_continuity():
    """Test that processing in two parts gives same result as one pass."""
    q, k, v, beta, g = make_delta_rule_inputs(seq_len=64, num_heads=2, head_k_dim=32, head_v_dim=64)

    # Full pass
    o_full, _ = recurrent_gated_delta_rule(q, k, v, beta, g, use_qk_l2norm=True)

    # Two-part pass
    o1, state1 = recurrent_gated_delta_rule(
        q[:, :32],
        k[:, :32],
        v[:, :32],
        beta[:, :32],
        g[:, :32],
        output_final_state=True,
        use_qk_l2norm=True,
    )
    o2, _ = recurrent_gated_delta_rule(
        q[:, 32:],
        k[:, 32:],
        v[:, 32:],
        beta[:, 32:],
        g[:, 32:],
        initial_state=state1,
        use_qk_l2norm=True,
    )
    o_parts = torch.cat([o1, o2], dim=1)

    diff = (o_full - o_parts).abs().max().item()
    assert diff < 1e-4, f"State continuity violated: max diff = {diff}"
    print(f"PASS: test_state_continuity (diff={diff:.6f})")


def test_gated_deltanet_layer_shape():
    """Test full GatedDeltaNet layer output shape."""
    params = make_gated_deltanet_params()
    output, cache = gated_deltanet_forward(**params)

    B, T, D = params["hidden_states"].shape
    assert output.shape == (B, T, D), f"Expected {(B, T, D)}, got {output.shape}"
    print("PASS: test_gated_deltanet_layer_shape")


def test_gated_deltanet_layer_with_cache():
    """Test GatedDeltaNet layer with state caching."""
    params = make_gated_deltanet_params()
    params["output_final_state"] = True
    output, cache = gated_deltanet_forward(**params)

    assert cache is not None, "Cache should be returned"
    assert "recurrent_state" in cache, "Cache should contain recurrent_state"
    assert cache["recurrent_state"] is not None, "Recurrent state should not be None"
    print("PASS: test_gated_deltanet_layer_with_cache")


def test_gated_deltanet_deterministic():
    """Test determinism of forward pass."""
    params = make_gated_deltanet_params()
    o1, _ = gated_deltanet_forward(**params)
    o2, _ = gated_deltanet_forward(**params)
    diff = (o1 - o2).abs().max().item()
    assert diff == 0.0, f"Non-deterministic: diff = {diff}"
    print("PASS: test_gated_deltanet_deterministic")


def test_gated_deltanet_no_gate():
    """Test GatedDeltaNet without output gating."""
    params = make_gated_deltanet_params(use_gate=False)
    output, _ = gated_deltanet_forward(**params)
    B, T, D = params["hidden_states"].shape
    assert output.shape == (B, T, D), f"Expected {(B, T, D)}, got {output.shape}"
    print("PASS: test_gated_deltanet_no_gate")


if __name__ == "__main__":
    test_recurrent_delta_rule_shape()
    test_chunk_delta_rule_shape()
    test_recurrent_vs_chunk_equivalence()
    test_recurrent_single_step()
    test_state_continuity()
    test_gated_deltanet_layer_shape()
    test_gated_deltanet_layer_with_cache()
    test_gated_deltanet_deterministic()
    test_gated_deltanet_no_gate()
    print("\nAll Gated DeltaNet tests passed!")
