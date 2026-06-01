# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Verify the standalone DeltaNet reference against itself:
  - decode (step-by-step) vs chunk-prefill must produce matching outputs + states
  - single GatedDeltaNetLayer in decode mode matches prefill on the same input
"""

import torch
import pytest

from models.demos.qwen36_27b.reference.deltanet_reference import (
    Qwen36Config,
    GatedDeltaNetLayer,
    recurrent_gated_delta_rule,
    chunk_gated_delta_rule,
)


@pytest.fixture
def small_config():
    return Qwen36Config(
        hidden_size=256,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        intermediate_size=512,
    )


def test_recurrent_vs_chunk_consistency():
    """recurrent decode over S tokens must match chunk prefill output and final state."""
    torch.manual_seed(42)
    B, S, H, Dk, Dv = 2, 17, 8, 32, 32

    query = torch.randn(B, S, H, Dk)
    key = torch.randn(B, S, H, Dk)
    value = torch.randn(B, S, H, Dv)
    g = torch.randn(B, S, H) * 0.5
    beta = torch.rand(B, S, H)

    out_rec, state_rec = recurrent_gated_delta_rule(query, key, value, g, beta, use_qk_l2norm=True)
    out_chunk, state_chunk = chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=8, use_qk_l2norm=True)

    torch.testing.assert_close(out_rec, out_chunk, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(state_rec, state_chunk, atol=1e-4, rtol=1e-3)


def test_recurrent_step_by_step_matches_batch():
    """Running recurrent one token at a time must match running the full sequence at once."""
    torch.manual_seed(123)
    B, S, H, Dk, Dv = 1, 10, 4, 16, 16

    query = torch.randn(B, S, H, Dk)
    key = torch.randn(B, S, H, Dk)
    value = torch.randn(B, S, H, Dv)
    g = torch.randn(B, S, H) * 0.3
    beta = torch.rand(B, S, H)

    out_full, state_full = recurrent_gated_delta_rule(query, key, value, g, beta, use_qk_l2norm=True)

    state = None
    outputs = []
    for t in range(S):
        out_t, state = recurrent_gated_delta_rule(
            query[:, t : t + 1],
            key[:, t : t + 1],
            value[:, t : t + 1],
            g[:, t : t + 1],
            beta[:, t : t + 1],
            initial_state=state,
            use_qk_l2norm=True,
        )
        outputs.append(out_t)

    out_step = torch.cat(outputs, dim=1)
    torch.testing.assert_close(out_step, out_full, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(state, state_full, atol=1e-5, rtol=1e-4)


def test_gated_deltanet_layer_decode_vs_prefill(small_config):
    """Full layer: prefill output must match sequential decode tokens."""
    torch.manual_seed(7)
    B, S = 1, 12
    layer = GatedDeltaNetLayer(small_config, layer_idx=0)
    layer.eval()

    x = torch.randn(B, S, small_config.hidden_size)

    with torch.no_grad():
        out_prefill, conv_state_pf, rec_state_pf = layer(x)

    with torch.no_grad():
        conv_state = None
        rec_state = None
        outputs = []
        for t in range(S):
            out_t, conv_state, rec_state = layer(
                x[:, t : t + 1],
                conv_state=conv_state,
                recurrent_state=rec_state,
            )
            outputs.append(out_t)

    out_decode = torch.cat(outputs, dim=1)

    torch.testing.assert_close(out_decode, out_prefill, atol=5e-4, rtol=1e-3)
    torch.testing.assert_close(rec_state, rec_state_pf, atol=5e-4, rtol=1e-3)


def test_state_dtype_float32():
    """DeltaNet state must be maintained in float32 for numerical stability."""
    torch.manual_seed(0)
    B, S, H, Dk, Dv = 1, 5, 4, 16, 16

    query = torch.randn(B, S, H, Dk, dtype=torch.bfloat16)
    key = torch.randn(B, S, H, Dk, dtype=torch.bfloat16)
    value = torch.randn(B, S, H, Dv, dtype=torch.bfloat16)
    g = torch.randn(B, S, H, dtype=torch.bfloat16) * 0.3
    beta = torch.rand(B, S, H, dtype=torch.bfloat16)

    _, state = recurrent_gated_delta_rule(query, key, value, g, beta)
    assert state.dtype == torch.float32
