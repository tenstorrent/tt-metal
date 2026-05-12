# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""T1.1 — DeltaNet recurrent vs chunked consistency at Qwen3.6 shapes.

TEST_PLAN.md §Phase 1:
  What:      recurrent and chunked DeltaNet paths produce identical output at every step.
  Input:     B=1, T=64, n_v=48, n_k=16, d_k=d_v=128, conv_k=4. Random bf16 weights, fixed seed.
  RED:       reference has not been verified at our exact head counts; expect a shape mismatch
             (n_v=48 vs default n_v=4 in the upstream FLA tests) or > 1e-2 deviation between paths.
  GREEN:     allclose(recurrent_out, chunked_out, atol=1e-3, rtol=1e-3) at every t ∈ [0, T).
  Depends:   none — first test of the relay.
"""
import sys

import pytest
import torch

sys.path.insert(0, "/home/tt-admin/ssinghal/tt-metal")

from models.experimental.gated_attention_gated_deltanet.torch_functional.delta_rule_ops import (
    chunk_gated_delta_rule as _chunked_kernel,
)
from models.experimental.gated_attention_gated_deltanet.torch_functional.delta_rule_ops import (
    recurrent_gated_delta_rule as _recurrent_kernel,
)


@pytest.mark.parametrize("seq_len", [1, 8, 64, 256])
def test_recurrent_matches_chunked_qwen36_shapes(seq_len):
    """At Qwen3.6 DeltaNet shapes (post-GQA-expand), recurrent and chunked outputs must agree.

    Both kernels expect Q/K/V/beta/g in the same head dim H = n_v_heads = 48.
    GQA expansion (K from 16 → 48 via repeat_interleave) happens BEFORE kernel call.
    """
    torch.manual_seed(0)
    B = 1
    n_v_heads = 48  # Qwen3.6 linear_num_value_heads (Q/K/V/beta/g all live in this dim)
    n_k_heads = 16  # Qwen3.6 linear_num_key_heads (K from proj; will be GQA-expanded)
    d_k = 128
    d_v = 128
    gqa_repeat = n_v_heads // n_k_heads  # 3

    # Pre-expansion shapes (post-projection): K in n_k_heads, V in n_v_heads, Q in n_v_heads
    # (After conv1d+SiLU, before kernel call, the wrapper does repeat_interleave on K)
    q_pre = torch.randn(B, seq_len, n_v_heads, d_k, dtype=torch.float32)
    k_pre = torch.randn(B, seq_len, n_k_heads, d_k, dtype=torch.float32)
    v_pre = torch.randn(B, seq_len, n_v_heads, d_v, dtype=torch.float32)

    # GQA expansion (matches the wrapper's repeat_interleave behaviour)
    k = k_pre.repeat_interleave(gqa_repeat, dim=2)  # [B, T, 48, 128]
    q = q_pre
    v = v_pre

    g = torch.randn(B, seq_len, n_v_heads, dtype=torch.float32) * 0.1
    beta = torch.sigmoid(torch.randn(B, seq_len, n_v_heads, dtype=torch.float32))

    # Note: kernel signatures have g and beta in DIFFERENT positional order!
    # recurrent: (q, k, v, beta, g, ...)
    # chunked:   (q, k, v, g, beta, ...)
    out_rec, state_rec = _recurrent_kernel(q, k, v, beta, g, output_final_state=True, use_qk_l2norm=True)
    out_chk, state_chk = _chunked_kernel(q, k, v, g, beta, chunk_size=64, output_final_state=True, use_qk_l2norm=True)

    assert out_rec.shape == out_chk.shape, f"shape mismatch: rec={out_rec.shape}, chk={out_chk.shape}"

    max_abs = (out_rec - out_chk).abs().max().item()
    rel = max_abs / (out_chk.abs().max().item() + 1e-9)
    assert torch.allclose(
        out_rec, out_chk, atol=1e-3, rtol=1e-3
    ), f"recurrent vs chunked mismatch at T={seq_len}: max_abs={max_abs:.6f}, rel={rel:.6f}"
    assert torch.allclose(
        state_rec.to(torch.float32), state_chk.to(torch.float32), atol=1e-3, rtol=1e-3
    ), f"state mismatch at T={seq_len}"


if __name__ == "__main__":
    for T in [1, 8, 64, 256]:
        test_recurrent_matches_chunked_qwen36_shapes(T)
        print(f"T={T}: PASS")
