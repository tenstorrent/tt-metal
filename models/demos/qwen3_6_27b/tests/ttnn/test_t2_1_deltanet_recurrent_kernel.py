# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""T2.1 — DeltaNet recurrent kernel PCC at Qwen3.6 shapes (single BH chip).

  Input:  H_QK=16, H_V=48, K=V=128, conv_k=4. Q/K/V pre-projected with GQA-expansion done outside.
  RED:    kernel hardcoded to WH compute configs; expect either hang or PCC <0.99 from BFP8 vs FP32
          mismatch in the SSM step; or matmul program config tuned for K=128,V=256 default may fail.
  GREEN:  PCC > 0.99 vs torch reference at every step.
"""
import sys

import pytest
import torch

import ttnn

sys.path.insert(0, "/home/tt-admin/ssinghal/tt-metal")

from models.experimental.gated_attention_gated_deltanet.torch_functional.delta_rule_ops import (
    recurrent_gated_delta_rule as _recurrent_ref,
)
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import recurrent_gated_delta_rule_ttnn


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


def _pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _torch_to_tt(t, device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)


@pytest.mark.parametrize("seq_len", [1, 4])
def test_deltanet_recurrent_ttnn_qwen36(device, seq_len):
    """TTNN recurrent kernel at Qwen3.6 shapes vs torch reference, single chip."""
    torch.manual_seed(0)
    B = 1
    n_v_heads = 48
    n_k_heads = 16
    d_k = d_v = 128
    g_ratio = n_v_heads // n_k_heads  # 3

    # Build inputs: K/V/Q post-GQA-expansion (kernel expects H=n_v_heads=48 for all)
    q_pre = torch.randn(B, seq_len, n_v_heads, d_k, dtype=torch.float32) * 0.1
    k_pre = torch.randn(B, seq_len, n_k_heads, d_k, dtype=torch.float32) * 0.1
    v_pre = torch.randn(B, seq_len, n_v_heads, d_v, dtype=torch.float32) * 0.1
    k = k_pre.repeat_interleave(g_ratio, dim=2)
    q = q_pre
    v = v_pre
    g = torch.randn(B, seq_len, n_v_heads, dtype=torch.float32) * 0.05
    beta = torch.sigmoid(torch.randn(B, seq_len, n_v_heads, dtype=torch.float32))

    # Torch reference (FP32)
    out_ref, state_ref = _recurrent_ref(q, k, v, beta, g, output_final_state=True, use_qk_l2norm=True)

    # TTNN (BF16/FP32 mix per kernel)
    q_tt = _torch_to_tt(q, device, ttnn.bfloat16)
    k_tt = _torch_to_tt(k, device, ttnn.bfloat16)
    v_tt = _torch_to_tt(v, device, ttnn.bfloat16)
    beta_tt = _torch_to_tt(beta, device, ttnn.bfloat16)
    g_tt = _torch_to_tt(g, device, ttnn.bfloat16)

    out_tt, state_tt = recurrent_gated_delta_rule_ttnn(q_tt, k_tt, v_tt, beta_tt, g_tt, device=device)

    out_back = ttnn.to_torch(out_tt).float()
    print(f"T={seq_len}: TTNN out shape={out_back.shape}, ref shape={out_ref.shape}")

    # Reshape to common form
    if out_back.shape != out_ref.shape:
        # Try transpose if dim order swapped
        if out_back.dim() == 4 and out_back.shape[1] == n_v_heads:
            out_back = out_back.transpose(1, 2)

    assert out_back.shape == out_ref.shape, f"shape: tt={out_back.shape}, ref={out_ref.shape}"
    pcc = _pcc(out_back, out_ref)
    print(f"T={seq_len}: PCC = {pcc:.6f}")
    assert pcc > 0.99, f"PCC {pcc:.6f} < 0.99 at T={seq_len}"
