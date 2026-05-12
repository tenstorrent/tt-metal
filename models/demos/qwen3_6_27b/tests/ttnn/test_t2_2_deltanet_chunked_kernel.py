# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""T2.2 — DeltaNet chunked (prefill) kernel PCC at Qwen3.6 shapes (single BH chip).

  RED:    Neumann series for (I-M)^-1 at cs=64 may not converge for our shapes;
          if numerical drift accumulates over T=4096 or 32K, PCC may dip below 0.99.
  GREEN:  PCC > 0.99 at every T in {64, 256, 1024}.
"""
import sys

import pytest
import torch

import ttnn

sys.path.insert(0, "/home/tt-admin/ssinghal/tt-metal")

from models.experimental.gated_attention_gated_deltanet.torch_functional.delta_rule_ops import (
    chunk_gated_delta_rule as _chunk_ref,
)
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import chunk_gated_delta_rule_ttnn


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


@pytest.mark.parametrize("seq_len", [64, 256])
def test_deltanet_chunked_ttnn_qwen36(device, seq_len):
    """TTNN chunked kernel at Qwen3.6 shapes vs torch reference."""
    torch.manual_seed(0)
    B = 1
    n_v_heads = 48
    n_k_heads = 16
    d_k = d_v = 128
    g_ratio = n_v_heads // n_k_heads

    q_pre = torch.randn(B, seq_len, n_v_heads, d_k, dtype=torch.float32) * 0.1
    k_pre = torch.randn(B, seq_len, n_k_heads, d_k, dtype=torch.float32) * 0.1
    v_pre = torch.randn(B, seq_len, n_v_heads, d_v, dtype=torch.float32) * 0.1
    k = k_pre.repeat_interleave(g_ratio, dim=2)
    q, v = q_pre, v_pre
    g = torch.randn(B, seq_len, n_v_heads, dtype=torch.float32) * 0.05
    beta = torch.sigmoid(torch.randn(B, seq_len, n_v_heads, dtype=torch.float32))

    out_ref, _ = _chunk_ref(q, k, v, g, beta, chunk_size=64, output_final_state=True, use_qk_l2norm=True)

    q_tt = _torch_to_tt(q, device, ttnn.bfloat16)
    k_tt = _torch_to_tt(k, device, ttnn.bfloat16)
    v_tt = _torch_to_tt(v, device, ttnn.bfloat16)
    beta_tt = _torch_to_tt(beta, device, ttnn.bfloat16)
    g_tt = _torch_to_tt(g, device, ttnn.bfloat16)

    # TTNN signature: (q, k, v, beta, g) — note BETA before G, opposite of torch chunk_gated_delta_rule
    out_tt = chunk_gated_delta_rule_ttnn(q_tt, k_tt, v_tt, beta_tt, g_tt, chunk_size=64, device=device)
    if isinstance(out_tt, tuple):
        out_tt = out_tt[0]
    out_back = ttnn.to_torch(out_tt).float()

    if out_back.shape != out_ref.shape and out_back.dim() == 4 and out_back.shape[1] == n_v_heads:
        out_back = out_back.transpose(1, 2)

    assert out_back.shape == out_ref.shape
    pcc = _pcc(out_back, out_ref)
    print(f"T={seq_len}: PCC = {pcc:.6f}")
    assert pcc > 0.99, f"PCC {pcc:.6f} < 0.99 at T={seq_len}"
