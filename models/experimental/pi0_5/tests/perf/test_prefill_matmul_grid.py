# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Core-grid sweep for the TP=4 VLM-prefill matmuls (seq=1024, 3-camera).

Answers: do the per-chip prefill matmuls regress on a 64-core (8×8) grid vs the
96-core (12×8) grid they use today? This gates whether a fully-width-sharded
backbone (everything on the norm's 64-core grid → no I2S/S2I reshards) can pay
for itself, since the block-sharded RMSNorm caps at 8×8 for hidden=2048.

Per-chip shapes at TP=4 (bf8 weights, bf16 acts):
  mlp_gate_up : M=1024 K=2048  N=4096   (col-parallel, mlp_dim/4)
  mlp_down    : M=1024 K=4096  N=2048   (row-parallel)
  attn_qkv    : M=1024 K=2048  N=2560   (replicated, fused Q|K|V)
  attn_o      : M=1024 K=2048  N=2048   (replicated)

PCC vs torch (bf8 single matmul) ≥ 0.99. Device-kernel time is read from tracy.
Use a non-mmio chip for clean timing, e.g. TT_VISIBLE_DEVICES=9,8,10,11.
"""

import pytest
import torch
import ttnn

from models.experimental.pi0_5.tt.ttnn_gemma import build_matmul_pcfg

# (name, M, K, N)
_SHAPES = [
    ("mlp_gate_up", 1024, 2048, 4096),
    ("mlp_down", 1024, 4096, 2048),
    ("attn_qkv", 1024, 2048, 2560),
    ("attn_o", 1024, 2048, 2048),
]
_GRIDS = [(8, 8), (12, 8), (12, 10)]


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return (torch.mean((a - a.mean()) * (b - b.mean())) / (a.std() * b.std() + 1e-9)).item()


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("grid", _GRIDS, ids=[f"{g[0]}x{g[1]}" for g in _GRIDS])
@pytest.mark.parametrize("shape", _SHAPES, ids=[s[0] for s in _SHAPES])
def test_prefill_matmul_grid(device, shape, grid):
    name, M, K, N = shape
    gx, gy = grid
    m_t, k_t, n_t = M // 32, K // 32, N // 32

    pcfg = build_matmul_pcfg(m_t, k_t, n_t, gx, gy, in0_block_w=8)
    if pcfg is None:
        pytest.skip(f"no pcfg for {name} @ {gx}x{gy}")

    torch.manual_seed(0)
    act = torch.randn(1, 1, M, K) * 0.1
    w = torch.randn(K, N) * 0.02
    ref = act.reshape(M, K).float() @ w.float()  # [M, N]

    act_tt = ttnn.from_torch(
        act, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    w_tt = ttnn.from_torch(
        w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    out = ttnn.linear(act_tt, w_tt, program_config=pcfg, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
    out_t = ttnn.to_torch(out).reshape(M, N)

    pcc = _pcc(ref, out_t)
    print(f"\n{name} {gx}x{gy} ({gx*gy} cores)  M={M} K={K} N={N}  PCC={pcc:.5f}")
    assert pcc >= 0.99, f"{name} {gx}x{gy}: PCC {pcc:.5f} < 0.99"
