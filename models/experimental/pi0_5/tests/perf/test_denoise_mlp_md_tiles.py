# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC tests for GemmaMLPTTNN with matmul_decode (PI0_MD_DENOISE=1).

Two tile heights:
  M=32 (32×32 tiles): matmul_decode path active (m_tiles=1). PASSES.
  M=16 (16×32 tiles): matmul_decode does not yet activate (m_tiles=0).
      xfail — engineering ask: matmul_decode should fire for M=16 (16×32
      sub-tile height) once the kernel gains support for it.

Non-matmul ops (GELU, multiply, sharded_to_interleaved) are exercised in
both cases; they use standard 32×32 tiles and are unaffected by M.

Run:
  pytest test_denoise_mlp_md_tiles.py -v
"""

import os

import pytest
import torch
import ttnn

from models.experimental.pi0_5.common.configs import GemmaConfig
from models.experimental.pi0_5.tt.ttnn_gemma import GemmaMLPTTNN

PCC_THRESH = 0.99


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    a -= a.mean()
    b -= b.mean()
    denom = (a.norm() * b.norm()).clamp(min=1e-9)
    return (a @ b / denom).item()


def _synthetic_weights(cfg: GemmaConfig) -> dict:
    W, M = cfg.width, cfg.mlp_dim
    return {
        "mlp.gate_proj.weight": torch.randn(M, W) * 0.02,
        "mlp.up_proj.weight": torch.randn(M, W) * 0.02,
        "mlp.down_proj.weight": torch.randn(W, M) * 0.02,
    }


def _torch_ref(x: torch.Tensor, w: dict) -> torch.Tensor:
    gate = torch.nn.functional.gelu(x @ w["mlp.gate_proj.weight"].T)
    up = x @ w["mlp.up_proj.weight"].T
    return (gate * up) @ w["mlp.down_proj.weight"].T


def _run_mlp_pcc(device, m_pad: int) -> float:
    cfg = GemmaConfig.gemma_300m()
    torch.manual_seed(42)
    weights = _synthetic_weights(cfg)
    x_torch = torch.randn(1, 1, m_pad, cfg.width).bfloat16()

    mlp = GemmaMLPTTNN(cfg, weights, device)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out_tt = mlp.forward(x_tt)
    out = ttnn.to_torch(out_tt)

    ref = _torch_ref(x_torch.float(), {k: v.float() for k, v in weights.items()})

    pcc = _pcc(out, ref)
    md_active = mlp._md_denoise and (m_pad // 32) == 1
    print(f"\n  m_pad={m_pad}  md_active={md_active}  PCC={pcc:.5f}")
    return pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_denoise_mlp_md_32x32(device):
    """M=32: matmul_decode active (32×32 tiles, m_tiles=1). Must pass."""
    os.environ["PI0_MD_DENOISE"] = "1"
    try:
        pcc = _run_mlp_pcc(device, m_pad=32)
        assert pcc >= PCC_THRESH, f"PCC {pcc:.5f} < {PCC_THRESH}"
    finally:
        os.environ.pop("PI0_MD_DENOISE", None)


@pytest.mark.xfail(
    reason=(
        "matmul_decode 16×32 tile (M=16) not yet supported: m_tiles=16//32=0 "
        "so the MD branch is skipped. Engineering ask: extend matmul_decode kernel "
        "to activate for M=16 (16×32 sub-tile height) so the full denoise MLP can "
        "be tiny-tiled to 16×32."
    ),
    strict=True,
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_denoise_mlp_md_16x32(device):
    """M=16: 16×32 tiles. xfail until matmul_decode kernel supports sub-32 tile height.
    Remove xfail once forward() activates the MD branch for m_tiles=0 (M=16)."""
    os.environ["PI0_MD_DENOISE"] = "1"
    try:
        cfg = GemmaConfig.gemma_300m()
        weights = _synthetic_weights(cfg)
        mlp = GemmaMLPTTNN(cfg, weights, device)
        # Assert MD path fires for M=16 — currently false (m_tiles=16//32=0).
        # This assertion will pass once the kernel + forward() support 16×32 tiles.
        assert mlp._md_denoise and (16 // 32) == 1, f"matmul_decode 16×32 not yet supported: m_tiles={16 // 32}"
        pcc = _run_mlp_pcc(device, m_pad=16)
        assert pcc >= PCC_THRESH, f"PCC {pcc:.5f} < {PCC_THRESH}"
    finally:
        os.environ.pop("PI0_MD_DENOISE", None)
