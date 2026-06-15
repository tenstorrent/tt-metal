# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for BOTH denoise matmul paths now that matmul_decode is DEFAULT ON.

`PI0_MMDECODE_DENOISE` defaults to ON (CLAUDE-optimized resident-weight matmul_decode at
M=32). Setting it to 0/false/no/off falls back to the native denoise matmuls. Because the
native path is no longer the default, it must stay covered — this test exercises BOTH:

  1. test_mmdecode_denoise_default_enabled  (device-free): the flag defaults to ON and the
     off-switch values disable it.
  2. test_mmdecode_denoise_paths_pcc (device): at the four real M=32 denoise shapes, BOTH
     the matmul_decode path and the native ttnn.linear path are numerically correct
     (PCC >= 0.99 vs torch). This guards the non-default (native) path from rotting.

Run:
  PYTHONPATH=<clone> <clone>/python_env/bin/pytest -xvs \\
    models/experimental/pi0_5/tests/pcc/test_pcc_mmdecode_denoise_paths.py
"""

from __future__ import annotations

import torch
import pytest
import ttnn

from models.experimental.pi0_5.tt.ttnn_gemma import _mmdecode_denoise_enabled
from models.experimental.pi0_5.tt.mmdecode.matmul_decode_linear import MatmulDecodeLinear

M = 32
PCC_THRESHOLD = 0.99
SHAPES = [
    ("qkv_fused", 1024, 2560),
    ("mlp_gate_up", 1024, 4096),
    ("o_proj", 2048, 1024),
    ("mlp_down", 4096, 1024),
]


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    t1, t2 = a.flatten().float(), b.flatten().float()
    if t1.numel() != t2.numel():
        return -1.0
    m1, m2 = torch.mean(t1), torch.mean(t2)
    s1, s2 = torch.std(t1), torch.std(t2)
    if s1 < 1e-9 or s2 < 1e-9:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    return (torch.mean((t1 - m1) * (t2 - m2)) / (s1 * s2)).item()


def test_mmdecode_denoise_default_enabled(monkeypatch):
    """matmul_decode denoise is DEFAULT ON; only 0/false/no/off disables it (device-free)."""
    monkeypatch.delenv("PI0_MMDECODE_DENOISE", raising=False)
    assert _mmdecode_denoise_enabled() is True, "default must be ON (not opt-in)"
    for on in ("1", "true", "yes", "on", "ON", "True"):
        monkeypatch.setenv("PI0_MMDECODE_DENOISE", on)
        assert _mmdecode_denoise_enabled() is True
    for off in ("0", "false", "no", "off", "OFF", "False"):
        monkeypatch.setenv("PI0_MMDECODE_DENOISE", off)
        assert _mmdecode_denoise_enabled() is False, f"{off!r} must disable (non-default native path)"


@pytest.mark.parametrize("label,K,N", SHAPES, ids=[s[0] for s in SHAPES])
def test_mmdecode_denoise_paths_pcc(device, label, K, N):
    """BOTH paths correct at the real M=32 denoise shapes: native (non-default) AND matmul_decode (default)."""
    torch.manual_seed(1234)
    a_torch = torch.randn(1, M, K) * 0.1
    w_KN = torch.randn(K, N) * 0.02
    ref = a_torch.reshape(M, K).float() @ w_KN.float()

    a_dev = ttnn.from_torch(a_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # --- non-default path: native ttnn.linear (bf8_b weight, bf16 out, as the model uses) ---
    w_dev = ttnn.from_torch(w_KN, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    y_native = ttnn.to_torch(ttnn.linear(a_dev, w_dev, dtype=ttnn.bfloat16)).reshape(M, N).float()
    pcc_native = _pcc(ref, y_native)

    # --- default path: CLAUDE-optimized resident-weight matmul_decode ---
    lin = MatmulDecodeLinear(
        device, w_KN.clone(), weight_dtype=ttnn.bfloat8_b, out_dtype=ttnn.bfloat16, role=f"denoise_{label}"
    )
    y_mmd = ttnn.to_torch(lin(a_dev)).reshape(M, N).float()
    pcc_mmd = _pcc(ref, y_mmd)

    assert pcc_native >= PCC_THRESHOLD, f"native (non-default) path PCC {pcc_native:.5f} < {PCC_THRESHOLD} [{label}]"
    assert pcc_mmd >= PCC_THRESHOLD, f"matmul_decode (default) path PCC {pcc_mmd:.5f} < {PCC_THRESHOLD} [{label}]"
