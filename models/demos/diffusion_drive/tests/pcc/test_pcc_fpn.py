# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PCC test for TtnnFPN — the TTNN Stage-3 replacement of TransfuserBackbone._top_down.

Validates that the three TTNN conv2d layers (c5_conv, up_conv5, up_conv4)
produce output with PCC ≥ 0.99 vs the PyTorch reference _top_down on a
synthetic (1, 512, 8, 8) input matching the real backbone layer4 output shape.

Bilinear upsampling steps remain in PyTorch in both implementations so they
are not a source of divergence.

No real checkpoint or plan_anchor_path is required — TransfuserBackbone is
instantiated with random weights.
"""

from __future__ import annotations

import pytest
import torch

from models.demos.diffusion_drive.reference.model import DiffusionDriveConfig, TransfuserBackbone
from models.demos.diffusion_drive.tt.ttnn_fpn import TtnnFPN

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom < 1e-12:
        return 1.0
    return (a @ b).item() / denom


def _make_backbone() -> TransfuserBackbone:
    """Build a TransfuserBackbone with random FPN weights (no checkpoint needed)."""
    cfg = DiffusionDriveConfig(plan_anchor_path=None, latent=True)
    bb = TransfuserBackbone(cfg)
    bb.eval()
    return bb


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.timeout(300)
def test_fpn_pcc(device) -> None:
    """TtnnFPN vs _top_down reference PCC ≥ 0.99 on (1, 512, 8, 8) input."""
    torch.manual_seed(42)

    bb_ref = _make_backbone()
    ttnn_fpn = TtnnFPN(bb_ref, device)

    # Synthetic layer4 output shape: (B, 512, 8, 8) — matches real backbone
    x = torch.randn(1, 512, 8, 8)

    with torch.no_grad():
        ref_out = bb_ref._top_down(x)  # (1, 64, 64, 64)

    ttnn_out = ttnn_fpn(x)  # (1, 64, 64, 64)

    assert ttnn_out.shape == ref_out.shape, f"shape mismatch: {ttnn_out.shape} vs {ref_out.shape}"

    pcc = _pcc(ttnn_out, ref_out)
    assert pcc >= 0.99, f"FPN PCC {pcc:.6f} < 0.99"
