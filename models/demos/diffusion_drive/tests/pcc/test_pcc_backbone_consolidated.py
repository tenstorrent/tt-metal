# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PCC + wiring tests for the device-native consolidated backbone (Stage 5).

The consolidated path keeps the image/LiDAR feature maps on-device across all
four ResNet-34 stages and the 4× GPT fusion, removing the 8 per-stage host
round-trips of the staged path.  It is enabled **by default** once the FPN
(build_stage3) and the stems+fusion (build_stage3_6) are installed; the staged
path stays reachable via DD_CONSOLIDATE=0.

These tests guard:
  • the round-trip-removal *identity* — consolidated vs staged ≈ 1.0 (the
    full-model Stage-3.6/4 tests now run the consolidated path, so this is the
    only place the staged backbone is still exercised against it);
  • consolidated vs the PyTorch reference ≥ 0.99 (bf16);
  • the default-on wiring and the DD_CONSOLIDATE=0 escape hatch.

Runs at the production resolution (camera 256×1024, LiDAR 256×256) — the GPT
fusion's avg_pool/upsample ratios are only integer there (same as Stage 3.6).
"""

from __future__ import annotations

import pytest
import torch

from models.demos.diffusion_drive.reference.model import DiffusionDriveConfig, DiffusionDriveModel, TransfuserBackbone
from models.demos.diffusion_drive.tt.ttnn_backbone import TtnnTransfuserBackbone
from models.demos.diffusion_drive.tt.ttnn_diffusion_drive import TtnnDiffusionDriveModel
from models.demos.diffusion_drive.tt.ttnn_fpn import TtnnFPN
from models.demos.diffusion_drive.tt.ttnn_gpt_fusion import TtnnFuseFeatures

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
    """Production-resolution TransfuserBackbone with random weights (no checkpoint).

    Uses the default LiDAR resolution (256×256) so the GPT-fusion pool/upsample
    ratios are integer; latent=True supplies the LiDAR so no sensor data is needed.
    """
    cfg = DiffusionDriveConfig(plan_anchor_path=None, latent=True)
    bb = TransfuserBackbone(cfg)
    bb.eval()
    return bb


def _install_prereqs(ttnn_bb: TtnnTransfuserBackbone, ref: TransfuserBackbone, device) -> None:
    """Install the three consolidation prerequisites (FPN + stems + fusion) directly.

    Mirrors what build_stage3 + build_stage3_6 install, but without going through
    the model wrapper — so the consolidated flag is left OFF (these low-level
    installs do not auto-enable) and the caller controls the staged↔consolidated
    switch explicitly.
    """
    ttnn_bb._ttnn_fpn = TtnnFPN(ref, device)
    ttnn_bb.install_stems(device)
    ttnn_bb.install_fusion(TtnnFuseFeatures(ref, device))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.timeout(1800)
def test_consolidated_matches_staged_and_reference(device) -> None:
    """consolidated == staged (identity) and both ≥ 0.99 vs the reference backbone."""
    torch.manual_seed(42)
    bb_ref = _make_backbone()
    ttnn_bb = TtnnTransfuserBackbone(bb_ref, device)
    _install_prereqs(ttnn_bb, bb_ref, device)

    camera = torch.randn(1, 3, 256, 1024)
    lidar_dummy = torch.zeros(1, 1, 256, 256)  # unused (latent=True)

    with torch.no_grad():
        ref_up, ref_feat, _ = bb_ref(camera, lidar_dummy)

        assert ttnn_bb._consolidated is False, "low-level installs must not auto-enable"
        staged_up, staged_feat, _ = ttnn_bb(camera, lidar_dummy)

        ttnn_bb.enable_consolidated()
        assert ttnn_bb._consolidated is True
        cons_up, cons_feat, _ = ttnn_bb(camera, lidar_dummy)

    pccs = {
        "bev_upscale  cons-vs-ref": _pcc(cons_up, ref_up),
        "bev_feature  cons-vs-ref": _pcc(cons_feat, ref_feat),
        "bev_upscale  cons-vs-staged": _pcc(cons_up, staged_up),
        "bev_feature  cons-vs-staged": _pcc(cons_feat, staged_feat),
    }
    for k, v in pccs.items():
        print(f"{k:30s} PCC = {v:.6f}")

    # consolidated reproduces the reference (bf16 backbone)
    assert pccs["bev_upscale  cons-vs-ref"] >= 0.99
    assert pccs["bev_feature  cons-vs-ref"] >= 0.99
    # removing the per-stage host round-trips is an identity transform
    assert pccs["bev_upscale  cons-vs-staged"] >= 0.999
    assert pccs["bev_feature  cons-vs-staged"] >= 0.999


@pytest.mark.timeout(600)
def test_consolidated_enabled_by_default_and_escape_hatch(device, monkeypatch) -> None:
    """_maybe_enable_consolidated flips on when prereqs land; DD_CONSOLIDATE=0 opts out."""
    torch.manual_seed(0)
    bb_ref = _make_backbone()

    # No prerequisites installed yet → auto-enable is a silent no-op.
    bb = TtnnTransfuserBackbone(bb_ref, device)
    assert bb._maybe_enable_consolidated() is False
    assert bb._consolidated is False

    # All three prerequisites present → consolidated turns on by default.
    _install_prereqs(bb, bb_ref, device)
    assert bb._maybe_enable_consolidated() is True
    assert bb._consolidated is True
    assert bb._maybe_enable_consolidated() is True  # idempotent

    # DD_CONSOLIDATE=0 keeps the staged path even with prereqs present, but the
    # explicit enable_consolidated() primitive still forces it (env is ignored).
    monkeypatch.setenv("DD_CONSOLIDATE", "0")
    bb2 = TtnnTransfuserBackbone(bb_ref, device)
    _install_prereqs(bb2, bb_ref, device)
    assert bb2._maybe_enable_consolidated() is False
    assert bb2._consolidated is False
    bb2.enable_consolidated()
    assert bb2._consolidated is True


@pytest.mark.timeout(600)
def test_model_build_enables_consolidated_by_default(device, model_config) -> None:
    """The documented build chain (build_stage2 → 3 → 3_6) auto-enables consolidation."""
    if model_config.plan_anchor_path is None:
        pytest.skip("plan_anchor_path not set — run scripts/prepare_assets.py first")

    torch.manual_seed(42)
    cfg = DiffusionDriveConfig(plan_anchor_path=model_config.plan_anchor_path, latent=True)
    ref_model = DiffusionDriveModel(cfg).eval()

    model = TtnnDiffusionDriveModel(ref_model, model_config, device)
    model.build_stage2(device).build_stage3(device).build_stage3_6(device)
    assert model._model._backbone._ttnn._consolidated is True

    # build_stage5 is the explicit (redundant) entry point — must stay a no-op here.
    model.build_stage5(device)
    assert model._model._backbone._ttnn._consolidated is True
