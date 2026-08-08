# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Backbone-loop trace capture/replay gate (Stage 7).

``compile()`` captures the consolidated ``[stage → fusion] × 4`` backbone loop as
a TTNN trace; ``execute_compiled()`` replays it (collapsing the loop's per-op host
dispatch into one ``execute_trace``) and runs the not-yet-traced FPN/perception/
heads eagerly. This asserts the traced forward matches the eager ``__call__``
trajectory at production resolution with the real trained checkpoint — i.e. trace
replay is numerically transparent.

The noise stream is re-seeded before each forward (DD-5) so the two DDIM draws
match. Skips cleanly if the checkpoint or plan-anchor asset is absent (set
``DD_CHECKPOINT_PATH`` / ``DD_ANCHOR_PATH``).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from models.demos.diffusion_drive.reference.model import DiffusionDriveConfig, load_model
from models.demos.diffusion_drive.tt.ttnn_diffusion_drive import TtnnDiffusionDriveModel


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom < 1e-12:
        return 1.0
    return (a @ b).item() / denom


def _checkpoint_path() -> str | None:
    data_root = os.environ.get("DD_DATA_ROOT", "/mnt/diffusion-drive")
    candidates = [
        os.environ.get("DD_CHECKPOINT_PATH"),
        f"{data_root}/weights/diffusiondrive_navsim_88p1_PDMS.pth",
    ]
    return next((p for p in candidates if p and Path(p).exists()), None)


@pytest.mark.timeout(1800)
def test_backbone_trace_matches_eager(device, model_config) -> None:
    if model_config.plan_anchor_path is None:
        pytest.skip("plan_anchor_path not set — run scripts/prepare_assets.py first")
    ckpt = _checkpoint_path()
    if ckpt is None:
        pytest.skip("real checkpoint not found — set DD_CHECKPOINT_PATH")

    ref_cfg = DiffusionDriveConfig(plan_anchor_path=model_config.plan_anchor_path, latent=False)
    ref_model = load_model(ckpt, ref_cfg, device=torch.device("cpu")).eval()

    ttnn_model = TtnnDiffusionDriveModel(ref_model, model_config, device)
    (
        ttnn_model.build_stage2(device)
        .build_stage3(device)
        .build_stage3_4(device)
        .build_stage3_5(device)
        .build_stage3_6(device)
        .build_stage3_7(device)
        .build_stage4(device)
    )

    features = {
        "camera_feature": torch.randn(1, 3, 256, 1024),
        "lidar_feature": torch.randn(1, 1, 256, 256),
        "status_feature": torch.randn(1, 8),
    }

    # Eager forward (the [stage→fusion] loop runs op-by-op).
    torch.manual_seed(1234)  # pin DDIM noise (DD-5)
    eager_out = ttnn_model(features)

    # Capture the backbone-loop trace, then replay it on the SAME features.
    ttnn_model.compile(features)
    assert ttnn_model._compiled

    torch.manual_seed(1234)  # same noise stream
    traced_out = ttnn_model.execute_compiled(features)

    pcc = _pcc(traced_out["trajectory"], eager_out["trajectory"])
    print(f"traced-vs-eager trajectory PCC = {pcc:.6f}")
    assert pcc >= 0.99, f"traced trajectory PCC {pcc:.6f} < 0.99"

    ttnn_model.release_compiled()
    assert not ttnn_model._compiled
