# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
match. Assets resolve through the shared ``checkpoint_path`` fixture; a missing one
skips locally but fails under ``DD_REQUIRE_ASSETS=1`` (see conftest).
"""

from __future__ import annotations

import pytest
import torch

from models.common.utility_functions import comp_pcc
from models.experimental.diffusion_drive.reference.model import DiffusionDriveConfig, load_model
from models.experimental.diffusion_drive.tt.ttnn_diffusion_drive import TtnnDiffusionDriveModel


@pytest.mark.timeout(300)
def test_backbone_trace_matches_eager(device, model_config, checkpoint_path, missing_asset) -> None:
    if model_config.plan_anchor_path is None:
        missing_asset("plan_anchor_path not set — run scripts/prepare_assets.py first")
    ckpt = checkpoint_path
    if ckpt is None:
        missing_asset("real checkpoint not found — run scripts/prepare_assets.py or set DD_CHECKPOINT_PATH")

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

    pcc = comp_pcc(eager_out["trajectory"], traced_out["trajectory"])[1]
    print(f"traced-vs-eager trajectory PCC = {pcc:.6f}")
    assert pcc >= 0.99, f"traced trajectory PCC {pcc:.6f} < 0.99"

    ttnn_model.release_compiled()
    assert not ttnn_model._compiled
