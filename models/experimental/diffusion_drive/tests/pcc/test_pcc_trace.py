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

**Capture and replay deliberately use different inputs.** Trace inputs live at
fixed device addresses, and ``run_backbone_trace`` refills them before each
replay. Capturing and replaying the same features cannot distinguish a working
refill from a trace that ignored it and replayed capture-time data — both give
PCC 1.0. So capture on A, replay on B, and check the result against eager B
*and* against eager A: matching B proves the refill happened, and the A
comparison is the control that proves the two inputs are far enough apart for
that to mean anything.

The noise stream is re-seeded before each forward (README 3.5) so the DDIM draws
match.
Assets resolve through the shared ``checkpoint_path`` fixture; a missing one
skips locally but fails under ``DD_REQUIRE_ASSETS=1`` (see conftest).
"""

from __future__ import annotations

import pytest
import torch

from models.common.utility_functions import comp_pcc
from models.experimental.diffusion_drive.reference.model import DiffusionDriveConfig, load_model
from models.experimental.diffusion_drive.tt.ttnn_diffusion_drive import TtnnDiffusionDriveModel

# Minimum gap between traced(B)-vs-eager(B) and traced(B)-vs-eager(A).
#
# Measured separation is ~0.0299: replay reproduces eager(B) exactly (PCC
# 1.000000) while eager(A)-vs-eager(B) sits at 0.970103. Note how high that is —
# the trajectories are anchor-structured, so even well-separated inputs stay
# strongly correlated, and the usable margin is the ~0.03 gap rather than the
# absolute PCC. 0.01 leaves ~3x headroom over that gap while still being far
# below it, so a replay that ignored the refill (separation 0) cannot clear it.
_MIN_SEPARATION = 0.01


def _features(scale: float = 1.0) -> dict:
    """Production-resolution random features (README 3.6)."""
    return {
        "camera_feature": torch.randn(1, 3, 256, 1024) * scale,
        "lidar_feature": torch.randn(1, 1, 256, 256) * scale,
        "status_feature": torch.randn(1, 8),
    }


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
    ttnn_model.build_all(device)

    # Two materially different inputs: independent draws, and B is scaled so the
    # stem activations differ in magnitude as well as in sign pattern.
    torch.manual_seed(7)
    features_a = _features()
    torch.manual_seed(99)
    features_b = _features(scale=2.5)

    # Eager references for both, before any trace exists.
    torch.manual_seed(1234)  # pin DDIM noise (README 3.5)
    eager_a = ttnn_model(features_a)
    torch.manual_seed(1234)  # same noise stream
    eager_b = ttnn_model(features_b)

    # Diagnostic only: how far apart the two inputs drive the model. The
    # separation assertion below is what actually keeps the test non-vacuous.
    ab_pcc = comp_pcc(eager_a["trajectory"], eager_b["trajectory"])[1]
    print(f"eager A-vs-B trajectory PCC = {ab_pcc:.6f}")

    # Capture on A, replay on B. The replay refills the fixed-address trace inputs.
    ttnn_model.compile(features_a)
    assert ttnn_model._compiled

    torch.manual_seed(1234)  # same noise stream
    traced_b = ttnn_model.execute_compiled(features_b)

    pcc = comp_pcc(eager_b["trajectory"], traced_b["trajectory"])[1]
    print(f"traced(B)-vs-eager(B) trajectory PCC = {pcc:.6f}")
    assert pcc >= 0.99, f"traced trajectory PCC {pcc:.6f} < 0.99"

    # A trace that replayed its capture-time inputs would track A, not B. One
    # assertion covers both failure modes: a broken refill drives stale_pcc up to
    # pcc, and inputs too alike to tell apart do the same, so either way the
    # separation collapses and the test fails instead of passing vacuously.
    stale_pcc = comp_pcc(eager_a["trajectory"], traced_b["trajectory"])[1]
    print(f"traced(B)-vs-eager(A) trajectory PCC = {stale_pcc:.6f} (must NOT match)")
    assert pcc - stale_pcc > _MIN_SEPARATION, (
        f"replay is not tracking input B: traced-vs-eager(B) {pcc:.6f} is not meaningfully "
        f"better than traced-vs-eager(A) {stale_pcc:.6f}. Either the trace-input refill is "
        f"not happening, or features A and B (eager PCC {ab_pcc:.6f}) are too alike to tell apart."
    )

    ttnn_model.release_compiled()
    assert not ttnn_model._compiled
