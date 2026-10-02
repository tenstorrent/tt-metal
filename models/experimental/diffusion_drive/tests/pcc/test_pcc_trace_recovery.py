# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Trace-capture failure recovery: the device must stay usable after a failed capture.

``begin_trace_capture`` puts the command queue into recording state. Only
``end_trace_capture`` clears it — it is the sole path to ``record_end()``, which
resets ``trace_id_`` and turns sysmem bypass mode back off
(``tt_metal/distributed/fd_mesh_command_queue.cpp``). ``release_trace`` merely
drops the trace buffer.

So an exception raised between begin and end must be handled by ending capture
*first* and releasing after. A handler that only releases leaves the queue
recording, and the next host upload dies with ``Writes are not supported during
trace capture.`` — which matters because both production callers treat capture
failure as recoverable and keep using the same device
(``diffusiondrive_ttnn_inproc_agent._build`` logs "falling back to eager forward").
Without this test the fallback is advertised but never exercised.

These tests inject a failure inside the captured region at each of the two capture
sites, then assert that a plain H2D upload and a full eager forward both still
work on the same device.

Note: the ``device`` fixture is session-scoped. If the recovery path regresses,
this test leaves the queue recording and later tests in the session fail too —
that cascade *is* the bug being covered, so read a broad failure here as the
signal rather than as flakiness.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.experimental.diffusion_drive.reference.model import DiffusionDriveConfig, load_model
from models.experimental.diffusion_drive.tt.ttnn_diffusion_drive import TtnnDiffusionDriveModel

_INJECTED = "injected failure inside trace capture"


def _features() -> dict:
    """Production-resolution random features (README 3.6)."""
    return {
        "camera_feature": torch.randn(1, 3, 256, 1024),
        "lidar_feature": torch.randn(1, 1, 256, 256),
        "status_feature": torch.randn(1, 8),
    }


def _arm_failure(monkeypatch, owner, method_name: str) -> None:
    """Make ``owner.method_name`` raise, but only once capture is recording.

    Keyed off ``begin_trace_capture`` rather than a call counter so the injection
    stays put no matter how many warm-up passes run first — ``compile`` warms the
    full forward twice and each capture site double-warms its own body.
    """
    state = {"capturing": False}
    real_begin = ttnn.begin_trace_capture
    real_end = ttnn.end_trace_capture
    real_method = getattr(owner, method_name)

    def begin(*args, **kwargs):
        trace_id = real_begin(*args, **kwargs)
        state["capturing"] = True
        return trace_id

    def end(*args, **kwargs):
        # Clearing here matters for the perception site: compile() captures the
        # backbone trace first, so without this the flag would still be set during
        # the perception double warm-up and the failure would fire outside the
        # captured region — testing nothing.
        try:
            return real_end(*args, **kwargs)
        finally:
            state["capturing"] = False

    def failing(*args, **kwargs):
        if state["capturing"]:
            raise RuntimeError(_INJECTED)
        return real_method(*args, **kwargs)

    monkeypatch.setattr(ttnn, "begin_trace_capture", begin)
    monkeypatch.setattr(ttnn, "end_trace_capture", end)
    monkeypatch.setattr(owner, method_name, failing)


def _assert_device_usable(device, ttnn_model, features) -> None:
    """A host upload and a full eager forward must both still succeed."""
    # The direct probe: this is the call that fatals with "Writes are not
    # supported during trace capture." if the queue is still recording.
    probe = ttnn.from_torch(
        torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    ttnn.deallocate(probe)

    # And the recovery the agent actually advertises: keep serving eagerly.
    torch.manual_seed(1234)  # pin DDIM noise (README 3.5)
    out = ttnn_model(features)
    assert out["trajectory"].shape == (1, 8, 3)
    assert torch.isfinite(out["trajectory"]).all(), "eager fallback produced non-finite trajectory"


@pytest.mark.timeout(300)
@pytest.mark.parametrize("site", ["backbone", "perception"])
def test_capture_failure_leaves_device_usable(
    device, model_config, checkpoint_path, missing_asset, monkeypatch, expect_error, site
) -> None:
    if model_config.plan_anchor_path is None:
        missing_asset("plan_anchor_path not set — run scripts/prepare_assets.py first")
    ckpt = checkpoint_path
    if ckpt is None:
        missing_asset("real checkpoint not found — run scripts/prepare_assets.py or set DD_CHECKPOINT_PATH")

    ref_cfg = DiffusionDriveConfig(plan_anchor_path=model_config.plan_anchor_path, latent=False)
    ref_model = load_model(ckpt, ref_cfg, device=torch.device("cpu")).eval()

    ttnn_model = TtnnDiffusionDriveModel(ref_model, model_config, device)
    ttnn_model.build_all(device)
    features = _features()

    if site == "backbone":
        # TtnnTransfuserBackbone.capture_backbone_trace → _traced → _run_loop_dev
        _arm_failure(monkeypatch, ttnn_model._model._backbone._ttnn, "_run_loop_dev")
    else:
        # TtnnPerceptionForward.capture_trace → _traced → _forward_dev. Reached only
        # after the backbone trace captures cleanly, so this also covers compile()'s
        # own release_backbone_trace unwind.
        _arm_failure(monkeypatch, ttnn_model._perception, "_forward_dev")

    with expect_error(RuntimeError, _INJECTED):
        ttnn_model.compile(features)
    assert not ttnn_model._compiled, "compile() reported success after the captured body raised"

    # Drop the injection before probing, so the probe measures the device and not
    # the monkeypatch.
    monkeypatch.undo()

    _assert_device_usable(device, ttnn_model, features)
