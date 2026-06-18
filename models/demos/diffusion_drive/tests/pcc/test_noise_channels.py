# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Regression test for the DDIM noise-channel fix (no device needed).

Upstream DiffusionDrive (`transfuser_model_v2.forward_test`) runs the diffusion
on the 2 (x, y) channels only — its `norm_odo` slices an empty heading column,
so `noise = torch.randn(img.shape)` is (B, K, T, 2).  An earlier version of the
in-repo reference padded a zero heading column, making the noise (B, K, T, 3);
because randn fills row-major, that shifts the x/y noise and breaks bit-for-bit
reproduction of the upstream PDMS-88.04 behaviour.

This test fails if that regression is reintroduced.
"""

from __future__ import annotations

import pytest
import torch

from models.demos.diffusion_drive.reference.model import DiffusionDriveConfig, DiffusionDriveModel

_ANCHORS = "models/demos/diffusion_drive/data/kmeans_navsim_traj_20.npy"


def test_diffusion_noise_is_two_channels(monkeypatch) -> None:
    import os

    if not os.path.exists(_ANCHORS):
        pytest.skip("anchor file not found — run scripts/prepare_assets.py first")

    cfg = DiffusionDriveConfig(plan_anchor_path=_ANCHORS, latent=True)
    torch.manual_seed(0)
    model = DiffusionDriveModel(cfg).eval()

    captured = []
    real_randn = torch.randn

    def spy_randn(*args, **kwargs):
        # randn may be called as randn(shape_tuple) or randn(*dims)
        shape = args[0] if (len(args) == 1 and isinstance(args[0], (tuple, torch.Size, list))) else args
        captured.append(tuple(shape))
        return real_randn(*args, **kwargs)

    monkeypatch.setattr(torch, "randn", spy_randn)

    features = {
        "camera_feature": torch.randn(1, 3, 64, 128),
        "lidar_feature": torch.zeros(1, 1, 64, 64),
        "status_feature": torch.zeros(1, 8),
    }
    with torch.no_grad():
        model(features)

    # The DDIM noise draw is the (B, K, T, C) call inside the trajectory head.
    traj_noise = [s for s in captured if len(s) == 4 and s[1:3] == (20, 8)]
    assert traj_noise, f"did not observe the DDIM noise draw; captured={captured}"
    for s in traj_noise:
        assert s[-1] == 2, f"DDIM noise must be 2-channel (got {s}); upstream diffuses x,y only"
