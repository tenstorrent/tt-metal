#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Bird's-eye-view visualisation of a DiffusionDrive trajectory prediction.

Renders the model's planned ego trajectory (the ``trajectory`` output, B x 8 x 3 =
x, y, heading) in a top-down ego frame, alongside the K=20 anchor modes shaded by
their predicted ``scores``, so a forward pass can be eyeballed for plausibility.
This is the visual half of the bounty's "output is verifiable (visualization +
metric checks)" requirement; the metric half is the PCC gates + NavSim PDM score.

Two modes:

  # Real: run the on-device TTNN model, plot its trajectory (needs a device + assets)
  python models/demos/diffusion_drive/scripts/visualize_trajectory.py \
      --checkpoint "$DD_CHECKPOINT_PATH" --anchors "$DD_ANCHOR_PATH" -o traj.png

  # Demo: no device — render a synthetic trajectory to sanity-check the plot itself
  python models/demos/diffusion_drive/scripts/visualize_trajectory.py --demo -o demo.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np


def plot_bev(
    trajectory: np.ndarray,  # (T, 3): x, y, heading  (metres, ego frame)
    scores: Optional[np.ndarray] = None,  # (K,)
    anchors: Optional[np.ndarray] = None,  # (K, T, 2): x, y per anchor mode
    out_path: str = "trajectory.png",
    title: str = "DiffusionDrive planned trajectory (BEV, ego frame)",
) -> str:
    """Render a top-down plot and save it. Pure (no device / no model) so it is
    unit-testable on synthetic arrays. Returns the written path."""
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 9))

    # Anchor modes (faint), shaded by score if provided.
    if anchors is not None:
        if scores is not None and len(scores) == len(anchors):
            s = np.asarray(scores, dtype=np.float64)
            s = (s - s.min()) / (s.ptp() + 1e-9)  # normalise to [0, 1] for alpha
        else:
            s = np.full(len(anchors), 0.3)
        for k, anc in enumerate(anchors):
            ax.plot(anc[:, 0], anc[:, 1], "-", color="steelblue", alpha=0.15 + 0.5 * s[k], linewidth=1.0)
        best = int(np.argmax(scores)) if scores is not None else 0
        ax.plot(
            anchors[best][:, 0],
            anchors[best][:, 1],
            "o-",
            color="tab:orange",
            markersize=4,
            linewidth=1.5,
            label=f"top-scored anchor (#{best})",
        )

    # Predicted trajectory with heading arrows.
    xs, ys = trajectory[:, 0], trajectory[:, 1]
    ax.plot(xs, ys, "o-", color="crimson", markersize=6, linewidth=2.5, label="predicted trajectory", zorder=5)
    if trajectory.shape[1] >= 3:
        for x, y, h in trajectory:
            ax.arrow(x, y, 0.6 * np.cos(h), 0.6 * np.sin(h), head_width=0.25, color="crimson", alpha=0.7, zorder=6)

    # Ego marker at origin.
    ax.plot(0, 0, "s", color="black", markersize=10, label="ego (t=0)", zorder=7)

    ax.set_xlabel("x — lateral (m)")
    ax.set_ylabel("y — longitudinal (m)")
    ax.set_title(title)
    ax.axhline(0, color="gray", lw=0.5, alpha=0.4)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.4)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _demo(out_path: str) -> str:
    """Synthetic gentle right curve + fanned anchors — exercises the plot with no device."""
    rng = np.random.default_rng(0)
    t = np.linspace(0.5, 4.0, 8)
    traj = np.stack([0.4 * t, 4.0 * t, np.full_like(t, np.deg2rad(6.0))], axis=1)  # (8, 3)
    anchors = np.stack([np.stack([(0.4 + 0.5 * (k / 20 - 0.5)) * t, 4.0 * t], axis=1) for k in range(20)])  # (20, 8, 2)
    scores = rng.random(20)
    scores[9] = 1.5  # a clear winner
    return plot_bev(traj, scores, anchors, out_path, title="DiffusionDrive (synthetic demo — no device)")


def _run_model(checkpoint: Optional[str], anchors_path: Optional[str], out_path: str) -> str:
    import torch

    import ttnn
    from models.demos.diffusion_drive.reference.model import DiffusionDriveConfig, DiffusionDriveModel, load_model
    from models.demos.diffusion_drive.tt.config import ModelConfig
    from models.demos.diffusion_drive.tt.ttnn_diffusion_drive import TtnnDiffusionDriveModel

    latent = checkpoint is None
    ref_cfg = DiffusionDriveConfig(plan_anchor_path=anchors_path, latent=latent)
    ref_model = (
        DiffusionDriveModel(ref_cfg) if latent else load_model(checkpoint, ref_cfg, torch.device("cpu"))
    ).eval()

    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        ttnn_model = TtnnDiffusionDriveModel(ref_model, ModelConfig(plan_anchor_path=anchors_path), device)
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
        torch.manual_seed(1234)  # pin DDIM noise (DD-5)
        out = ttnn_model(features)
    finally:
        ttnn.close_device(device)

    traj = out["trajectory"][0].detach().cpu().numpy()  # (8, 3)
    scores = out["scores"][0].detach().cpu().numpy()  # (20,)
    anc = ref_model._trajectory_head.plan_anchor.detach().cpu().numpy() if anchors_path else None
    return plot_bev(traj, scores, anc, out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualise a DiffusionDrive trajectory in BEV")
    ap.add_argument("--demo", action="store_true", help="render synthetic data (no device / no assets)")
    ap.add_argument("--checkpoint", default=os.environ.get("DD_CHECKPOINT_PATH"))
    ap.add_argument("--anchors", default=os.environ.get("DD_ANCHOR_PATH"))
    ap.add_argument("-o", "--out", default="trajectory.png")
    args = ap.parse_args()

    if args.demo:
        print(f"[viz] wrote {_demo(args.out)} (synthetic demo)")
        return
    ckpt = args.checkpoint if args.checkpoint and Path(args.checkpoint).exists() else None
    print(f"[viz] wrote {_run_model(ckpt, args.anchors, args.out)}")


if __name__ == "__main__":
    main()
