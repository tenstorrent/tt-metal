# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""KITTI Eigen-split test loader for DA3-Metric evaluation.

Pairs each line of `eigen_test_files.txt` (697 entries) with its corresponding
ground-truth depth map from `gt_depths.npy`. Returns RGB images at native
KITTI resolution (375 x 1242) plus the dense GT depth (interpolated LiDAR)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

import cv2
import numpy as np


KITTI_ROOT = Path("/home/ttuser/experiments/da3/eval_data/kitti_test")
EIGEN_LIST = Path("/home/ttuser/experiments/da3/eval_data/splits/eigen_test_files.txt")
GT_DEPTHS = Path("/home/ttuser/experiments/da3/eval_data/gt_depths.npy")
SIDE_TO_DIR = {"l": "image_02", "r": "image_03"}


@dataclass
class KittiSample:
    index: int
    drive: str             # e.g. "2011_09_26/2011_09_26_drive_0002_sync"
    frame: str             # e.g. "0000000069"
    side: str              # "l" or "r"
    rgb: np.ndarray        # (H, W, 3) uint8 RGB
    gt_depth: np.ndarray   # (H, W) float32, depth in metres; 0 means invalid


def _resolve_rgb_path(drive: str, frame: str, side: str) -> Path:
    drive_basename = drive.split("/")[-1]
    p = KITTI_ROOT / drive_basename / SIDE_TO_DIR[side] / "data" / f"{frame}.png"
    if p.exists():
        return p
    return KITTI_ROOT / drive / SIDE_TO_DIR[side] / "data" / f"{frame}.png"


def load_kitti_eigen_test(limit: int | None = None) -> Iterator[KittiSample]:
    """Yield Eigen-split test samples in canonical order (matches `gt_depths.npy`)."""
    with open(EIGEN_LIST) as f:
        lines = [ln.strip().split() for ln in f if ln.strip()]
    gt_all = np.load(GT_DEPTHS, allow_pickle=True)
    assert len(lines) == len(gt_all), (
        f"split list ({len(lines)}) and GT npy ({len(gt_all)}) mismatch"
    )
    if limit is not None:
        lines = lines[:limit]
    for i, (drive, frame, side) in enumerate(lines):
        rgb_bgr = cv2.imread(str(_resolve_rgb_path(drive, frame, side)))
        if rgb_bgr is None:
            raise FileNotFoundError(f"Missing RGB for {drive}/{frame}/{side}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        yield KittiSample(
            index=i, drive=drive, frame=frame, side=side, rgb=rgb,
            gt_depth=np.asarray(gt_all[i], dtype=np.float32),
        )


def list_kitti_eigen() -> List[tuple[str, str, str]]:
    with open(EIGEN_LIST) as f:
        return [tuple(ln.strip().split()) for ln in f if ln.strip()]
