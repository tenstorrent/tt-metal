# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standard KITTI Eigen-split depth metrics + Pearson correlation utility.

Follows the convention used in BTS/MonoDepth2/DA1/DA2:
- Garg crop on the bottom-half of the image to focus on driveable area.
- min_depth=1e-3, max_depth=80m (clip predictions before metrics).
- Median scaling so monocular metric depth can be compared with sparse LiDAR
  GT (the model is unitless up to a scale; median scaling = standard practice).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


MIN_DEPTH = 1e-3
MAX_DEPTH = 80.0


def garg_crop_mask(h: int, w: int) -> np.ndarray:
    """Standard Eigen/Garg crop. Trims top, bottom, and small side margins."""
    mask = np.zeros((h, w), dtype=bool)
    y0 = int(0.40810811 * h)
    y1 = int(0.99189189 * h)
    x0 = int(0.03594771 * w)
    x1 = int(0.96405229 * w)
    mask[y0:y1, x0:x1] = True
    return mask


@dataclass
class DepthMetrics:
    abs_rel: float
    sq_rel: float
    rmse: float
    rmse_log: float
    delta1: float
    delta2: float
    delta3: float

    def as_dict(self) -> dict[str, float]:
        return {
            "abs_rel": self.abs_rel,
            "sq_rel": self.sq_rel,
            "rmse": self.rmse,
            "rmse_log": self.rmse_log,
            "delta1": self.delta1,
            "delta2": self.delta2,
            "delta3": self.delta3,
        }


def compute_depth_metrics(
    pred: np.ndarray, gt: np.ndarray, *, use_garg_crop: bool = True,
    median_scale: bool = True,
) -> DepthMetrics:
    """Compute the seven canonical KITTI depth metrics.

    `pred` and `gt` must have the same (H, W) shape and be in metres. Pixels
    where gt <= 0 are ignored. `pred` is clipped to [MIN_DEPTH, MAX_DEPTH]."""
    assert pred.shape == gt.shape, f"pred {pred.shape} vs gt {gt.shape}"
    valid = gt > MIN_DEPTH
    if use_garg_crop:
        valid &= garg_crop_mask(*gt.shape)
    valid &= gt < MAX_DEPTH
    pred = np.clip(pred.astype(np.float64), MIN_DEPTH, MAX_DEPTH)
    gt = gt.astype(np.float64)

    pred_v = pred[valid]
    gt_v = gt[valid]
    if median_scale and pred_v.size > 0:
        pred_v = pred_v * (np.median(gt_v) / np.median(pred_v))
        pred_v = np.clip(pred_v, MIN_DEPTH, MAX_DEPTH)

    thresh = np.maximum(gt_v / pred_v, pred_v / gt_v)
    d1 = float((thresh < 1.25).mean())
    d2 = float((thresh < 1.25**2).mean())
    d3 = float((thresh < 1.25**3).mean())
    rmse = float(np.sqrt(((gt_v - pred_v) ** 2).mean()))
    rmse_log = float(np.sqrt(((np.log(gt_v) - np.log(pred_v)) ** 2).mean()))
    abs_rel = float(np.mean(np.abs(gt_v - pred_v) / gt_v))
    sq_rel = float(np.mean(((gt_v - pred_v) ** 2) / gt_v))
    return DepthMetrics(abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3)


def pearson_pcc(a, b) -> float:
    """Pearson correlation coefficient between two flat tensors."""
    if isinstance(a, torch.Tensor):
        a = a.detach().to(torch.float64).flatten().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().to(torch.float64).flatten().numpy()
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())) + 1e-12
    return float((a * b).sum() / denom)


def aggregate(metric_list: list[DepthMetrics]) -> dict[str, float]:
    keys = list(metric_list[0].as_dict().keys())
    out = {}
    for k in keys:
        vals = [m.as_dict()[k] for m in metric_list]
        out[k] = float(np.mean(vals))
    return out
