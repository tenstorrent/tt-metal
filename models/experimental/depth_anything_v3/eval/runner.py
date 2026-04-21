# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Adapters that take a KITTI RGB image (uint8 HxWx3, 0..255) and produce a
metric depth map (float32 HxW, in metres) using either:

  - the CPU fp32 reference (`models...reference.dinov2_l_dpt.DA3Metric`), or
  - the chip pipeline (`models...tt.ttnn_da3_metric.run`).

Both share the standard DA3 input pipeline: ImageNet normalize, resize to a
square model-input size that is a multiple of patch_size, exp-activate the
raw head output, and bilinearly upsample back to the GT resolution."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from models.experimental.depth_anything_v3.reference.dinov2_l_dpt import (
    DEFAULT_INPUT_SIZE, build_da3_metric,
)


_IMAGENET_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def preprocess(rgb_uint8: np.ndarray, *, side: int = DEFAULT_INPUT_SIZE) -> torch.Tensor:
    """RGB uint8 (H, W, 3) -> normalised float32 tensor (1, 3, side, side)."""
    t = torch.from_numpy(rgb_uint8).permute(2, 0, 1).float() / 255.0
    t = _IMAGENET_NORM(t).unsqueeze(0)
    if t.shape[-2:] != (side, side):
        t = F.interpolate(t, size=(side, side), mode="bilinear", align_corners=False)
    return t


def postprocess_depth(raw_logits: torch.Tensor, target_hw: tuple[int, int]) -> np.ndarray:
    """Apply DA3's `exp` activation, upsample to target HxW, return numpy meters."""
    if raw_logits.dim() == 3:
        raw_logits = raw_logits.unsqueeze(1)
    depth = torch.exp(raw_logits.float())
    depth = F.interpolate(depth, size=target_hw, mode="bilinear", align_corners=False)
    return depth.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)


_REF_MODEL = None


def reference_predict(rgb_uint8: np.ndarray) -> np.ndarray:
    """CPU fp32 reference. Returns a (H, W) float32 depth map in metres."""
    global _REF_MODEL
    if _REF_MODEL is None:
        _REF_MODEL = build_da3_metric(load_weights=True, img_size=DEFAULT_INPUT_SIZE).eval()
    x = preprocess(rgb_uint8)
    with torch.inference_mode():
        raw = _REF_MODEL(x)
    return postprocess_depth(raw, target_hw=rgb_uint8.shape[:2])


def chip_predict(rgb_uint8: np.ndarray) -> np.ndarray:
    """Chip-backbone + bf16 CPU head. Returns metric depth at native resolution."""
    from models.experimental.depth_anything_v3.tt import ttnn_da3_metric as M
    x = preprocess(rgb_uint8)
    raw, _peak_dram = M.run(x)
    return postprocess_depth(raw, target_hw=rgb_uint8.shape[:2])
