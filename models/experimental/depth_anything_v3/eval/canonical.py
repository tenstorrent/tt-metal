# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Loader for the canonical Depth-Anything-3 metric model.

Bypasses `depth_anything_3.api` (which transitively requires moviepy/gsplat/etc.)
and instantiates the underlying `DepthAnything3Net` from the registry config,
loading weights directly from the standalone `DA3Metric-Large` HF checkpoint.

Returns the RAW DPT depth (post-`exp` activation, pre-sky-post-processing) so
that comparisons against our own implementation (which has no sky head) are
apples-to-apples. The canonical full pipeline ALSO clips sky regions to the
non-sky 99th-percentile depth — that step is intentionally skipped here."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from models.experimental.depth_anything_v3.eval.runner import preprocess


_CANONICAL_REPO_SRC = Path("/home/ttuser/experiments/da3/Depth-Anything-3/src")
_DA3_METRIC_WEIGHTS = Path(
    "/home/ttuser/.cache/huggingface/hub/"
    "models--depth-anything--DA3Metric-Large/snapshots/"
    "4010e39f3634a45bc60553321fb49fb760bd594e/model.safetensors"
)


_CANON_MODEL = None


def _ensure_da3_on_path() -> None:
    if str(_CANONICAL_REPO_SRC) not in sys.path:
        sys.path.insert(0, str(_CANONICAL_REPO_SRC))


def build_canonical_metric() -> torch.nn.Module:
    """Construct DA3Metric-Large with weights loaded. Cached after first call."""
    global _CANON_MODEL
    if _CANON_MODEL is not None:
        return _CANON_MODEL
    _ensure_da3_on_path()
    import safetensors.torch
    from depth_anything_3.cfg import create_object, load_config
    from depth_anything_3.registry import MODEL_REGISTRY

    cfg = load_config(MODEL_REGISTRY["da3metric-large"])
    m = create_object(cfg).eval()
    sd = {
        k[len("model."):]: v
        for k, v in safetensors.torch.load_file(str(_DA3_METRIC_WEIGHTS)).items()
        if k.startswith("model.")
    }
    m.load_state_dict(sd, strict=True)
    _CANON_MODEL = m
    return m


def canonical_predict(rgb_uint8: np.ndarray) -> np.ndarray:
    """Run the canonical DA3-Metric on an RGB image, returning a depth map at
    the input's native resolution (in metres)."""
    m = build_canonical_metric()
    x = preprocess(rgb_uint8)             # (1, 3, 518, 518), normalised
    x = x.unsqueeze(1)                    # (1, S=1, 3, 518, 518) — DA3 input convention
    with torch.inference_mode():
        feats_tup, _ = m.backbone(x)
        depth = m.head(feats_tup, 518, 518, patch_start_idx=0)["depth"]  # (1, 1, H, W)
    depth = F.interpolate(depth, size=rgb_uint8.shape[:2], mode="bilinear", align_corners=False)
    return depth.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
