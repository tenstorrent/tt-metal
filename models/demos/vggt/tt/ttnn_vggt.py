"""ttnn port scaffolding for VGGT.

Stage 0 (this file): passthrough that opens a ttnn device and runs the
torch reference. This gives us a working baseline benchmark to iterate
against. Later stages will replace individual operators with ttnn kernels,
commit by commit, measured against PCC >= 0.99.
"""
from __future__ import annotations

from typing import Any, Dict

import torch


_CACHED_MODEL = None


def _get_model():
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        from models.demos.vggt.reference.torch_vggt import load_vggt
        _CACHED_MODEL = load_vggt(eval_mode=True)
    return _CACHED_MODEL


def vggt_forward(images: torch.Tensor, device: Any = None, query_points: torch.Tensor = None) -> Dict[str, torch.Tensor]:
    """Run VGGT on the given images. Currently CPU-only passthrough.

    Args:
        images: [B, S, 3, H, W] in [0, 1].
        device: ttnn device handle (unused in stage 0).
        query_points: optional [B, N, 2] tracking queries.

    Returns:
        dict matching the VGGT forward output, minus the track branch
        unless query_points is provided.
    """
    model = _get_model()
    with torch.no_grad():
        return model(images, query_points=query_points)
