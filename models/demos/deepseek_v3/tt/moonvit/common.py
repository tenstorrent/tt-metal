# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Host-side helpers for MoonViT preprocessing.

Builds the cu_seqlens / grid_hws inputs that NaViT-style packed attention
expects, and converts HF KimiVLImageProcessor output into the format the
MoonViT modules consume.
"""
from __future__ import annotations

import torch


def build_cu_seqlens(grid_hws: torch.Tensor) -> torch.Tensor:
    """
    Convert a per-image (H, W) grid into the cumulative sequence-length
    offsets needed by ttnn.transformer.windowed_scaled_dot_product_attention.

    Args:
        grid_hws: int64 tensor of shape [num_images, 2] with (H, W) per image.

    Returns:
        uint32 tensor of shape [num_images + 1]; prefix sum of H*W with a
        leading zero.
    """
    raise NotImplementedError("Phase 1 — build_cu_seqlens")
