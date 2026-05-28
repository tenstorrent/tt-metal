# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Pure 2x2 spatial patch merger.

Reshape+concat that collapses each 2x2 block of vision tokens into a
single token, growing the channel dim from 1152 to 1152*4 = 4608.

No LayerNorm and no Linear inside the merger — those live in the
multi-modal projector (see projector.py).

Reference: `patch_merger` function in modeling_kimi_vl.py.
"""
from __future__ import annotations


def patch_merger(x, grid_hws, merge_kernel_size=(2, 2)):
    """
    Args:
        x: vision tokens, shape [sum(H_i * W_i), 1152] (on device).
        grid_hws: per-image (H, W); used to reshape before merging.
        merge_kernel_size: spatial merge factor; MoonViT default (2, 2).

    Returns:
        Merged tokens, shape [sum((H_i/2) * (W_i/2)), 1152 * 4].
    """
    raise NotImplementedError("Phase 1 — patch_merger")
