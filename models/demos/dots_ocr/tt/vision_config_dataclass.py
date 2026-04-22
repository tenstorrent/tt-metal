# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Lightweight vision config dataclass for Dots OCR.

Kept in a tiny module so tests and docs can import it without pulling in
``ModelArgs`` / ``tt_transformers`` (which require a full ``ttnn`` install).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DotsVisionConfig:
    """Configuration specific to Dots Vision Transformer."""

    hidden_size: int = 1536
    num_hidden_layers: int = 42
    num_attention_heads: int = 12
    intermediate_size: int = 4224
    patch_size: int = 14
    spatial_merge_size: int = 2
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    init_merger_std: float = 0.02
    post_norm: bool = True  # Dots uses post-norm
    num_channels: int = 3
    temporal_patch_size: int = 1
