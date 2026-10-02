# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image-Edit on the shared tt_dit infra.

Built on top of the base Qwen-Image port (``pipelines/qwenimage``): the edit variant reuses the same
sequence-parallel ``QwenImageTransformer``, VAE and Qwen2.5-VL encoder, and adds the edit-specific
pieces (condition-image VAE encode, VL image conditioning, latent concatenation on the token dim, and
true-CFG with norm rescale). Target Galaxy layout: TP=8 x SP=4 on the 4x8 mesh.
"""
from __future__ import annotations

from models.tt_dit.pipelines.qwenimage_edit.pipeline_qwenimage_edit import (
    QwenImageEditPipeline,
    QwenImageEditPipelineConfig,
)

__all__ = ["QwenImageEditPipeline", "QwenImageEditPipelineConfig"]
