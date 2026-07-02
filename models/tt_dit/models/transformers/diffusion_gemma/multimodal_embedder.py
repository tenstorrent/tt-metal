# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Projects vision-tower soft tokens into the text encoder's hidden space.

Mirrors ``Gemma4MultimodalEmbedder.forward`` (which inherits ``Gemma3nMultimodalEmbedder``)::

    embs_normed = embedding_pre_projection_norm(inputs_embeds)   # RMSNorm, with_scale=False
    return embedding_projection(embs_normed)                       # Linear, multimodal_hidden → text_hidden
"""

from __future__ import annotations

import ttnn

from ....layers.linear import Linear
from ....layers.module import Module
from ....layers.normalization import RMSNorm


class DiffusionGemmaMultimodalEmbedder(Module):
    """Vision soft tokens → text-hidden space."""

    def __init__(
        self,
        *,
        multimodal_hidden_size: int,
        text_hidden_size: int,
        rms_norm_eps: float,
        mesh_device: ttnn.MeshDevice,
    ) -> None:
        super().__init__()
        self.embedding_pre_projection_norm = RMSNorm(
            embedding_dim=multimodal_hidden_size,
            norm_eps=rms_norm_eps,
            norm_elementwise_affine=False,
            bias=False,
            mesh_device=mesh_device,
        )
        self.embedding_projection = Linear(
            multimodal_hidden_size,
            text_hidden_size,
            bias=False,
            mesh_device=mesh_device,
        )

    def forward(self, inputs_embeds: ttnn.Tensor) -> ttnn.Tensor:
        normed = self.embedding_pre_projection_norm(inputs_embeds)
        out = self.embedding_projection(normed)
        ttnn.deallocate(normed)
        return out
