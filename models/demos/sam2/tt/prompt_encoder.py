# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Prompt Encoder for TTNN SAM2 (sam2-hiera-tiny Image Mode).
Embeds sparse point coordinates into feature space via ttnn.linear projection.
Architecture follows qwen3_vl verified patterns."""

from typing import Dict, Optional, Any
import torch
import ttnn


class TtnnSam2PromptEncoder:
    """TTNN native Prompt Encoder for sparse point coordinates.
    Projects (x, y) coordinates -> embed_dim space using ttnn.linear."""

    def __init__(
        self,
        device: ttnn.Device,
        parameters: Dict[str, Any],
        embed_dim: int = 256,
    ):
        self.device = device
        self.embed_dim = embed_dim

        # Linear projection weight: input 2 (x,y) -> output embed_dim
        pw = parameters.get("point_proj_weight", torch.randn(embed_dim, 2))
        pb = parameters.get("point_proj_bias", torch.randn(embed_dim))

        self.point_proj_weight = ttnn.from_torch(
            pw.T.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.point_proj_bias = ttnn.from_torch(
            pb,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    def forward(
        self,
        points: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Embed point coordinates into feature space.

        Args:
            points: [B, N, 2] normalized (x, y) coordinates

        Returns:
            dict with 'sparse_embeddings': [B, N, embed_dim] ttnn tensor
        """
        if points is None:
            return {"sparse_embeddings": None}

        # Move points to device
        tt_pts = ttnn.from_torch(
            points,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

        # Linear projection: [B, N, 2] -> [B, N, embed_dim]
        # Weight is [embed_dim, 2], so no transpose needed
        sparse_embeddings = ttnn.linear(
            tt_pts,
            self.point_proj_weight,
            bias=self.point_proj_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(tt_pts)

        return {"sparse_embeddings": sparse_embeddings}
