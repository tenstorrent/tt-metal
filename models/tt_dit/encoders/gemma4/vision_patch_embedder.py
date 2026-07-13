# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma 4 vision patch embedder.

Mirrors ``Gemma4VisionPatchEmbedder`` from ``transformers.models.gemma4.modeling_gemma4``:

    1. Normalize pixels to [-1, 1]: ``2 * (px - 0.5)``
    2. Project flattened patches ``3 * patch_size**2 → hidden_size`` (input_proj, no bias)
    3. Add 2-D position embeddings: separate x- and y-tables looked up and summed
    4. Zero out positions marked as padding (pixel_position_ids == -1)
"""

from __future__ import annotations

import torch

import ttnn

from ...layers.linear import Linear
from ...layers.module import Module, Parameter
from ...parallel.config import DiTParallelConfig

TILE = ttnn.TILE_SIZE


class Gemma4VisionPatchEmbedder(Module):
    """Patch embedder for the vision encoder.

    The position embedding table is held replicated (no TP) — at ~47 MB bf16
    (2 × 10240 × 1152) it fits per-device comfortably.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        patch_size: int,
        position_embedding_size: int,
        mesh_device: ttnn.MeshDevice,
        ccl_manager,
        parallel_config: DiTParallelConfig,
    ) -> None:
        super().__init__()
        assert hidden_size % TILE == 0

        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.position_embedding_size = position_embedding_size
        self.mesh_device = mesh_device

        # Input projection: 3 * patch_size^2 → hidden_size. Kept replicated — this is a
        # one-shot pre-attention op and the projection is small (768 × 1152 ≈ 884K params).
        self.input_proj = Linear(
            3 * patch_size**2,
            hidden_size,
            bias=False,
            mesh_device=mesh_device,
        )
        self.parallel_config = parallel_config

        # Position embedding table: replicated [2, position_embedding_size, hidden_size].
        self.position_embedding_table = Parameter(
            total_shape=[2, position_embedding_size, hidden_size],
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def forward(
        self,
        pixel_values: ttnn.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> ttnn.Tensor:
        """
        Args:
            pixel_values:        ttnn ``[B, num_patches, 3 * patch_size**2]`` bf16 already-pixel-normalized
                                  (caller is responsible for the ``2 * (px - 0.5)`` step — TT-side here is just
                                  the projection + position embeddings).

                                  Note: the HF reference does the ``2*(p-0.5)`` inside this module on
                                  raw [0,1] pixel inputs. For TT we expect the caller to have done it
                                  during pixel preprocessing on host.

            pixel_position_ids:  torch ``[B, num_patches, 2]`` long tensor, ``-1`` for padding.
            padding_positions:   torch ``[B, num_patches]`` bool tensor, ``True`` for padding patches.

        Returns:
            ttnn ``[B, num_patches, hidden_size]`` bf16, with padding positions zeroed.
        """
        # 1. Input projection (replicated).
        h = self.input_proj(pixel_values)

        # 2. Build position embeddings on host (small in absolute terms; saves the
        #    complexity of doing a 2-table ttnn.embedding gather on device for now).
        clamped = pixel_position_ids.clamp(min=0)  # (B, P, 2)
        # The position_embedding_table is stored on device. Read it back as torch to do
        # the lookup, then upload the result. (Optimization opportunity: on-device gather
        # via ttnn.embedding with a 2-table slice.)
        pet = ttnn.to_torch(self.position_embedding_table.data).float()  # (2, P, hidden)
        x_emb = pet[0][clamped[..., 0]]  # (B, P, hidden)
        y_emb = pet[1][clamped[..., 1]]
        pos_emb = x_emb + y_emb
        pos_emb = pos_emb.masked_fill(padding_positions.unsqueeze(-1), 0.0)
        # Upload as bf16, replicated on the mesh.
        tt_pos_emb = ttnn.from_torch(
            pos_emb.to(torch.bfloat16),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
        )

        # 3. Add (both replicated).
        out = ttnn.add(h, tt_pos_emb)
        ttnn.deallocate(tt_pos_emb)
        return out
