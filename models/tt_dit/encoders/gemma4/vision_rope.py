# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma 4 vision rotary embedding (2-D multidim).

Mirrors ``Gemma4VisionRotaryEmbedding`` + ``apply_multidimensional_rope`` from
``transformers.models.gemma4.modeling_gemma4``: each patch has 2D position ids
``(x, y)``; the per-head ``head_dim`` is split into two chunks (one per spatial
dim), and standard 1-D RoPE is applied to each chunk independently with
chunk-specific cos/sin.

For Option A (head_dim padded from 72 → 96 for tile alignment), the layout of
the *padded* cos/sin per patch is::

    [ cos_x   :  36 channels  ][ cos_y  :  36 channels  ][ pad : 24 channels (=1) ]
    [ sin_x   :  36           ][ sin_y  :  36           ][ pad : 24       (=0) ]

Storage convention is tt_dit's half-dim (we keep the *unique* 18 entries per
spatial-dim chunk, not the duplicated 36; the attention's ``_apply_rope`` uses
the half-dim formula on each chunk's halves). Layout of our returned half-dim
cos/sin (shape ``[B, 1, num_patches, head_dim_padded/2]``)::

    [ cos_x_half : 18 ][ cos_y_half : 18 ][ pad : (head_dim_padded/2 - 36) (=1) ]
"""

from __future__ import annotations

import torch

import ttnn

from ...layers.module import Module


class Gemma4VisionRotaryEmbedding(Module):
    """Multidim RoPE for the Gemma 4 vision encoder.

    Args:
        head_dim:           the *unpadded* vision head dim (72 for the released config).
        head_dim_padded:    the padded vision head dim (e.g. 96 to tile-align). Equals
                            ``head_dim`` if no padding required.
        position_embedding_size: max value of any (x, y) coordinate (10240 default).
        rope_theta:         RoPE base frequency (100.0 default for vision).
    """

    def __init__(
        self,
        *,
        head_dim: int,
        head_dim_padded: int,
        position_embedding_size: int,
        rope_theta: float,
        mesh_device: ttnn.MeshDevice,
    ) -> None:
        super().__init__()
        assert head_dim % 2 == 0
        assert head_dim_padded >= head_dim
        assert head_dim_padded % 2 == 0
        # Each spatial dim consumes ``head_dim / 2`` of the unpadded head_dim. The
        # *unique* freq count per dim is ``head_dim / 4``.
        self.head_dim = head_dim
        self.head_dim_padded = head_dim_padded
        self.spatial_dim = head_dim // 2  # 36 for head_dim=72
        self.unique_per_dim = head_dim // 4  # 18 for head_dim=72
        assert self.spatial_dim % 2 == 0
        self.mesh_device = mesh_device

        # inv_freq follows the HF reference:
        #   inv_freq = 1 / theta^(2i / spatial_dim) for i in [0, spatial_dim/2)
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, self.spatial_dim, 2, dtype=torch.int64).float() / self.spatial_dim)
        )
        # Precompute per-spatial-dim tables once on host. Shape: [position_embedding_size, unique_per_dim].
        positions = torch.arange(position_embedding_size, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)  # (P, unique_per_dim)
        self._cos_axis = freqs.cos()  # (P, unique_per_dim)
        self._sin_axis = freqs.sin()  # (P, unique_per_dim)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Gemma4VisionRotaryEmbedding is used via get_cos_sin(pixel_position_ids).")

    def get_cos_sin(self, pixel_position_ids: torch.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Args:
            pixel_position_ids: ``[B, num_patches, 2]`` long tensor. The last dim is (x, y);
                                 padding patches use (-1, -1) which we clamp to 0 (caller is
                                 responsible for masking padded positions downstream).

        Returns:
            (cos, sin) ttnn tensors of shape ``[B, 1, num_patches, head_dim_padded / 2]`` (bf16).
            Layout per patch (half-dim):
                [ cos_x_half : unique_per_dim ][ cos_y_half : unique_per_dim ][ pad : 1's ]
        """
        clamped = pixel_position_ids.clamp(min=0)
        x_ids = clamped[..., 0]  # (B, P)
        y_ids = clamped[..., 1]
        cos_x = self._cos_axis[x_ids]  # (B, P, unique_per_dim)
        sin_x = self._sin_axis[x_ids]
        cos_y = self._cos_axis[y_ids]
        sin_y = self._sin_axis[y_ids]

        half_padded = self.head_dim_padded // 2
        pad_len = half_padded - 2 * self.unique_per_dim
        assert pad_len >= 0, "head_dim_padded too small for the spatial-dim layout"

        B, P = pixel_position_ids.shape[0], pixel_position_ids.shape[1]
        if pad_len > 0:
            cos_pad = torch.ones(B, P, pad_len, dtype=torch.float32)
            sin_pad = torch.zeros(B, P, pad_len, dtype=torch.float32)
            cos = torch.cat([cos_x, cos_y, cos_pad], dim=-1)
            sin = torch.cat([sin_x, sin_y, sin_pad], dim=-1)
        else:
            cos = torch.cat([cos_x, cos_y], dim=-1)
            sin = torch.cat([sin_x, sin_y], dim=-1)

        # (B, P, half_padded) → (B, 1, P, half_padded) for broadcast over heads.
        cos = cos.unsqueeze(1).to(torch.bfloat16)
        sin = sin.unsqueeze(1).to(torch.bfloat16)
        tt_cos = ttnn.from_torch(cos, device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        tt_sin = ttnn.from_torch(sin, device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        return tt_cos, tt_sin
