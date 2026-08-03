# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL vision tower.

The tower turns raw patches into `out_hidden_size` tokens for the text decoder, and additionally
exports intermediate features from `deepstack_visual_indexes` that the decoder adds into its first few
layers. MiniMax-H3 needs it for the `fl2va` and `ref2va` tasks; `t2va` passes no pixels and never
reaches it.

Currently ported: the input stage (patch embedding and interpolated position embedding). The blocks,
attention and mergers are still to come.

Two things about this tower differ from the text encoder in `model_qwen3vl.py`:

- `head_dim` is `1152 // 16 == 72`, which is not tile-aligned. `ttnn` SDPA rejects it outright
  (`TT_FATAL logical_shape[3] == legacy_shape[3]`), so the projections must be padded to 96 and the
  softmax `scale` passed explicitly as `72 ** -0.5` -- the padding zeros must not change the
  temperature.
- Norms are `LayerNorm`, attention is plain MHA with biases, and there is no QK-norm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import ttnn

from ...layers.linear import Linear
from ...layers.module import Module

if TYPE_CHECKING:
    pass


def vision_bilinear_indices_and_weights(
    grid_thw: torch.Tensor,
    *,
    num_grid_per_side: int,
    spatial_merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(4, total_patches)` corner indices into the position table, and their bilinear weights.

    The table is a fixed `num_grid_per_side ** 2` grid (48x48 for MiniMax-H3's conditioner), so any
    canvas that is not exactly that shape is interpolated onto it. That is the common case rather than
    an edge case: a 768x1344 keyframe is 48x84 patches against a 48x48 table.

    The `reorder` step is what puts patches into `spatial_merge_size` blocks, matching the order the
    merger later consumes them in.
    """
    side = num_grid_per_side
    merge_size = spatial_merge_size
    idx_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    weight_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]

    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)

        h_grid = torch.linspace(0, side - 1, h)
        w_grid = torch.linspace(0, side - 1, w)
        h_floor = h_grid.int()
        w_floor = w_grid.int()
        h_ceil = (h_floor + 1).clamp(max=side - 1)
        w_ceil = (w_floor + 1).clamp(max=side - 1)
        h_frac = h_grid - h_floor
        w_frac = w_grid - w_floor
        h_floor_offset = h_floor * side
        h_ceil_offset = h_ceil * side

        corner_indices = [
            (h_floor_offset[:, None] + w_floor[None, :]).flatten(),
            (h_floor_offset[:, None] + w_ceil[None, :]).flatten(),
            (h_ceil_offset[:, None] + w_floor[None, :]).flatten(),
            (h_ceil_offset[:, None] + w_ceil[None, :]).flatten(),
        ]
        corner_weights = [
            ((1 - h_frac)[:, None] * (1 - w_frac)[None, :]).flatten(),
            ((1 - h_frac)[:, None] * w_frac[None, :]).flatten(),
            (h_frac[:, None] * (1 - w_frac)[None, :]).flatten(),
            (h_frac[:, None] * w_frac[None, :]).flatten(),
        ]

        h_idx = torch.arange(h).view(h // merge_size, merge_size)
        w_idx = torch.arange(w).view(w // merge_size, merge_size)
        reorder = (h_idx[:, :, None, None] * w + w_idx[None, None, :, :]).transpose(1, 2).flatten().repeat(t)

        for i in range(4):
            idx_parts[i].append(corner_indices[i][reorder])
            weight_parts[i].append(corner_weights[i][reorder])

    return (
        torch.stack([torch.cat(p) for p in idx_parts]),
        torch.stack([torch.cat(p) for p in weight_parts]),
    )


def vision_pos_embeds(
    pos_embed_weight: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    num_grid_per_side: int,
    spatial_merge_size: int,
) -> torch.Tensor:
    """The `(total_patches, hidden_size)` position embedding to add after the patch embedding.

    Computed on the host and uploaded, the way [`create_rope_tensors`] supplies the decoder's rotary
    tensors: this is a pure function of `grid_thw` and the position table -- no pixels enter -- so
    there is nothing for the device to do that host arithmetic does not already settle.
    """
    indices, weights = vision_bilinear_indices_and_weights(
        grid_thw, num_grid_per_side=num_grid_per_side, spatial_merge_size=spatial_merge_size
    )
    return (pos_embed_weight[indices] * weights[:, :, None]).sum(0)


class Qwen3VlVisionPatchEmbed(Module):
    """Patch embedding of the vision tower.

    The reference is a `Conv3d` whose kernel equals its stride over patches that the processor has
    already flattened, which makes it a plain matmul: each `(in_channels, temporal_patch_size,
    patch_size, patch_size)` patch is one independent row. Verified equal to the reference `Conv3d` to
    3.8e-6 in fp32 (accumulation order only), which is far below bf16 resolution. So this is a
    `Linear` and `_prepare_torch_state` flattens the convolution weight into it.
    """

    def __init__(
        self,
        *,
        in_channels: int = 3,
        hidden_size: int = 1152,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        mesh_device: ttnn.MeshDevice,
    ) -> None:
        super().__init__()
        self.patch_dim = in_channels * temporal_patch_size * patch_size * patch_size
        self.hidden_size = hidden_size
        self.proj = Linear(self.patch_dim, hidden_size, bias=True, mesh_device=mesh_device)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # `proj.weight` arrives as the Conv3d kernel (hidden, in_ch, t, p, p); flatten the spatial and
        # channel axes so it multiplies the already-flattened patch rows.
        weight = state.get("proj.weight")
        if weight is not None and weight.ndim > 2:
            state["proj.weight"] = weight.reshape(self.hidden_size, self.patch_dim)

    def forward(self, patches: ttnn.Tensor) -> ttnn.Tensor:
        """`(total_patches, patch_dim)` -> `(total_patches, hidden_size)`."""
        return self.proj.forward(patches)
