# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
V2-VISION-V2 (8/N): qwen3.6 vision PatchMerger.

Mirrors `Qwen3VLVisionPatchMerger` (HF):
  - LayerNorm on hidden_size=1152, eps=1e-6
  - reshape: [N, 1152] -> [N // spatial_merge_unit, 1152 * spatial_merge_unit = 4608]
  - Linear(4608 -> 4608) + GELU
  - Linear(4608 -> out_hidden_size=5120)

qwen3.6 uses `use_postshuffle_norm=False` (norm BEFORE the spatial reshape).
qwen3.6 has empty `deepstack_visual_indexes`, so only the main merger exists
(no deepstack mergers).

Weights are replicated across the full BH GLX mesh — the merger is small
(<150 MB) and runs only once per image at prefill, so TP-sharding it offers
no meaningful speedup.
"""

from __future__ import annotations

import torch

import ttnn
from models.tt_dit.layers.linear import Linear
from models.tt_dit.layers.module import Module
from models.tt_dit.layers.normalization import LayerNorm


class Qwen36VisionPatchMergerTP(Module):
    """qwen3.6 vision PatchMerger. Replicated weights across the mesh."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        state_dict: dict[str, torch.Tensor],
        *,
        hidden_size: int = 1152,
        spatial_merge_size: int = 2,
        out_hidden_size: int = 5120,
        norm_eps: float = 1e-6,
        dtype: ttnn.DataType = ttnn.bfloat16,
        state_dict_prefix: str = "",
    ) -> None:
        super().__init__()
        self.mesh_device = mesh_device
        self.hidden_size = hidden_size
        self.merged_dim = hidden_size * (spatial_merge_size**2)  # 4608
        self.spatial_merge_unit = spatial_merge_size**2  # 4

        # LayerNorm on the UN-shuffled dim (hidden_size=1152), per use_postshuffle_norm=False
        self.norm = LayerNorm(
            embedding_dim=hidden_size,
            norm_eps=norm_eps,
            norm_elementwise_affine=True,
            bias=True,
            mesh_device=mesh_device,
        )

        # Linear fc1: 4608 -> 4608, GELU fused
        self.linear_fc1 = Linear(
            in_features=self.merged_dim,
            out_features=self.merged_dim,
            bias=True,
            activation_fn="gelu",
            dtype=dtype,
            mesh_device=mesh_device,
        )

        # Linear fc2: 4608 -> 5120
        self.linear_fc2 = Linear(
            in_features=self.merged_dim,
            out_features=out_hidden_size,
            bias=True,
            dtype=dtype,
            mesh_device=mesh_device,
        )

        clean: dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            k2 = k[len(state_dict_prefix) :] if state_dict_prefix and k.startswith(state_dict_prefix) else k
            clean[k2] = v
        self.load_torch_state_dict(clean, strict=False)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Input: replicated `[B, 1, seq, hidden_size=1152]` (or 3D `[B, seq, 1152]`).
        Output: replicated `[B, 1, seq//spatial_merge_unit, out_hidden_size=5120]`.
        """
        # LayerNorm on the last dim (1152) — operates on whatever shape we pass
        x = self.norm.forward(x)

        # Reshape to merge spatial_merge_unit consecutive tokens into one row of 4608.
        # If x is [B, 1, seq, 1152], reshape to [B, 1, seq // unit, 4608].
        orig_shape = list(x.shape)
        assert orig_shape[-1] == self.hidden_size, f"unexpected last dim {orig_shape[-1]}"
        seq_old = orig_shape[-2]
        assert (
            seq_old % self.spatial_merge_unit == 0
        ), f"seq_len {seq_old} not divisible by spatial_merge_unit {self.spatial_merge_unit}"
        new_shape = list(orig_shape)
        new_shape[-2] = seq_old // self.spatial_merge_unit
        new_shape[-1] = self.merged_dim
        x = ttnn.reshape(x, new_shape)

        # Linear fc1 (GELU fused) + Linear fc2
        x = self.linear_fc1.forward(x)
        x = self.linear_fc2.forward(x)
        return x
