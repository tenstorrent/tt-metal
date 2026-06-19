# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
MiniMax-M3-VL patch embedding (Conv3d-as-Linear).

The HF reference patch embed is `nn.Conv3d(3, 1280, kernel=(2,14,14),
stride=(2,14,14), bias=False)`. The M3 image processor (Qwen2-VL lineage)
pre-cuts each image into non-overlapping spatio-temporal patches and
*flattens* them, so `pixel_values` already arrives as `(L, 1176)` where
1176 = 3 (channels) * 2 (temporal_patch) * 14 * 14. On such a patch the
Conv3d reduces to a single dot product per output channel — i.e. a
per-patch Linear from 1176 to 1280, with no bias.

We implement exactly that: flatten the Conv3d weight `[1280, 3, 2, 14, 14]`
to `[1280, 1176]` and run one `ttnn.linear`. (The `_m3_loader` reference
already pre-flattens the Conv3d weight into an `nn.Linear`, so `from_torch`
also accepts a plain 2D Linear weight.)

There is no learned position embedding (M3-VL is RoPE-only) and no class
token, so the projection output feeds straight into `pre_layrnorm`.

Reference: `MiniMaxM3VLVisionEmbeddings.proj` (Conv3d).
"""
from __future__ import annotations

from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

PATCH_SIZE = 14
IN_CHANNELS = 3
TEMPORAL_PATCH_SIZE = 2
# Flattened patch dim: 3 * 2 * 14 * 14 = 1176.
PATCH_FLAT_DIM = IN_CHANNELS * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE


def _is_mesh_device(device) -> bool:
    return type(device).__name__ == "MeshDevice"


class M3VLPatchEmbed(LightweightModule):
    """Conv3d-as-Linear patch projection: (L, 1176) -> (L, 1280), no bias."""

    def __init__(
        self,
        mesh_device,
        out_dim: int,
        proj_weight: torch.Tensor,  # [out_dim, 1176] (Linear) or [out_dim, 3, 2, 14, 14] (Conv3d)
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = mesh_device
        self.dtype = dtype
        self.out_dim = int(out_dim)

        # Accept either a pre-flattened Linear weight or a raw Conv3d weight.
        if proj_weight.ndim == 5:
            c_out, c_in, kt, kh, kw = proj_weight.shape
            assert c_out == out_dim, f"weight out dim {c_out} != out_dim {out_dim}"
            assert c_in == IN_CHANNELS, f"weight in channels {c_in} != {IN_CHANNELS}"
            assert (kt, kh, kw) == (
                TEMPORAL_PATCH_SIZE,
                PATCH_SIZE,
                PATCH_SIZE,
            ), f"kernel {(kt, kh, kw)} != {(TEMPORAL_PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)}"
            flat = proj_weight.reshape(out_dim, -1)
        else:
            assert proj_weight.ndim == 2, f"expected 2D or 5D weight, got {tuple(proj_weight.shape)}"
            assert proj_weight.shape[0] == out_dim, f"weight out dim {proj_weight.shape[0]} != {out_dim}"
            flat = proj_weight
        assert flat.shape[1] == PATCH_FLAT_DIM, f"flattened in dim {flat.shape[1]} != {PATCH_FLAT_DIM}"

        # Flatten to ttnn.linear convention [in, out].
        flat_weight = flat.detach().to(torch.bfloat16).transpose(0, 1).contiguous()  # [1176, out_dim]
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if _is_mesh_device(mesh_device) else None
        self.weight = ttnn.as_tensor(
            flat_weight,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    @classmethod
    def from_torch(
        cls,
        mesh_device,
        ref_proj: torch.nn.Module,  # nn.Linear (pre-flattened) or nn.Conv3d
        out_dim: Optional[int] = None,
        dtype=ttnn.bfloat16,
    ) -> "M3VLPatchEmbed":
        assert hasattr(ref_proj, "weight"), f"expected a Conv3d/Linear-like module, got {type(ref_proj).__name__}"
        weight = ref_proj.weight.data
        return cls(
            mesh_device=mesh_device,
            out_dim=int(out_dim if out_dim is not None else weight.shape[0]),
            proj_weight=weight,
            dtype=dtype,
        )

    @staticmethod
    def flatten_patches(x: torch.Tensor) -> torch.Tensor:
        """(L, 3, 2, 14, 14) -> (L, 1176). For tests / host preprocessing of raw patches."""
        assert x.ndim == 5 and x.shape[1:] == (
            IN_CHANNELS,
            TEMPORAL_PATCH_SIZE,
            PATCH_SIZE,
            PATCH_SIZE,
        ), f"expected (L, {IN_CHANNELS}, {TEMPORAL_PATCH_SIZE}, {PATCH_SIZE}, {PATCH_SIZE}), got {tuple(x.shape)}"
        return x.reshape(x.shape[0], -1)

    def forward(self, x_flat: ttnn.Tensor, memory_config: Optional["ttnn.MemoryConfig"] = None) -> ttnn.Tensor:
        """Apply the per-patch projection.

        Args:
            x_flat: device tensor of shape (..., L, 1176).

        Returns:
            device tensor of shape (..., L, out_dim).
        """
        return ttnn.linear(
            x_flat,
            self.weight,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG,
        )
