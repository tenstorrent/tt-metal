# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
MoonViT patch embedding.

`KimiVLImageProcessor.patchify` pre-cuts the image into a stack of
per-patch tiles, so the tensor that reaches `MoonVisionPatchEmbed`
has shape `(L, 3, 14, 14)` — one row per patch.

`MoonVisionPatchEmbed.self.proj` is `nn.Conv2d(3, 1152, kernel=14, stride=14)`,
which on a 14x14 patch produces a (L, 1152, 1, 1) tensor. The
subsequent `.view(L, -1)` makes that (L, 1152). End result: the Conv2d
is functionally a per-patch Linear projection from 588 (=3*14*14) to
1152.

We implement it as exactly that: reshape the Conv2d weight from
[1152, 3, 14, 14] to [1152, 588] and run a single `ttnn.linear`. This
matches the gemma3 / llama-vision patch-embed pattern.

The full `MoonVisionPatchEmbed` (Conv2d projection + bicubic-interpolated
2D learned posemb add) is built in two stages:
  - `MoonVisionPatchProj`  -- this file, this class (step 3 of plan).
  - `MoonVisionPatchEmbed` -- to be added in step 5, composes the
    projection with `Learnable2DInterpPosEmb` from pos_emb.py.

Reference: `MoonVisionPatchEmbed` in modeling_kimi_k25.py.
"""
from __future__ import annotations

from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3.tt.moonvit.pos_emb import GridHws, Learnable2DInterpPosEmb

PATCH_SIZE = 14
IN_CHANNELS = 3
# Flattened patch dim: 3 * 14 * 14 = 588 (not tile-aligned; ttnn auto-pads).
PATCH_FLAT_DIM = IN_CHANNELS * PATCH_SIZE * PATCH_SIZE


def _is_mesh_device(device) -> bool:
    return type(device).__name__ == "MeshDevice"


class MoonVisionPatchProj(LightweightModule):
    """Conv2d-as-Linear patch projection: (L, 3, 14, 14) -> (L, 1152).

    The HF reference is `MoonVisionPatchEmbed.self.proj`. Because the
    image processor already extracts 14x14 patches, the Conv2d is
    equivalent to a per-patch Linear from 588 to `out_dim`.
    """

    def __init__(
        self,
        mesh_device,
        out_dim: int,
        proj_weight: torch.Tensor,  # shape [out_dim, 3, 14, 14] (Conv2d weight)
        proj_bias: Optional[torch.Tensor],  # shape [out_dim] or None
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = mesh_device
        self.dtype = dtype
        self.out_dim = int(out_dim)

        # Validate Conv2d weight shape.
        assert proj_weight.ndim == 4, f"expected 4D Conv2d weight, got shape {tuple(proj_weight.shape)}"
        c_out, c_in, kh, kw = proj_weight.shape
        assert c_out == out_dim, f"weight out dim {c_out} != out_dim {out_dim}"
        assert c_in == IN_CHANNELS, f"weight in channels {c_in} != {IN_CHANNELS}"
        assert (kh, kw) == (PATCH_SIZE, PATCH_SIZE), f"kernel {(kh, kw)} != {(PATCH_SIZE, PATCH_SIZE)}"

        # Flatten Conv2d weight [out, in, kh, kw] -> linear weight [in*kh*kw, out].
        # The Conv2d convolution result on a 14x14 patch is a 1x1 spatial output
        # whose value is sum over (c, kh, kw) of weight[o, c, kh, kw] * x[c, kh, kw].
        # That's a dot product of x.flatten() with weight[o].flatten() — i.e.,
        # weight in linear form is weight.view(out_dim, -1) and we transpose
        # to ttnn.linear convention [in, out].
        flat_weight = proj_weight.detach().to(torch.bfloat16).reshape(out_dim, -1).transpose(0, 1).contiguous()
        # flat_weight shape: [588, out_dim].

        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if _is_mesh_device(mesh_device) else None

        self.weight = ttnn.as_tensor(
            flat_weight,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        if proj_bias is not None:
            assert proj_bias.shape == (out_dim,), f"bias shape {tuple(proj_bias.shape)} != ({out_dim},)"
            bias_4d = proj_bias.detach().to(torch.bfloat16).view(1, 1, 1, out_dim).contiguous()
            self.bias = ttnn.as_tensor(
                bias_4d,
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            )
        else:
            self.bias = None

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
        ref_proj: torch.nn.Conv2d,
        out_dim: Optional[int] = None,
        dtype=ttnn.bfloat16,
    ) -> "MoonVisionPatchProj":
        """Construct from a torch nn.Conv2d (or any module with .weight/.bias)."""
        assert hasattr(ref_proj, "weight"), f"expected a Conv2d-like module, got {type(ref_proj).__name__}"
        weight = ref_proj.weight.data
        bias = ref_proj.bias.data if getattr(ref_proj, "bias", None) is not None else None
        return cls(
            mesh_device=mesh_device,
            out_dim=int(out_dim if out_dim is not None else weight.shape[0]),
            proj_weight=weight,
            proj_bias=bias,
            dtype=dtype,
        )

    @staticmethod
    def flatten_patches(x: torch.Tensor) -> torch.Tensor:
        """(L, 3, 14, 14) -> (L, 588).

        Useful for tests / host-side preprocessing before pushing to device.
        """
        assert x.ndim == 4 and x.shape[1:] == (
            IN_CHANNELS,
            PATCH_SIZE,
            PATCH_SIZE,
        ), f"expected (L, {IN_CHANNELS}, {PATCH_SIZE}, {PATCH_SIZE}), got {tuple(x.shape)}"
        return x.reshape(x.shape[0], -1)

    def forward(self, x_flat: ttnn.Tensor, memory_config: Optional["ttnn.MemoryConfig"] = None) -> ttnn.Tensor:
        """Apply per-patch projection.

        Args:
            x_flat: device tensor of shape (..., L, 588). Caller is responsible
                for flattening (L, 3, 14, 14) -> (L, 588) host-side before
                pushing — see `flatten_patches`.

        Returns:
            device tensor of shape (..., L, out_dim).
        """
        out = ttnn.linear(
            x_flat,
            self.weight,
            bias=self.bias,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG,
        )
        return out


class MoonVisionPatchEmbed(LightweightModule):
    """Full HF-equivalent patch embed: per-patch projection + interpolated posemb.

    Forward mirrors `MoonVisionPatchEmbed.forward(pixel_values, grid_hws)`:
        x = self.proj(pixel_values).view(L, -1)
        x = x + interpolated_posemb(grid_hws)
        return x

    The posemb interpolation runs on host (see `pos_emb.py`); the add and
    the projection live on device. There is one host→device transfer per
    forward for the posemb tensor (size proportional to total patch
    count). This is the v1 design — see plan Deferred #2 for the
    (H, W)-keyed cache that removes the repeated cost.
    """

    def __init__(
        self,
        mesh_device,
        proj: MoonVisionPatchProj,
        pos_emb: Learnable2DInterpPosEmb,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = mesh_device
        self.dtype = dtype
        self.proj = proj
        self.pos_emb = pos_emb

    @classmethod
    def from_torch(
        cls,
        mesh_device,
        ref: torch.nn.Module,
        dtype=ttnn.bfloat16,
    ) -> "MoonVisionPatchEmbed":
        """Construct from the HF MoonVisionPatchEmbed reference module.

        Pulls `ref.proj` (Conv2d) and `ref.pos_emb` (Learnable2DInterpPosEmb).
        """
        assert hasattr(ref, "proj"), f"expected HF MoonVisionPatchEmbed-like module; got {type(ref).__name__}"
        assert hasattr(ref, "pos_emb"), f"expected `.pos_emb` on {type(ref).__name__}"
        return cls(
            mesh_device=mesh_device,
            proj=MoonVisionPatchProj.from_torch(mesh_device, ref.proj, dtype=dtype),
            pos_emb=Learnable2DInterpPosEmb.from_torch(ref.pos_emb),
            dtype=dtype,
        )

    def _push_pos_embs(self, grid_hws: GridHws) -> ttnn.Tensor:
        """Host-compute the interpolated posemb and push to device.

        Returns a 4D device tensor shaped (1, 1, L, D) so it broadcasts
        cleanly with the projection output.
        """
        pos_pt = self.pos_emb.compute(grid_hws, dtype=torch.bfloat16)
        L, D = pos_pt.shape
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if _is_mesh_device(self.device) else None
        return ttnn.from_torch(
            pos_pt.view(1, 1, L, D).contiguous(),
            device=self.device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

    def forward(
        self,
        x_flat: ttnn.Tensor,
        grid_hws: GridHws,
        memory_config: Optional["ttnn.MemoryConfig"] = None,
    ) -> ttnn.Tensor:
        """Apply projection then add interpolated posemb.

        Args:
            x_flat: device tensor of shape (1, 1, L, 588) — already flattened
                from (L, 3, 14, 14) per `MoonVisionPatchProj.flatten_patches`.
            grid_hws: per-image (H, W) sizes; used to select per-image posemb slices.

        Returns:
            device tensor (1, 1, L, hidden_size).
        """
        proj_out = self.proj(x_flat)
        pos_tt = self._push_pos_embs(grid_hws)
        out = ttnn.add(
            proj_out,
            pos_tt,
            memory_config=memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(proj_out)
        ttnn.deallocate(pos_tt)
        return out
