# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SD3.5 Medium PatchEmbed implementation,
matching MM-DiT reference.
"""

from __future__ import annotations
import torch
import ttnn

from models.experimental.tt_dit.layers.conv2d import Conv2d
from models.experimental.tt_dit.layers.module import Module


class PatchEmbed(Module):
    """
    PatchEmbed as used in SD3.5 Medium MM-DiT.
    Convolution and reshape
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        mesh_device: ttnn.MeshDevice,
        *,
        flatten: bool = True,
        bias: bool = True,
        out_mesh_axis: int | None = None,
    ):
        super().__init__()

        self.patch_size = (patch_size, patch_size)
        self.flatten = flatten

        if img_size is not None:
            self.img_size = (img_size, img_size)
            self.grid_size = (
                img_size // patch_size,
                img_size // patch_size,
            )
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        # TTNN Conv2D
        self.proj = Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            mesh_device=mesh_device,
            out_mesh_axis=None,
            in_mesh_axis=None,
            ccl_manager=None,
        )

    @classmethod
    def from_torch(
        cls,
        torch_ref: torch.nn.Module,
        mesh_device: ttnn.MeshDevice,
    ):
        """Build TTNN PatchEmbed from PyTorch version"""
        model = cls(
            img_size=torch_ref.img_size[0],
            patch_size=torch_ref.patch_size[0],
            in_channels=torch_ref.proj.in_channels,
            embed_dim=torch_ref.proj.out_channels,
            mesh_device=mesh_device,
            flatten=torch_ref.flatten,
            bias=torch_ref.proj.bias is not None,
        )

        model.proj.load_torch_state_dict(torch_ref.proj.state_dict())
        return model

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # x: [B, H, W, C]  NHWC
        x = self.proj(x)

        # output: [B, H/P, W/P, C]
        if self.flatten:
            B, H, W, C = x.shape
            x = ttnn.reshape(x, (B, H * W, C))

        return x


# Alias for consistency with other SD3.5 Medium components
SD35MediumPatchEmbed = PatchEmbed
