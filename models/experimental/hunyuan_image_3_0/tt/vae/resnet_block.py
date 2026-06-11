# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


import ttnn
from models.experimental.hunyuan_image_3_0.ref.vae.common import GN_EPS, LATENT_H, LATENT_T, LATENT_W, NUM_GROUPS
from models.experimental.hunyuan_image_3_0.tt.vae.conv3d import HunyuanSymmetricConv3d
from models.tt_dit.layers.module import Module
from models.tt_dit.layers.normalization import GroupNorm3D


class ResnetBlockTTNN(Module):
    """GroupNorm3D -> SiLU -> Conv3d x2 + residual (BTHWC ROW_MAJOR)."""

    def __init__(
        self,
        channels: int,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.channels = channels
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.input_nhw = LATENT_T * LATENT_H * LATENT_W

        gn_kwargs = dict(
            num_groups=NUM_GROUPS,
            input_nhw=self.input_nhw,
            num_batches=1,
            eps=GN_EPS,
            mesh_device=mesh_device,
            dtype=dtype,
        )
        self.norm1 = GroupNorm3D(num_channels=channels, **gn_kwargs)
        self.norm2 = GroupNorm3D(num_channels=channels, **gn_kwargs)
        self.conv1 = HunyuanSymmetricConv3d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            mesh_device=mesh_device,
            dtype=dtype,
            t=LATENT_T,
            h=LATENT_H,
            w=LATENT_W,
        )
        self.conv2 = HunyuanSymmetricConv3d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            mesh_device=mesh_device,
            dtype=dtype,
            t=LATENT_T,
            h=LATENT_H,
            w=LATENT_W,
        )

        if prefix:
            self._prefix = prefix

    def load_from_torch(self, torch_block) -> None:
        self.norm1.load_torch_state_dict(torch_block.norm1.state_dict())
        self.norm2.load_torch_state_dict(torch_block.norm2.state_dict())
        self.conv1.load_torch_state_dict(torch_block.conv1.state_dict())
        self.conv2.load_torch_state_dict(torch_block.conv2.state_dict())

    def forward(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        residual = x_bthwc

        h = self.norm1(x_bthwc)
        h = ttnn.silu(h, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        h = self.conv1(h)

        h = self.norm2(h)
        h = ttnn.silu(h, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        h = self.conv2(h)

        return ttnn.add(residual, h, memory_config=ttnn.DRAM_MEMORY_CONFIG)
