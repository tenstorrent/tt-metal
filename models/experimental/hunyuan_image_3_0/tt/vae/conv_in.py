# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TTNN Hunyuan VAE decoder conv_in — Conv3d(32→1024) + channel repeat shortcut."""

from __future__ import annotations

import torch

import ttnn
from models.experimental.hunyuan_image_3_0.ref.vae.vae_decoder import (
    BLOCK_IN_CHANNELS,
    Z_CHANNELS,
    load_conv_in,
)
from models.experimental.hunyuan_image_3_0.tt.vae.conv3d import (
    BLOCK_IN_CHANNELS as TT_BLOCK_IN,
    HunyuanSymmetricConv3d,
    LATENT_H,
    LATENT_T,
    LATENT_W,
    Z_CHANNELS as TT_Z_CHANNELS,
)
from models.experimental.hunyuan_image_3_0.tt.vae.io import bcthw_to_bthwc, download_bcthw, upload_bthwc
from models.tt_dit.layers.module import Module


class ConvInTTNN(Module):
    """conv_in on replicated mesh (Phase 1)."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()
        assert TT_Z_CHANNELS == Z_CHANNELS and TT_BLOCK_IN == BLOCK_IN_CHANNELS

        self.mesh_device = mesh_device
        self.dtype = dtype
        self.repeats = BLOCK_IN_CHANNELS // Z_CHANNELS

        self.conv = HunyuanSymmetricConv3d(
            Z_CHANNELS,
            BLOCK_IN_CHANNELS,
            kernel_size=3,
            stride=1,
            padding=1,
            mesh_device=mesh_device,
            dtype=dtype,
            t=LATENT_T,
            h=LATENT_H,
            w=LATENT_W,
        )

        pt_conv = load_conv_in(dtype=torch.float32)
        self.conv.load_torch_state_dict(pt_conv.conv.state_dict())
        del pt_conv

    def forward_device(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        return self.conv(x_bthwc)

    def forward(self, z_bcthw: torch.Tensor) -> torch.Tensor:
        z_bthwc = bcthw_to_bthwc(z_bcthw)
        shortcut_bthwc = z_bthwc.repeat_interleave(self.repeats, dim=-1)

        x = upload_bthwc(self.mesh_device, z_bthwc, dtype=self.dtype)
        conv_out = self.forward_device(x)
        ttnn.deallocate(x, force=False)

        shortcut = upload_bthwc(self.mesh_device, shortcut_bthwc, dtype=self.dtype)
        out_bthwc = ttnn.add(conv_out, shortcut, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(conv_out, force=False)
        ttnn.deallocate(shortcut, force=False)

        result = download_bcthw(self.mesh_device, out_bthwc)
        ttnn.deallocate(out_bthwc, force=False)
        return result
