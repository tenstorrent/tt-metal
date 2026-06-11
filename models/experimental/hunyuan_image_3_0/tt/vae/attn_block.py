# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import ttnn
from models.experimental.hunyuan_image_3_0.ref.vae.common import GN_EPS, LATENT_H, LATENT_T, LATENT_W, NUM_GROUPS
from models.experimental.hunyuan_image_3_0.tt.vae.conv3d import HunyuanSymmetricConv3d
from models.tt_dit.layers.module import Module
from models.tt_dit.layers.normalization import GroupNorm3D


class AttnBlockTTNN(Module):
    """GroupNorm3D + Q/K/V 1x1 Conv3d + SDPA + proj_out + residual."""

    ATTN_SCALE = 1.0 / math.sqrt(1024)

    def __init__(
        self,
        channels: int,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.spatial = LATENT_H * LATENT_W

        self.norm = GroupNorm3D(
            num_channels=channels,
            num_groups=NUM_GROUPS,
            input_nhw=LATENT_T * LATENT_H * LATENT_W,
            num_batches=1,
            eps=GN_EPS,
            mesh_device=mesh_device,
            dtype=dtype,
        )
        conv_kwargs = dict(
            mesh_device=mesh_device,
            dtype=dtype,
            t=LATENT_T,
            h=LATENT_H,
            w=LATENT_W,
        )
        self.q = HunyuanSymmetricConv3d(channels, channels, kernel_size=1, stride=1, padding=0, **conv_kwargs)
        self.k = HunyuanSymmetricConv3d(channels, channels, kernel_size=1, stride=1, padding=0, **conv_kwargs)
        self.v = HunyuanSymmetricConv3d(channels, channels, kernel_size=1, stride=1, padding=0, **conv_kwargs)
        self.proj_out = HunyuanSymmetricConv3d(channels, channels, kernel_size=1, stride=1, padding=0, **conv_kwargs)

    def load_from_torch(self, torch_block) -> None:
        self.norm.load_torch_state_dict(torch_block.norm.state_dict())
        self.q.load_torch_state_dict(torch_block.q.state_dict())
        self.k.load_torch_state_dict(torch_block.k.state_dict())
        self.v.load_torch_state_dict(torch_block.v.state_dict())
        self.proj_out.load_torch_state_dict(torch_block.proj_out.state_dict())

    def _to_sdpa(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        """BTHWC ROW_MAJOR -> [B, 1, T*H*W, C] TILE (matches ref AttnBlock)."""
        b, t, h, w, c = x_bthwc.shape
        x_btsc = ttnn.reshape(x_bthwc, (b, t, h * w, c))
        x_b1sc = ttnn.reshape(x_btsc, (b, 1, t * h * w, c))
        ttnn.deallocate(x_btsc, force=False)
        return ttnn.to_layout(x_b1sc, ttnn.TILE_LAYOUT)

    def _from_sdpa(self, x_b1sc: ttnn.Tensor, b: int, t: int, h: int, w: int, c: int) -> ttnn.Tensor:
        """[B, 1, T*H*W, C] TILE -> BTHWC ROW_MAJOR for conv3d."""
        x = ttnn.to_layout(x_b1sc, ttnn.ROW_MAJOR_LAYOUT)
        x_btsc = ttnn.reshape(x, (b, t, h * w, c))
        ttnn.deallocate(x, force=False)
        return ttnn.reshape(x_btsc, (b, t, h, w, c))

    def forward(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        residual = x_bthwc
        b, t, h, w, c = x_bthwc.shape

        normed = self.norm(x_bthwc)
        q = self._to_sdpa(self.q(normed))
        k = self._to_sdpa(self.k(normed))
        v = self._to_sdpa(self.v(normed))
        ttnn.deallocate(normed, force=False)

        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=False,
            scale=self.ATTN_SCALE,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q, force=False)
        ttnn.deallocate(k, force=False)
        ttnn.deallocate(v, force=False)

        attn_bthwc = self._from_sdpa(attn, b, t, h, w, c)
        ttnn.deallocate(attn, force=False)

        out = self.proj_out(attn_bthwc)
        ttnn.deallocate(attn_bthwc, force=False)
        return ttnn.add(residual, out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
