# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import ttnn
from models.experimental.hunyuan_image_3_0.ref.vae.vae_decoder import load_mid
from models.experimental.hunyuan_image_3_0.tt.vae.attn_block import AttnBlockTTNN
from models.experimental.hunyuan_image_3_0.tt.vae.io import download_bcthw, upload_bcthw
from models.experimental.hunyuan_image_3_0.tt.vae.resnet_block import ResnetBlockTTNN
from models.tt_dit.layers.module import Module


class MidBlockTTNN(Module):
    """mid.block_1 -> mid.attn_1 -> mid.block_2 on replicated mesh."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()
        self.mesh_device = mesh_device
        self.dtype = dtype

        self.block_1 = ResnetBlockTTNN(1024, mesh_device, dtype=dtype)
        self.attn_1 = AttnBlockTTNN(1024, mesh_device, dtype=dtype)
        self.block_2 = ResnetBlockTTNN(1024, mesh_device, dtype=dtype)

        pt_mid = load_mid(dtype=torch.float32)
        self.block_1.load_from_torch(pt_mid.block_1)
        self.attn_1.load_from_torch(pt_mid.attn_1)
        self.block_2.load_from_torch(pt_mid.block_2)
        del pt_mid

    def forward_device(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        x = self.block_1(x_bthwc)
        x = self.attn_1(x)
        return self.block_2(x)

    def forward(self, x_bcthw: torch.Tensor) -> torch.Tensor:
        x = upload_bcthw(self.mesh_device, x_bcthw, dtype=self.dtype)
        out = self.forward_device(x)
        ttnn.deallocate(x, force=False)
        result = download_bcthw(self.mesh_device, out)
        ttnn.deallocate(out, force=False)
        return result
