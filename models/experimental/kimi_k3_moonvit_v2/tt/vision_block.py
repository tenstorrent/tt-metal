# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Encoder block: `MoonViTEncoderLayer` in the reference.

x + attention(norm0(x)), then + mlp(norm1(x)). The norms are RMSNorm without bias, eps
`config.norm_eps`. The reference keeps `wqkv` and `wo` on the block itself; here they live in
`attention`, and `_prepare_torch_state` moves the keys.
"""

from __future__ import annotations

import torch

import ttnn

from .model_config import MoonViTConfig, VisionParallel
from .module import Module, rename_substate
from .rms_norm import RMSNorm
from .rope import RopeTensors
from .vision_attention import TT_Vision_Attention
from .vision_mlp import TT_Vision_MLP


class TT_Vision_Block(Module):
    def __init__(
        self, config: MoonViTConfig, *, mesh_device: ttnn.MeshDevice, parallel: VisionParallel, compute_kernel_config
    ) -> None:
        super().__init__()
        norm_kwargs = dict(eps=config.norm_eps, mesh_device=mesh_device, compute_kernel_config=compute_kernel_config)
        self.norm0 = RMSNorm(config.hidden_size, **norm_kwargs)
        self.norm1 = RMSNorm(config.hidden_size, **norm_kwargs)
        self.attention = TT_Vision_Attention(
            config, mesh_device=mesh_device, parallel=parallel, compute_kernel_config=compute_kernel_config
        )
        self.mlp = TT_Vision_MLP(config, mesh_device=mesh_device, compute_kernel_config=compute_kernel_config)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "wqkv", "attention.wqkv")
        rename_substate(state, "wo", "attention.wo")

    def forward(self, x: ttnn.Tensor, rope: RopeTensors, logical_n: ttnn.Tensor) -> ttnn.Tensor:
        """[1, 1, rows, hidden_size] -> the same shape; rows are this device's shard of the padded,
        window-ordered sequence. `x` is left allocated: the caller owns it."""
        normed = self.norm0(x)
        attn = self.attention(normed, rope, logical_n)
        ttnn.deallocate(normed)
        h = ttnn.add(x, attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn)

        normed = self.norm1(h)
        mlp = self.mlp(normed)
        ttnn.deallocate(normed)
        out = ttnn.add(h, mlp, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(h)
        ttnn.deallocate(mlp)
        return out
