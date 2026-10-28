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
        raise NotImplementedError("TT_Vision_Block.forward")
