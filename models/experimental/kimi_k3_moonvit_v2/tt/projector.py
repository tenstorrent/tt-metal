# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Projector: `PatchMergerMLPV2` in the reference. fc0 -> exact GELU -> fc1 -> RMSNorm, no biases.

Maps a merged token (4096) to the language model's width (7168). The GELU is the exact erf GELU
(`nn.GELU()`), applied by fc0's `ttnn.linear` as `activation="gelu"` (`UnaryOpType.GELU`, not the
fast approximation). The checkpoint names the linears `proj.0` and `proj.2`
(`nn.Sequential` indices); `_prepare_torch_state` moves them to `fc0` and `fc1`. Sequence parallel:
each device works on its own rows and no collectives are needed.
"""

from __future__ import annotations

import torch

import ttnn

from .linear import Linear
from .model_config import MoonViTConfig
from .module import Module, rename_substate
from .rms_norm import RMSNorm


class TT_Projector(Module):
    def __init__(self, config: MoonViTConfig, *, mesh_device: ttnn.MeshDevice, compute_kernel_config) -> None:
        super().__init__()
        kwargs = dict(bias=False, mesh_device=mesh_device, compute_kernel_config=compute_kernel_config)
        self.fc0 = Linear(config.merged_dim, config.merged_dim, activation_fn="gelu", **kwargs)
        self.fc1 = Linear(config.merged_dim, config.text_hidden_size, **kwargs)
        self.post_norm = RMSNorm(
            config.text_hidden_size,
            eps=config.projector_norm_eps,
            mesh_device=mesh_device,
            compute_kernel_config=compute_kernel_config,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "proj.0", "fc0")
        rename_substate(state, "proj.2", "fc1")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """[1, 1, S/(4*sp), 4096] -> [1, 1, S/(4*sp), 7168], bf16; rows are this device's shard,
        padding included. `x` is left allocated: the caller owns it."""
        h = self.fc0(x)
        y = self.fc1(h)
        ttnn.deallocate(h)
        out = self.post_norm(y)
        ttnn.deallocate(y)
        return out
