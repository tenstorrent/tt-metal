# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""RMSNorm with a weight and no bias: `nn.RMSNorm` in the reference."""

from __future__ import annotations

import torch

import ttnn

from .module import Module, Parameter


class RMSNorm(Module):
    def __init__(self, dim: int, *, eps: float, mesh_device: ttnn.MeshDevice, compute_kernel_config) -> None:
        super().__init__()
        self.eps = eps
        self.compute_kernel_config = compute_kernel_config
        self.weight = Parameter(shape=(1, dim), device=mesh_device)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "weight" in state:
            state["weight"] = state["weight"].reshape(1, -1)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(
            x,
            epsilon=self.eps,
            weight=self.weight.data,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
