# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Replicated linear layer without bias support beyond what MoonViT needs (it has none)."""

from __future__ import annotations

import torch

import ttnn

from .module import Module, Parameter


class Linear(Module):
    """`y = x @ W.T`, weight stored transposed as [in, out]; loads a torch `nn.Linear` state."""

    def __init__(
        self, in_features: int, out_features: int, *, bias: bool, mesh_device: ttnn.MeshDevice, compute_kernel_config
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_kernel_config = compute_kernel_config
        self.weight = Parameter(shape=(in_features, out_features), device=mesh_device)
        self.bias = Parameter(shape=(1, out_features), device=mesh_device) if bias else None

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "weight" in state:
            state["weight"] = state["weight"].transpose(0, 1)
        if "bias" in state:
            state["bias"] = state["bias"].reshape(1, -1)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.linear(
            x,
            self.weight.data,
            bias=self.bias.data if self.bias is not None else None,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
