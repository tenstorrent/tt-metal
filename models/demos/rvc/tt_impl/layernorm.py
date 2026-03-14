# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import ttnn


class LayerNorm:
    """Stateful LayerNorm wrapper around `ttnn.layer_norm` for last-dim normalization."""

    def __init__(
        self,
        device: ttnn.MeshDevice,
        normalized_shape: int,
        eps: float = 1e-5,
        weight: ttnn.Tensor | None = None,
        bias: ttnn.Tensor | None = None,
        dtype: ttnn.DataType | None = None,
        memory_config: ttnn.MemoryConfig | None = None,
    ) -> None:
        self.device = device
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = weight
        self.bias = bias
        self.dtype = dtype
        self.memory_config = memory_config

    def load_parameters(self, parameters: dict[str, torch.Tensor], key: str, prefix: str = "") -> None:
        base_key = f"{prefix}{key}" if prefix else key
        weight_key = f"{base_key}.weight"
        bias_key = f"{base_key}.bias"

        if weight_key not in parameters:
            raise KeyError(f"Missing required parameter: {weight_key}")
        if bias_key not in parameters:
            raise KeyError(f"Missing required parameter: {bias_key}")

        self.weight = ttnn.from_torch(
            parameters[weight_key].reshape(1, 1, self.normalized_shape),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        self.bias = ttnn.from_torch(
            parameters[bias_key].reshape(1, 1, self.normalized_shape),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

    def __call__(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        if self.weight is None or self.bias is None:
            raise ValueError("LayerNorm parameters are not set. Provide them in __init__ or call load_parameters().")

        self.weight = ttnn.to_memory_config(self.weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        out = ttnn.layer_norm(
            input_tensor,
            epsilon=self.eps,
            weight=self.weight,
            bias=self.bias,
            memory_config=self.memory_config,
        )
        return out

    def deallocate(self) -> None:
        if self.weight is not None:
            ttnn.deallocate(self.weight)
            self.weight = None
        if self.bias is not None:
            ttnn.deallocate(self.bias)
            self.bias = None
