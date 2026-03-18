# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import ttnn


class LayerNorm:
    """Stateful LayerNorm wrapper around `ttnn.layer_norm` for last-dim normalization."""

    total_layernorm_time = 0.0

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

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], key: str, module_prefix: str | None = None) -> None:
        if module_prefix is None:
            module_prefix = ""
        base_key = f"{module_prefix}{key}" if module_prefix else key
        weight_key = f"{base_key}.weight"
        bias_key = f"{base_key}.bias"

        if weight_key not in state_dict:
            raise KeyError(f"Missing required parameter: {weight_key}")
        if bias_key not in state_dict:
            raise KeyError(f"Missing required parameter: {bias_key}")

        self.weight = ttnn.from_torch(
            state_dict[weight_key].reshape(1, 1, self.normalized_shape),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        self.bias = ttnn.from_torch(
            state_dict[bias_key].reshape(1, 1, self.normalized_shape),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

    def __call__(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
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
