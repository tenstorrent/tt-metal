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

    def load_state_dict(
        self, state_dict: dict[str, torch.Tensor], key: str | None = None, module_prefix: str | None = None
    ) -> None:
        if module_prefix is None:
            module_prefix = ""

        if key is None:
            pairs = [("weight", "bias"), ("gamma", "beta")]
        else:
            base_key = f"{module_prefix}{key}" if module_prefix else key
            pairs = [(f"{base_key}.weight", f"{base_key}.bias"), (f"{base_key}.gamma", f"{base_key}.beta")]

        weight_key = None
        bias_key = None
        for candidate_weight_key, candidate_bias_key in pairs:
            resolved_weight_key = (
                f"{module_prefix}{candidate_weight_key}" if key is None and module_prefix else candidate_weight_key
            )
            resolved_bias_key = (
                f"{module_prefix}{candidate_bias_key}" if key is None and module_prefix else candidate_bias_key
            )
            if resolved_weight_key in state_dict and resolved_bias_key in state_dict:
                weight_key = resolved_weight_key
                bias_key = resolved_bias_key
                break

        if weight_key is None or bias_key is None:
            if key is None:
                raise KeyError("Missing required LayerNorm parameters: expected weight/bias or gamma/beta")
            raise KeyError(f"Missing required LayerNorm parameters under base key: {base_key}")

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
