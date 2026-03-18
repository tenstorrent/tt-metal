# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import ttnn


class Linear:
    """Stateful Linear wrapper around `ttnn.linear` for NLC tensors."""

    def __init__(
        self,
        device: ttnn.MeshDevice,
        in_features: int,
        out_features: int,
        weight_tensor: ttnn.Tensor | None = None,
        bias_tensor: ttnn.Tensor | None = None,
        dtype: ttnn.DataType | None = None,
        memory_config: ttnn.MemoryConfig | None = None,
        compute_config: ttnn.DeviceComputeKernelConfig | None = None,
        activation: str | None = None,
    ) -> None:
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.weight_tensor = weight_tensor
        self.bias_tensor = bias_tensor
        self.dtype = dtype
        self.memory_config = memory_config if memory_config is not None else ttnn.L1_MEMORY_CONFIG
        self.compute_config = compute_config
        self.activation = activation

    def load_state_dict(self, parameters: dict[str, torch.Tensor], key: str, module_prefix: str = "") -> None:
        base_key = f"{module_prefix}{key}" if module_prefix else key
        weight_key = f"{base_key}.weight"
        bias_key = f"{base_key}.bias"

        if weight_key not in parameters:
            raise KeyError(f"Missing required parameter: {weight_key}")

        # Torch Linear weight is [out_features, in_features]. For ttnn.linear use [in_features, out_features].
        wt = parameters[weight_key].reshape(self.out_features, self.in_features).transpose(0, 1).contiguous()
        self.weight_tensor = ttnn.from_torch(
            wt,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

        self.bias_tensor = None
        if bias_key in parameters and parameters[bias_key] is not None:
            self.bias_tensor = ttnn.from_torch(
                parameters[bias_key].reshape(1, 1, self.out_features),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )

    def __call__(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)
        out = ttnn.linear(
            input_tensor,
            self.weight_tensor,
            bias=self.bias_tensor,
            # memory_config=self.memory_config,
            dtype=self.dtype,
            activation=self.activation,
            compute_kernel_config=self.compute_config,
        )
        return out

    def deallocate(self):
        ttnn.deallocate(self.weight_tensor)
        ttnn.deallocate(self.bias_tensor)
