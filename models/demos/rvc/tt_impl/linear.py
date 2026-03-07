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
    ) -> None:
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.weight_tensor = weight_tensor
        self.bias_tensor = bias_tensor
        self.dtype = dtype
        self.memory_config = memory_config if memory_config is not None else ttnn.L1_MEMORY_CONFIG
        self.compute_config = compute_config

    def load_parameters(self, parameters: dict[str, torch.Tensor], key: str, prefix: str = "") -> None:
        base_key = f"{prefix}{key}" if prefix else key
        weight_key = f"{base_key}.weight"
        bias_key = f"{base_key}.bias"

        if weight_key not in parameters:
            raise KeyError(f"Missing required parameter: {weight_key}")

        # Torch Linear weight is [out_features, in_features]. For ttnn.linear use [in_features, out_features].
        wt = parameters[weight_key].reshape(self.out_features, self.in_features).transpose(0, 1).contiguous()
        self.weight_tensor = ttnn.from_torch(
            wt,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )

        self.bias_tensor = None
        if bias_key in parameters and parameters[bias_key] is not None:
            self.bias_tensor = ttnn.from_torch(
                parameters[bias_key].reshape(1, 1, self.out_features),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            )

    def __call__(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        if self.weight_tensor is None:
            raise ValueError("weight_tensor is not set. Provide it in __init__ or call load_parameters().")

        if not ttnn.is_tensor_storage_on_device(self.weight_tensor):
            self.weight_tensor = ttnn.to_device(self.weight_tensor, self.device)
        if self.bias_tensor is not None and not ttnn.is_tensor_storage_on_device(self.bias_tensor):
            self.bias_tensor = ttnn.to_device(self.bias_tensor, self.device)

        a = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)
        b = ttnn.to_layout(self.weight_tensor, ttnn.TILE_LAYOUT)
        bias = None if self.bias_tensor is None else ttnn.to_layout(self.bias_tensor, ttnn.TILE_LAYOUT)

        out = ttnn.linear(
            a,
            b,
            bias=bias,
            # memory_config=self.memory_config,
            dtype=self.dtype,
            compute_kernel_config=self.compute_config,
        )
        _w = self.weight_tensor
        self.weight_tensor = ttnn.to_memory_config(_w, ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)

    def deallocate(self):
        ttnn.deallocate(self.weight_tensor)
        ttnn.deallocate(self.bias_tensor)
