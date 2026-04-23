# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn


class BatchNorm2d:
    def __init__(self, *, device, num_features: int, momentum: float = 0.01, eps: float = 1e-5):
        self.device = device
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.running_mean = None
        self.running_var = None
        self.weight = None
        self.bias = None

    def _load_param(self, tensor: torch.Tensor) -> ttnn.Tensor:
        # mapper, _ = _mesh_mapper_and_composer(self.device)
        return ttnn.from_torch(
            tensor.detach().reshape(1, -1, 1, 1).contiguous(),
            device=self.device,
            # mesh_mapper=mapper,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], key: str, module_prefix: str | None = None) -> None:
        base_key = f"{module_prefix}{key}" if module_prefix else key
        # needed to prevent the bug that is causing the weight values to explode despite no writing to the memory that happens in infer_ttnn.py script
        # in 3rd run
        self._weight = self._load_param(state_dict[f"{base_key}.weight"])
        self.weight = self._load_param(state_dict[f"{base_key}.weight"])
        self.bias = self._load_param(state_dict[f"{base_key}.bias"])
        self.running_mean = self._load_param(state_dict[f"{base_key}.running_mean"])
        self.running_var = self._load_param(state_dict[f"{base_key}.running_var"])

    def __call__(self, input: ttnn.Tensor) -> ttnn.Tensor:
        if input.is_sharded():
            input = ttnn.sharded_to_interleaved(input, ttnn.L1_MEMORY_CONFIG)
        input = ttnn.permute(input, (0, 3, 1, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
        input = ttnn.to_layout(input, ttnn.TILE_LAYOUT)
        output = ttnn.batch_norm(
            input,
            eps=self.eps,
            momentum=self.momentum,
            running_mean=self.running_mean,
            running_var=self.running_var,
            weight=self.weight,
            bias=self.bias,
            training=False,
        )
        output = ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.permute(output, (0, 2, 3, 1))
        return output
