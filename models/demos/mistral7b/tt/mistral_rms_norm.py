# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch.nn as nn
import ttnn


class TtRMSNorm(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        weight_cache_path,
        dtype,
        layer_num,
        weight_key,
        eps: float = 1e-05,
    ):
        super().__init__()
        self.device = device
        self.eps = eps
        self.state_dict = state_dict

        if layer_num is None:
            weight_name = f"{weight_key}.weight"
        else:
            weight_name = f"layers.{layer_num}.{weight_key}.weight"

        torch_weight = self.state_dict[weight_name].unsqueeze(0).expand(32, -1)
        cache_name = weight_cache_path / weight_name

        self.weight = ttnn.as_tensor(
            torch_weight,
            device=self.device,
            dtype=ttnn.bfloat16,  # dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.rms_norm(x, weight=self.weight, epsilon=self.eps)
        return x
