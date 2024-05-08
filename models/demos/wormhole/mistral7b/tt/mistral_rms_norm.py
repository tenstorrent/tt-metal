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
        model_config,
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
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name,
        )
        self.model_config = model_config

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.experimental.tensor.interleaved_to_sharded(
            x, sharded_mem_config=self.model_config["SHARDED_NORM_INPUT_MEMCFG"]
        )
        h = ttnn.experimental.operations.primary.rmsnorm(
            x,
            self.eps,
            self.weight,
            program_config=self.model_config["SHARDED_NORM_PRGM_CFG"],
            output_mem_config=self.model_config["SHARDED_NORM_OUTPUT_MEMCFG"],
        )

        return ttnn.experimental.tensor.sharded_to_interleaved(h)
