# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


class TtMistralMLP(torch.nn.Module):
    def __init__(
        self,
        device,
        args,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        model_config,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.args = args
        self.model_config = model_config

        base_name = f"layers.{layer_num}.feed_forward"
        torch_weight = lambda name: torch.transpose(self.state_dict[f"{base_name}.{name}.weight"], -2, -1)
        cache_name = lambda name: weight_cache_path / (base_name + f".{name}")
        as_tensor = lambda name: ttnn.as_tensor(
            torch_weight(name),
            dtype=dtype,
            device=self.device,
            layout=self.model_config["MLP_W_LAYOUT_TILE"],
            memory_config=self.model_config["MLP_WEIGHTS_MEMCFG"],
            cache_file_name=cache_name(name),
        )

        self.w1 = as_tensor("w1")
        self.w2 = as_tensor("w2")
        self.w3 = as_tensor("w3")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        w1_out = ttnn.linear(
            x,
            self.w1,
            activation="silu",
            core_grid=self.args.max_grid_size,
            use_1d_systolic_array=True,
            memory_config=self.model_config["FF1_OUTPUT_MEMCFG"],
            compute_kernel_config=self.args.get_compute_kernel_config(),
        )
        w3_out = ttnn.linear(
            x,
            self.w3,
            core_grid=self.args.max_grid_size,
            use_1d_systolic_array=True,
            memory_config=self.model_config["FF3_OUTPUT_MEMCFG"],
            compute_kernel_config=self.args.get_compute_kernel_config(),
        )
        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            memory_config=self.model_config["FF3_OUTPUT_MEMCFG"],
        )
        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            core_grid=self.args.max_grid_size,
            use_1d_systolic_array=True,
            memory_config=self.model_config["FF2_OUTPUT_MEMCFG"],
            compute_kernel_config=self.args.get_compute_kernel_config(),
        )

        return w2_out
