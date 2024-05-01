# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


class TtMixtralMLP(torch.nn.Module):
    def __init__(self, device, state_dict, args, layer_num, expert_num, dtypes):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.dtypes = dtypes
        self.model_args = args
        self.model_config = args.get_model_config()

        base_name = f"layers.{layer_num}.feed_forward.experts.{expert_num}"
        torch_weight = lambda name: self.state_dict[f"{base_name}.{name}.weight"].permute(1, 0)
        cache_name = lambda name: args.weight_cache_path(dtypes[name]) / (f"{base_name}.{expert_num}.{name}")
        as_tensor = lambda name: ttnn.as_tensor(
            torch_weight(name),
            dtype=dtypes[name],
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
            core_grid=self.model_args.max_grid_size,
            use_1d_systolic_array=True,
            memory_config=self.model_config["FF1_OUTPUT_MEMCFG"],
            compute_kernel_config=self.model_args.get_compute_kernel_config(),
        )

        w3_out = ttnn.matmul(
            x,
            self.w3,
            core_grid=self.model_args.max_grid_size,
            use_1d_systolic_array=True,
            memory_config=self.model_config["FF3_OUTPUT_MEMCFG"],
            compute_kernel_config=self.model_args.get_compute_kernel_config(),
        )
        w2_in = ttnn.mul(w1_out, w3_out)
        w2_out = ttnn.matmul(
            w2_in,
            self.w2,
            core_grid=self.model_args.max_grid_size,
            use_1d_systolic_array=True,
            memory_config=self.model_config["FF2_OUTPUT_MEMCFG"],
            compute_kernel_config=self.model_args.get_compute_kernel_config(),
        )

        return w2_out
