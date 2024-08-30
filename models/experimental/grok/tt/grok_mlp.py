# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn import ShardTensorToMesh
from models.experimental.grok.tt.grok_common import LightweightModule


class TtGrokMLP(LightweightModule):
    def __init__(self, mesh_device, state_dict, args, layer_num, dtypes):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.dtypes = dtypes
        self.model_args = args
        self.model_config = args.get_model_config()

        base_name = lambda expert_num: f"model.layers.{layer_num}.moe_block.experts.{expert_num}"
        torch_weight = lambda name: torch.concat(
            [
                self.state_dict[f"{base_name(expert_num)}.{name}.weight"].permute(1, 0).unsqueeze(0).unsqueeze(0)
                for expert_num in range(8)
            ],
            dim=0,
        )
        if args.dummy_weights:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: args.weight_cache_path(dtypes[name]) / (
                f"model.layers.{layer_num}.moe_block.experts.{name}"
            )
        as_tensor = lambda name: ttnn.as_tensor(
            torch_weight(name),
            dtype=dtypes[name],
            device=self.mesh_device,
            mesh_mapper=ShardTensorToMesh(self.mesh_device, dim=0),
            layout=self.model_config["MLP_W_LAYOUT_TILE"],
            memory_config=self.model_config["MLP_WEIGHTS_MEMCFG"],
            cache_file_name=cache_name(name),
        )

        self.w1 = as_tensor("linear")
        self.w2 = as_tensor("linear_1")
        self.w3 = as_tensor("linear_v")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        w1_out = ttnn.matmul(
            x,
            self.w1,
            program_config=self.model_config["FF1_OUTPUT_PROGCFG"],  # GELU activation fused in the op
            memory_config=self.model_config["FF1_OUTPUT_MEMCFG"],
            compute_kernel_config=self.model_args.get_compute_kernel_config(),
            dtype=ttnn.bfloat16,
        )
        w3_out = ttnn.matmul(
            x,
            self.w3,
            program_config=self.model_config["FF3_OUTPUT_PROGCFG"],
            memory_config=self.model_config["FF3_OUTPUT_MEMCFG"],
            compute_kernel_config=self.model_args.get_compute_kernel_config(),
            dtype=ttnn.bfloat16,
        )
        w2_in = ttnn.mul(w1_out, w3_out)

        w2_out = ttnn.matmul(
            w2_in,
            self.w2,
            program_config=self.model_config["FF2_OUTPUT_PROGCFG"],
            memory_config=self.model_config["FF2_OUTPUT_MEMCFG"],
            compute_kernel_config=self.model_args.get_compute_kernel_config(),
            dtype=ttnn.bfloat16,
        )

        return w2_out
