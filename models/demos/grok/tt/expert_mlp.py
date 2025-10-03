# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from ttnn import ShardTensorToMesh


class ExpertMLP(LightweightModule):
    def __init__(self, mesh_device, state_dict, args, layer_num, dtypes):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.dtypes = dtypes
        self.model_args = args
        self.model_config = args.get_model_config()

        # Base name for expert weights in Grok
        base_name = lambda expert_num: f"model.layers.{layer_num}.block_sparse_moe.experts.{expert_num}"

        # Concatenate weights from all 8 experts
        torch_weight = lambda name: torch.concat(
            [
                self.state_dict[f"{base_name(expert_num)}.{name}.weight"]
                .permute(1, 0)
                .unsqueeze(0)
                .unsqueeze(0)  # [1, 1, 8192, 16384]
                for expert_num in range(8)
            ],
            dim=0,  # [8, 1, 8192, 16384]
        )

        # ShardTo2DMesh(mesh_shape=(8, 4), dims=(3,2))

        # Cache naming for weights
        if args.dummy_weights:
            cache_name = lambda _: None
        else:
            # Use a simple cache path structure for Grok
            cache_name = lambda name: None  # Simplified for minimal implementation

        # Convert torch weights to ttnn tensors
        as_tensor = lambda name: ttnn.as_tensor(
            torch_weight(name),
            dtype=dtypes[name],
            device=self.mesh_device,
            mesh_mapper=ShardTensorToMesh(self.mesh_device, dim=0),
            layout=self.model_config["MLP_W_LAYOUT_TILE_EXPERTS"],
            memory_config=self.model_config["MLP_WEIGHTS_MEMCFG_EXPERTS"],
            # cache_file_name=cache_name(name),
        )

        # Initialize weight tensors
        self.w1 = as_tensor("w1")  # gate_proj
        self.w2 = as_tensor("w2")  # down_proj
        self.w3 = as_tensor("w3")  # up_proj

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Expert MLP forward pass (decode mode only)
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        # Gate projection with SiLU activation
        w1_out = ttnn.matmul(
            x,
            self.w1,
            program_config=self.model_config["FF1_OUTPUT_PROGCFG_EXPERTS"],
            memory_config=self.model_config["FF1_OUTPUT_MEMCFG_EXPERTS"],
            compute_kernel_config=self.model_args.compute_kernel_config_hifi2,
            dtype=ttnn.bfloat8_b,
        )

        # Up projection
        w3_out = ttnn.matmul(
            x,
            self.w3,
            program_config=self.model_config["FF3_OUTPUT_PROGCFG_EXPERTS"],
            memory_config=self.model_config["FF3_OUTPUT_MEMCFG_EXPERTS"],
            compute_kernel_config=self.model_args.compute_kernel_config_hifi2,
            dtype=ttnn.bfloat8_b,
        )

        # Element-wise multiplication
        w2_in = ttnn.mul(w1_out, w3_out)

        # Down projection
        w2_out = ttnn.matmul(
            w2_in,
            self.w2,
            program_config=self.model_config["FF2_OUTPUT_PROGCFG_EXPERTS"],
            memory_config=self.model_config["FF2_OUTPUT_MEMCFG_EXPERTS"],
            compute_kernel_config=self.model_args.compute_kernel_config_hifi2,
            dtype=ttnn.bfloat8_b,
        )

        return w2_out
