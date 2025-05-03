# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn import ShardTensorToMesh
from models.common.lightweightmodule import LightweightModule


class TtMixtralMLP(LightweightModule):
    def __init__(self, mesh_device, state_dict, args, layer_num, dtypes):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.dtypes = dtypes
        self.model_args = args
        self.model_config = args.get_model_config()

        base_name = lambda expert_num: f"layers.{layer_num}.feed_forward.experts.{expert_num}"
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
                f"layers.{layer_num}.feed_forward_multidevice_unsqueezed.experts.{name}"
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

        self.w1 = as_tensor("w1")
        self.w2 = as_tensor("w2")
        self.w3 = as_tensor("w3")

        self.prefill_mlp_config = self.model_config["PREFILL_MLP_COMPUTE_CONFIG"]

    def forward(self, x: ttnn.Tensor, mode="decode") -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        if mode == "prefill":
            seq_len = x.shape[-2]
            compute_kernel_config = self.prefill_mlp_config
            if seq_len >= 2048 // 2:  # Too big to compute. Set different program configs based on seqlen
                # Reshape input to to fit on device and parallelize computation
                x = ttnn.reshape(x, [1, seq_len // 1024, 1024, self.model_args.dim])
                pc_1 = self.model_config["PREFILL_MLP_W1_PRG_CONFIG"]
                pc_2 = self.model_config["PREFILL_MLP_W2_PRG_CONFIG"]
                pc_3 = self.model_config["PREFILL_MLP_W3_PRG_CONFIG"]
            elif seq_len == 128:
                pc_1 = self.model_config["PREFILL_MLP_W1_PRG_CONFIG_128"]
                pc_2 = self.model_config["PREFILL_MLP_W2_PRG_CONFIG_128"]
                pc_3 = self.model_config["PREFILL_MLP_W3_PRG_CONFIG_128"]
            else:  # For some sequence lengths,just use default program config
                pc_1 = None
                pc_2 = None
                pc_3 = None

            w1_out = ttnn.linear(
                x,
                self.w1,
                compute_kernel_config=compute_kernel_config,
                core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_1 else None,
                dtype=ttnn.bfloat16,
                activation="silu" if not pc_1 else None,
                program_config=pc_1,
            )

            w3_out = ttnn.linear(
                x,
                self.w3,
                compute_kernel_config=compute_kernel_config,
                core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_3 else None,
                dtype=ttnn.bfloat16,
                program_config=pc_3,
            )

            x.deallocate(True)
            w2_in = ttnn.multiply(w1_out, w3_out, output_tensor=w1_out)

            w3_out.deallocate(True)

            w2_out = ttnn.linear(
                w2_in,
                self.w2,
                compute_kernel_config=compute_kernel_config,
                core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_2 else None,
                dtype=ttnn.bfloat8_b,
                program_config=pc_2,
            )

            w2_in.deallocate(True)

            if seq_len >= 2048:  # Reshape back to intended shape
                w2_out = ttnn.reshape(w2_out, [1, 1, seq_len, self.model_args.dim])

        else:  # Decode mode
            w1_out = ttnn.matmul(
                x,
                self.w1,
                program_config=self.model_config["FF1_OUTPUT_PROGCFG"],  # SILu activation fused in the op
                memory_config=self.model_config["FF1_OUTPUT_MEMCFG"],
                compute_kernel_config=self.model_args.get_compute_kernel_config(),
                dtype=ttnn.bfloat8_b,
            )
            w3_out = ttnn.matmul(
                x,
                self.w3,
                program_config=self.model_config["FF3_OUTPUT_PROGCFG"],
                memory_config=self.model_config["FF3_OUTPUT_MEMCFG"],
                compute_kernel_config=self.model_args.get_compute_kernel_config(),
                dtype=ttnn.bfloat8_b,
            )
            w2_in = ttnn.mul(w1_out, w3_out)

            w2_out = ttnn.matmul(
                w2_in,
                self.w2,
                program_config=self.model_config["FF2_OUTPUT_PROGCFG"],
                memory_config=self.model_config["FF2_OUTPUT_MEMCFG"],
                compute_kernel_config=self.model_args.get_compute_kernel_config(),
                dtype=ttnn.bfloat8_b,
            )

        return w2_out
