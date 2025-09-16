# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtMixtralMLP(LightweightModule):
    def __init__(self, mesh_device, state_dict, args, layer_num, dtypes):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.dtypes = dtypes
        self.model_args = args
        self.model_config = args.get_model_config()
        base_name = lambda expert_num: f"layers.{layer_num}.block_sparse_moe.experts.{expert_num}"
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
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
            layout=self.model_config["MLP_W_LAYOUT_TILE"],
            memory_config=self.get_mem_config(name, torch_weight),
            cache_file_name=cache_name(name),
        )

        self.w1 = as_tensor("w1")
        self.w2 = as_tensor("w2")
        self.w3 = as_tensor("w3")

        self.prefill_mlp_config = self.model_config["MIXTRAL_PREFILL_MLP_COMPUTE_CONFIG"]

    def get_mem_config(self, name: str, weight) -> ttnn._ttnn.tensor.MemoryConfig:
        num_device = self.mesh_device.get_num_devices()
        if name == "w2":
            _, _, hidden_dim, dim = weight(name).shape
            return self.model_args.create_dram_sharded_mem_config(hidden_dim, dim)
        else:
            _, _, dim, hidden_dim = weight(name).shape
            return self.model_args.create_dram_sharded_mem_config(dim, hidden_dim)

    def forward(self, x: ttnn.Tensor, mode="decode") -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        if mode == "prefill":
            seq_len = x.shape[-2]
            original_shape = x.shape
            compute_kernel_config = self.prefill_mlp_config
            if (
                seq_len >= self.model_args.prefill_len_cutoff
            ):  # Too big to compute. Set different program configs based on seqlen
                # Reshape input to to fit on device and parallelize computation
                x = ttnn.reshape(
                    x, [1, seq_len // self.model_args.prefill_len_cutoff, self.model_args.prefill_len_cutoff, -1]
                )
                pc_1 = self.model_config["PREFILL_MIXTRAL_MLP_W1_PRG_CONFIG"](seq_len)
                pc_2 = self.model_config["PREFILL_MLP_W2_PRG_CONFIG"](seq_len)
                pc_3 = self.model_config["PREFILL_MIXTRAL_MLP_W3_PRG_CONFIG"](seq_len)
            else:
                pc_1 = self.model_config["PREFILL_MLP_W1_PRG_CONFIG_128"]
                pc_2 = self.model_config["PREFILL_MLP_W2_PRG_CONFIG_128"]
                pc_3 = self.model_config["PREFILL_MLP_W3_PRG_CONFIG_128"]

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

            ttnn.deallocate(x)

            w2_in = ttnn.multiply(w1_out, w3_out, dtype=ttnn.bfloat16, memory_config=w1_out.memory_config())

            ttnn.deallocate(w3_out)
            ttnn.deallocate(w1_out)

            w2_out = ttnn.linear(
                w2_in,
                self.w2,
                compute_kernel_config=compute_kernel_config,
                core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_2 else None,
                dtype=ttnn.bfloat8_b,
                program_config=pc_2,
            )

            ttnn.deallocate(w2_in)

            w2_out = ttnn.reshape(w2_out, original_shape)

        else:  # Decode
            w1_out = ttnn.matmul(
                x,
                self.w1,
                program_config=self.model_args.dram_matmul_config(
                    1, 4096, 14336, num_cores=8, fused_activation=ttnn.UnaryOpType.SILU
                ),
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                compute_kernel_config=self.model_args.compute_kernel_config_lofi,
                dtype=ttnn.bfloat8_b,
            )
            w3_out = ttnn.matmul(
                x,
                self.w3,
                program_config=self.model_args.dram_matmul_config(1, 4096, 14336, num_cores=8),
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                compute_kernel_config=self.model_args.compute_kernel_config_lofi,
                dtype=ttnn.bfloat8_b,
            )

            w2_in = ttnn.mul(w1_out, w3_out)
            w1_out.deallocate(True)
            w3_out.deallocate(True)
            w2_out = ttnn.matmul(
                w2_in,
                self.w2,
                program_config=self.model_args.dram_matmul_config(1, 14336, 4096, num_cores=8),
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                compute_kernel_config=self.model_args.compute_kernel_config_lofi,
                dtype=ttnn.bfloat8_b,
            )
            w2_in.deallocate(True)
            mc = ttnn.MemoryConfig(memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type=ttnn.BufferType.L1)
            w2_out = ttnn.to_memory_config(w2_out, mc)

        return w2_out
