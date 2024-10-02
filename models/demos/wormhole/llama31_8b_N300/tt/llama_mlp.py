# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


class TtLlamaMLP(torch.nn.Module):
    def __init__(
        self,
        mesh_device,
        args,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        model_config,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.args = args
        self.model_config = model_config

        base_name = f"layers.{layer_num}.feed_forward"
        torch_weight = lambda name: torch.transpose(self.state_dict[f"{base_name}.{name}.weight"], -2, -1)

        if args.dummy_weights:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / (base_name + f".{name}")

        w1_w3_mem_config = args.create_dram_sharded_mem_config(args.dim, args.hidden_dim // args.num_devices)
        w2_mem_config = args.create_dram_sharded_mem_config(args.hidden_dim // args.num_devices, args.dim)

        # TODO Clean up this code. With sharding, we load the normal weights and then shard them
        as_sharded_tensor = lambda name, type, dim: ttnn.as_tensor(
            torch_weight(name[:2]),  # Grab only the wX part of the name
            dtype=type,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=dim),
            layout=self.model_config["MLP_W_LAYOUT_TILE"],
            memory_config=w2_mem_config if "w2" in name else w1_w3_mem_config,
            cache_file_name=cache_name(name),
        )

        # Sharded weights
        self.w1 = as_sharded_tensor(
            "w1_sharded", ttnn.bfloat8_b, dim=-1
        )  # bfp4 normally ok here but sub .99 pcc for llama 3.1 weights
        self.w2 = as_sharded_tensor("w2_sharded", ttnn.bfloat8_b, dim=-2)
        self.w3 = as_sharded_tensor("w3_sharded", ttnn.bfloat8_b, dim=-1)

    def forward(self, x: ttnn.Tensor, mode) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        seq_len = x.shape[-2]
        compute_kernel_config_hifi2 = self.model_config["MLP_KERNEL_CONFIG_HIFI2"]
        if mode == "decode":  # Sharded config
            pc_1 = self.model_config["DECODE_MLP_W1_W3_PRG_CONFIG"]
            pc_2 = self.model_config["DECODE_MLP_W2_PRG_CONFIG"]
            pc_3 = self.model_config["DECODE_MLP_W1_W3_PRG_CONFIG"]
            x_in = ttnn.interleaved_to_sharded(x, self.model_config["SHARDED_MLP_DECODE_INPUT_MEMCFG"])
        else:  # Update the program configs based for prefill
            x_in = x
            if seq_len >= 1024:  # Too big to compute. Set different program configs based on seqlen
                # Reshape input to to fit on device and parallelize computation
                x_in = ttnn.reshape(x_in, [1, seq_len // 1024, 1024, -1])
                pc_1 = self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG"]
                pc_2 = self.model_config["PREFILL_MLP_W2_PRG_CONFIG"]
                pc_3 = self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG"]
            else:
                pc_1 = self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG_128"](seq_len)
                pc_2 = self.model_config["PREFILL_MLP_W2_PRG_CONFIG_128"](seq_len)
                pc_3 = self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG_128"](seq_len)

        # In decode mode (seqlen <= 32) do DRAM sharded matmuls
        # These use HiFi2; this drops 1 bit of the activations but would be FLOP-bound on 12 cores with HiFi4
        w1_out = ttnn.linear(
            x_in,
            self.w1,
            compute_kernel_config=compute_kernel_config_hifi2,
            core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_1 else None,
            dtype=ttnn.bfloat16,
            program_config=pc_1,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG,
        )
        w3_out = ttnn.linear(
            x_in,
            self.w3,
            compute_kernel_config=compute_kernel_config_hifi2,
            core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_3 else None,
            dtype=ttnn.bfloat16,
            program_config=pc_3,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG,
        )

        x.deallocate(True)
        x_in.deallocate(True)

        w2_in = ttnn.multiply(
            w1_out,
            w3_out,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG,
            input_tensor_a_activation=ttnn.UnaryOpType.SILU,
            dtype=ttnn.bfloat8_b,
        )

        w3_out.deallocate(True)
        w1_out.deallocate(True)
        # This uses HiFi2 for full precision as it is dram-bound and uses bfp8 inputs
        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            compute_kernel_config=compute_kernel_config_hifi2,
            core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_2 else None,
            dtype=ttnn.bfloat16,
            program_config=pc_2,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG,
        )

        w2_in.deallocate(True)

        if mode == "decode":
            w2_out = ttnn.sharded_to_interleaved(w2_out, ttnn.L1_MEMORY_CONFIG)

        if seq_len >= 2048:  # Reshape back to intended shape
            w2_out = ttnn.reshape(w2_out, [1, 1, seq_len, -1])

        # All reduce
        if self.args.num_devices > 1:
            w2_out_gathered = ttnn.all_gather(w2_out, dim=1, num_links=1, topology=ttnn.Topology.Linear)
            w2_out_reduced = ttnn.experimental.fast_reduce_nc(
                w2_out_gathered, dims=[1], output=None, compute_kernel_config=None
            )
            return w2_out_reduced
        else:
            return w2_out
