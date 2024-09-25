# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


class TtLlamaMLP(torch.nn.Module):
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

        if args.dummy_weights:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / (base_name + f".{name}")

        # w1/w3: 4096 x 14336: width-sharded on 12 banks, 14340 over 12 banks.
        w1_w3_shard_shape = (
            4096,
            14592 // 12,
        )  # (38 shards) 14336 - padded cols to divide by 12 dram cores (and divisible by tile size of 32)
        w1_w3_shard_spec = ttnn.ShardSpec(
            args.dram_weight_grid, w1_w3_shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False
        )
        w1_w3_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, w1_w3_shard_spec
        )
        # w2: 14336 x 4096: width-sharded on 12 banks, 4096 over 12 banks.
        w2_shard_shape = (14336, 4224 // 12)  # (11 shards)  padded cols to divide by 12 dram cores
        w2_shard_spec = ttnn.ShardSpec(args.dram_weight_grid, w2_shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False)
        w2_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, w2_shard_spec)

        as_sharded_tensor = lambda name, type: ttnn.as_tensor(
            torch_weight(name[:2]),  # Grab only the wX part of the name
            dtype=type,
            device=self.device,
            layout=self.model_config["MLP_W_LAYOUT_TILE"],
            memory_config=w2_mem_config if "w2" in name else w1_w3_mem_config,
            cache_file_name=cache_name(name),
        )

        # Sharded weights
        self.w1 = as_sharded_tensor(
            "w1_dram_shard", ttnn.bfloat8_b
        )  # bfp4 normally ok here but sub .99 pcc for llama 3.1 weights
        self.w2 = as_sharded_tensor("w2_dram_shard", ttnn.bfloat8_b)
        self.w3 = as_sharded_tensor("w3_dram_shard", ttnn.bfloat8_b)

    def forward(self, x: ttnn.Tensor, mode) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        seq_len = x.shape[-2]
        compute_kernel_config = self.model_config["MLP_KERNEL_CONFIG"]
        if mode == "decode":  # Sharded config
            pc_1 = self.model_config["DECODE_MLP_W1_W3_PRG_CONFIG"]
            pc_2 = self.model_config["DECODE_MLP_W2_PRG_CONFIG"]
            pc_3 = self.model_config["DECODE_MLP_W1_W3_PRG_CONFIG"]
        else:  # Update the program configs based for prefill
            if seq_len >= 1024:  # Too big to compute. Set different program configs based on seqlen
                # Reshape input to to fit on device and parallelize computation
                x = ttnn.reshape(x, [1, seq_len // 1024, 1024, -1])
                pc_1 = self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG"]
                pc_2 = self.model_config["PREFILL_MLP_W2_PRG_CONFIG"]
                pc_3 = self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG"]
            else:
                pc_1 = self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG_128"](seq_len)
                pc_2 = self.model_config["PREFILL_MLP_W2_PRG_CONFIG_128"](seq_len)
                pc_3 = self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG_128"](seq_len)

        # In decode mode (seqlen <= 32) do DRAM sharded matmuls
        w1_out = ttnn.linear(
            x,
            self.w1,
            compute_kernel_config=compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_1 else None,
            dtype=ttnn.bfloat16,
            program_config=pc_1,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG,
        )
        w3_out = ttnn.linear(
            x,
            self.w3,
            compute_kernel_config=compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_3 else None,
            dtype=ttnn.bfloat16,
            program_config=pc_3,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG,
        )

        x.deallocate(True)

        w2_in = ttnn.multiply(
            w1_out,
            w3_out,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG,
            input_tensor_a_activation=ttnn.UnaryOpType.SILU,
            dtype=ttnn.bfloat8_b,
        )

        w3_out.deallocate(True)
        w1_out.deallocate(True)

        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            compute_kernel_config=compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_2 else None,
            dtype=ttnn.bfloat8_b,
            program_config=pc_2,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG,
        )

        w2_in.deallocate(True)

        if mode == "decode":
            w2_out = ttnn.sharded_to_interleaved(w2_out, ttnn.L1_MEMORY_CONFIG)

        if seq_len >= 2048:  # Reshape back to intended shape
            w2_out = ttnn.reshape(w2_out, [1, 1, seq_len, -1])

        return w2_out
