# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtQwenMLP(LightweightModule):
    def __init__(
        self, mesh_device, args, state_dict, weight_cache_path, layer_num, dtype, model_config, state_dict_prefix=None
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.args = args
        self.model_config = model_config
        state_dict_prefix = state_dict_prefix or args.get_state_dict_prefix(self.__class__.__name__, layer_num)
        torch_weight = lambda name: torch.transpose(self.state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)

        if args.dummy_weights:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / (state_dict_prefix + f".{name}")

        w1_w3_mem_config = args.create_dram_sharded_mem_config(args.dim, args.hidden_dim // args.num_devices)
        w2_mem_config = args.create_dram_sharded_mem_config(args.hidden_dim // args.num_devices, args.dim)

        name_dict = {
            "w1": "gate_proj",
            "w2": "down_proj",
            "w3": "up_proj",
        }
        # TODO Clean up this code. With sharding, we load the normal weights and then shard them
        as_sharded_tensor = lambda name, type, dim: ttnn.as_tensor(
            torch_weight(name_dict[name[:2]]),  # Grab only the wX part of the name
            dtype=type,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=dim),
            layout=ttnn.TILE_LAYOUT,
            memory_config=w2_mem_config if "w2" in name else w1_w3_mem_config,
            cache_file_name=cache_name(name),
        )

        # Sharded weights
        self.w1 = as_sharded_tensor(
            "w1_sharded", ttnn.bfloat4_b if self.args.is_large_model else ttnn.bfloat8_b, dim=-1
        )  # bfp4 normally ok here but sub .99 pcc for llama 3.1 weights
        self.w2 = as_sharded_tensor("w2_sharded", ttnn.bfloat8_b, dim=-2)
        self.w3 = as_sharded_tensor(
            "w3_sharded", ttnn.bfloat4_b if self.args.is_large_model else ttnn.bfloat8_b, dim=-1
        )

    def forward(self, x: ttnn.Tensor, mode) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """

        seq_len = x.shape[-2]

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
        # These use HiFi2; this drops 1 bit of the activations but would be FLOP-bound on 12 cores with HiFi4
        w1_out = ttnn.linear(
            x,
            self.w1,
            compute_kernel_config=self.args.compute_kernel_config_hifi2,
            core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_1 else None,
            dtype=ttnn.bfloat16,
            program_config=pc_1,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG,
        )

        w3_out = ttnn.linear(
            x,
            self.w3,
            compute_kernel_config=self.args.compute_kernel_config_hifi2,
            core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_3 else None,
            dtype=ttnn.bfloat16,
            program_config=pc_3,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn.deallocate(x)
        w2_in = ttnn.multiply(
            w1_out,
            w3_out,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
        )
        if (
            mode == "decode"
        ):  # TODO Add a check for a match between FF1/FF3 and FF2 memory configs. If they match avoid doing the reshard
            # Reshard w2_in to a different core_grid configuration. Avoid using ttnn.reshard() due to incompatibility with trace mode
            w2_in = ttnn.reshard(w2_in, self.model_config["SHARDED_MLP2_INPUT_MEMCFG"])

        ttnn.deallocate(w3_out)
        ttnn.deallocate(w1_out)
        # This uses HiFi2 for full precision as it is dram-bound and uses bfp8 inputs
        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            compute_kernel_config=self.args.compute_kernel_config_hifi2,
            core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_2 else None,
            dtype=ttnn.bfloat16,
            program_config=pc_2,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn.deallocate(w2_in)

        if mode == "decode":
            w2_out = ttnn.sharded_to_interleaved(
                w2_out, ttnn.L1_MEMORY_CONFIG
            )  # FIXME: When h is L1 interleaved in decoder, this call corrupts it!

        if seq_len >= 1024:  # Reshape back to intended shape
            w2_out = ttnn.reshape(w2_out, [1, 1, seq_len, -1])

        # All reduce
        if self.args.is_multichip:
            w2_out_reduced = ttnn.reduce_scatter(
                w2_out,
                dim=3,
                math_op=ttnn.ReduceType.Sum,
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG if mode == "prefill" else ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(w2_out)
            return w2_out_reduced
        else:
            return w2_out
