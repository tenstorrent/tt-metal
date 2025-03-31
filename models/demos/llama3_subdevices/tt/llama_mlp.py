# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.llama3_subdevices.tt.llama_ccl import tt_all_reduce
import torch.nn.functional as F


def pad_to_next_multiple(tensor):
    # Get the current size of the last two dimensions
    height, width = tensor.shape[-2], tensor.shape[-1]
    if height < 9216:
        pad_height = 9216 - height
        pad_width = 3840 * 8 - width
    else:
        pad_height = 3840 * 8 - height
        pad_width = 9216 - width

    # Apply padding (padding is in the format: (left, right, top, bottom))
    padding = (0, pad_width, 0, pad_height)
    padded_tensor = F.pad(tensor, padding, mode="constant", value=0)  # You can change `value` for a different pad value

    return padded_tensor


class TtLlamaMLP(LightweightModule):
    def __init__(
        self,
        mesh_device,
        args,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        model_config,
        state_dict_prefix=None,
        prefetcher_setup=None,
        tt_ccl=None,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.args = args
        self.dim = args.dim
        self.model_config = model_config
        self.prefetcher_setup = prefetcher_setup
        self.tt_ccl = tt_ccl
        self.worker_sub_device_id = prefetcher_setup.worker_sub_device_id
        state_dict_prefix = state_dict_prefix or args.get_state_dict_prefix(self.__class__.__name__, layer_num)
        torch_weight = lambda name: torch.transpose(self.state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)

        if args.dummy_weights:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / (state_dict_prefix + f".{name}" + "prefetcher")

        w1_w3_mem_config = self.model_config[
            "W1W3_RING_MEMCFG"
        ]  # args.create_dram_sharded_mem_config(args.dim, args.hidden_dim // args.num_devices)
        w2_mem_config = self.model_config[
            "W2_RING_MEMCFG"
        ]  # args.create_dram_sharded_mem_config(args.hidden_dim // args.num_devices, args.dim)

        # TODO Clean up this code. With sharding, we load the normal weights and then shard them
        as_sharded_tensor = lambda name, type, dim: ttnn.as_tensor(
            torch_weight(name[:2]).unsqueeze(0).unsqueeze(0),  # Grab only the wX part of the name
            dtype=type,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=dim, mesh_shape=args.cluster_shape),
            layout=ttnn.TILE_LAYOUT,
            memory_config=w2_mem_config if "w2" in name else w1_w3_mem_config,
            cache_file_name=cache_name(name),
        )

        self.four_bit_mlp = args.optimizations.bfp4_mlp

        # Sharded weights
        w1_dim = (-1, -2) if args.is_galaxy else (-2, -1)
        w2_dim = (-2, -1) if args.is_galaxy else (-1, -2)

        self.w1 = as_sharded_tensor(
            "w1_sharded", ttnn.bfloat4_b if self.four_bit_mlp else ttnn.bfloat8_b, dim=w1_dim
        )  # bfp4 normally ok here but sub .99 pcc for llama 3.1 weights
        if self.model_config["USE_PREFETCHER"]:
            self.prefetcher_setup.insert_tensor(self.w1)
        self.w2 = as_sharded_tensor("w2_sharded", ttnn.bfloat8_b, dim=w2_dim)
        self.w3 = as_sharded_tensor("w3_sharded", ttnn.bfloat4_b if self.four_bit_mlp else ttnn.bfloat8_b, dim=w1_dim)
        if self.model_config["USE_PREFETCHER"]:
            self.prefetcher_setup.insert_tensor(self.w3)
        if self.model_config["USE_PREFETCHER"]:
            self.prefetcher_setup.insert_tensor(self.w2)
        # [:2304, :3840]

    def forward(self, x: ttnn.Tensor, mode) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        # self.w1 = ttnn.to_memory_config(self.w1, self.model_config["W1W3_RING_MEMCFG"])
        # self.w2 = ttnn.to_memory_config(self.w2, self.model_config["W2_RING_MEMCFG"])
        # self.w3 = ttnn.to_memory_config(self.w3, self.model_config["W1W3_RING_MEMCFG"])

        # x = ttnn.to_memory_config(x, self.model_config["SHARDED_FF12_RING_MEMCFG"])
        seq_len = x.shape[-2]
        TG = self.args.is_galaxy

        if mode == "decode":  # Sharded config
            if TG:  # TODO: Fix this when TG supports DRAM sharded matmuls
                pc_1 = self.model_config["FF1_3_TG_RING_PROGCFG"] if self.dim >= 4096 else None
                pc_2 = self.model_config["FF2_TG_RING_PROGCFG"] if self.dim >= 4096 else None
                pc_3 = self.model_config["FF1_3_TG_RING_PROGCFG"] if self.dim >= 4096 else None
            else:
                pc_1 = self.model_config["DECODE_MLP_W1_W3_PRG_CONFIG"]
                pc_2 = self.model_config["DECODE_MLP_W2_PRG_CONFIG"]
                pc_3 = self.model_config["DECODE_MLP_W1_W3_PRG_CONFIG"]
        else:  # Update the program configs based for prefill
            if seq_len >= 1024:
                # Reshape input to to fit on device and parallelize computation
                x = ttnn.reshape(x, [1, seq_len // 1024, 1024, -1])
            pc_1 = self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG"](seq_len)
            pc_2 = self.model_config["PREFILL_MLP_W2_PRG_CONFIG"](seq_len)
            pc_3 = self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG"](seq_len)

        # #print(pc_1, pc_2)
        # In decode mode (seqlen <= 32) do DRAM sharded matmuls
        # These use HiFi2; this drops 1 bit of the activations but would be FLOP-bound on 12 cores with HiFi4
        # #print(x.shape, self.w1.shape, self.w3.shape)
        w1_out = ttnn.linear(
            x,
            self.w1,
            compute_kernel_config=self.args.compute_kernel_config_lofi
            if self.four_bit_mlp
            else self.args.compute_kernel_config_hifi2,
            core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_1 else None,
            dtype=ttnn.bfloat8_b if TG else ttnn.bfloat16,
            program_config=pc_1,
            memory_config=self.model_config["SHARDED_FF12_OUT_RING_MEMCFG"],
            global_cb=self.prefetcher_setup.global_circular_buffer if self.model_config["USE_PREFETCHER"] else None,
            sub_device_id=self.worker_sub_device_id if mode == "decode" else None,
        )

        w1_out_reduced = self.tt_ccl.line_all_reduce(
            w1_out,
            cluster_axis=1,
            num_links=3,
            memory_config=self.model_config["MUL_IN_MEMCFG"],
        )

        w3_out = ttnn.linear(
            x,
            self.w3,
            compute_kernel_config=self.args.compute_kernel_config_lofi
            if self.four_bit_mlp
            else self.args.compute_kernel_config_hifi2,
            core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_3 else None,
            dtype=ttnn.bfloat8_b if TG else ttnn.bfloat16,
            program_config=pc_3,
            memory_config=self.model_config["SHARDED_FF12_OUT_RING_MEMCFG"],
            global_cb=self.prefetcher_setup.global_circular_buffer if self.model_config["USE_PREFETCHER"] else None,
            sub_device_id=self.worker_sub_device_id if mode == "decode" else None,
        )
        ttnn.deallocate(x)
        # print("linear", w3_out)
        try:
            w3_out_reduced = self.tt_ccl.line_all_reduce(
                w3_out,
                cluster_axis=1,
                num_links=3,
                memory_config=self.model_config["MUL_IN_MEMCFG"],
            )

            # print("reduced", w1_out_reduced)
        except Exception as e:
            # print(e)
            self.tt_ccl.close()

        w2_in = ttnn.mul(
            w1_out_reduced,
            w3_out_reduced,
            input_tensor_a_activation=ttnn.UnaryOpType.SILU,
            dtype=ttnn.bfloat16,
            memory_config=self.model_config["MUL_IN_MEMCFG"],
        )

        w2_in = ttnn.to_memory_config(w2_in, self.model_config["FF2_IN_RING_MEMCFG"])

        # print("eltwise mul", w2_in)

        ttnn.deallocate(w3_out_reduced)
        ttnn.deallocate(w1_out_reduced)

        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            compute_kernel_config=self.args.compute_kernel_config_hifi2,
            dtype=ttnn.bfloat8_b,
            program_config=pc_2,
            memory_config=(self.model_config["FF2_OUT_RING_MEMCFG"] if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG)
            if TG
            else w2_in.memory_config(),
            core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_2 else None,
            global_cb=self.prefetcher_setup.global_circular_buffer if self.model_config["USE_PREFETCHER"] else None,
            sub_device_id=self.worker_sub_device_id if mode == "decode" else None,
        )
        # print("linear", w2_out)
        ttnn.deallocate(w2_in)

        w2_out_reduced = self.tt_ccl.line_all_reduce(
            w2_out, cluster_axis=0, num_links=3, memory_config=self.model_config["DECODE_RESIDUAL_MEMCFG"]
        )
        # print("reduced", w2_out_reduced)

        ttnn.deallocate(w2_out)

        return w2_out_reduced
