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
        state_dict_prefix = state_dict_prefix or args.get_state_dict_prefix(self.__class__.__name__, layer_num)
        torch_weight = lambda name: torch.transpose(self.state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)

        if args.dummy_weights:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / (state_dict_prefix + f".{name}")

        w1_w3_mem_config = self.model_config[
            "W1W3_RING_MEMCFG"
        ]  # args.create_dram_sharded_mem_config(args.dim, args.hidden_dim // args.num_devices)
        w2_mem_config = self.model_config[
            "W2_RING_MEMCFG"
        ]  # args.create_dram_sharded_mem_config(args.hidden_dim // args.num_devices, args.dim)

        # TODO Clean up this code. With sharding, we load the normal weights and then shard them
        as_sharded_tensor = lambda name, type, dim: ttnn.as_tensor(
            pad_to_next_multiple(torch_weight(name[:2])).unsqueeze(0).unsqueeze(0),  # Grab only the wX part of the name
            dtype=type,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=dim, mesh_shape=args.cluster_shape),
            layout=ttnn.TILE_LAYOUT,
            memory_config=w2_mem_config if "w2" in name else w1_w3_mem_config,
            # cache_file_name=cache_name(name) + "padded",
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

        # print(pc_1, pc_2)
        # In decode mode (seqlen <= 32) do DRAM sharded matmuls
        # These use HiFi2; this drops 1 bit of the activations but would be FLOP-bound on 12 cores with HiFi4
        # print(x.shape, self.w1.shape, self.w3.shape)
        w1_out = ttnn.linear(
            x,
            self.w1,
            compute_kernel_config=self.args.compute_kernel_config_lofi
            if self.four_bit_mlp
            else self.args.compute_kernel_config_hifi2_fp16,
            core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_1 else None,
            dtype=ttnn.bfloat8_b if TG else ttnn.bfloat16,
            program_config=pc_1,
            memory_config=self.model_config["SHARDED_FF12_OUT_RING_MEMCFG"],
            global_cb=self.prefetcher_setup.global_circular_buffer
            if self.model_config["USE_PREFETCHER"]
            else None,
        )
        ttnn.synchronize_devices(self.mesh_device)

        w3_out = ttnn.linear(
            x,
            self.w3,
            compute_kernel_config=self.args.compute_kernel_config_lofi
            if self.four_bit_mlp
            else self.args.compute_kernel_config_hifi2_fp16,
            core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_3 else None,
            dtype=ttnn.bfloat8_b if TG else ttnn.bfloat16,
            program_config=pc_3,
            memory_config=self.model_config["SHARDED_FF12_OUT_RING_MEMCFG"],
            global_cb=self.prefetcher_setup.global_circular_buffer
            if self.model_config["USE_PREFETCHER"]
            else None,
        )
        ttnn.synchronize_devices(self.mesh_device)
        ttnn.deallocate(x)

        if TG:
            # if mode == "decode" and self.dim!=8192:
            #     w1_out = ttnn.to_memory_config(w1_out, ttnn.DRAM_MEMORY_CONFIG)
            #     w3_out = ttnn.to_memory_config(w3_out, ttnn.DRAM_MEMORY_CONFIG)
            if self.dim == 8192 or mode == "prefill":
                # input_mem_cfg = w1_out.memory_config()
                # w1_out = ttnn.reduce_scatter(
                #     w1_out,
                #     dim=3,
                #     math_op=ttnn.ReduceType.Sum,
                #     num_links=self.args.num_reduce_scatter_links,
                #     cluster_axis=1,
                #     mesh_device=self.mesh_device,
                #     topology=ttnn.Topology.Linear,
                #     memory_config=self.model_config["FF1_OUT_REDUCE_SCATTER_MEMCFG"] if mode == "decode" else None,
                # )
                # w3_out = ttnn.reduce_scatter(
                #     w3_out,
                #     dim=3,
                #     math_op=ttnn.ReduceType.Sum,
                #     num_links=1,
                #     cluster_axis=1,
                #     mesh_device=self.mesh_device,
                #     topology=ttnn.Topology.Linear,
                #     memory_config=self.model_config["FF1_OUT_REDUCE_SCATTER_MEMCFG"] if mode == "decode" else None,
                # )
                # print(w1_out.shape, w3_out.shape)
                # All reduce W1
                w1_out = ttnn.to_memory_config(w1_out, ttnn.DRAM_MEMORY_CONFIG)
                w1_out_reduced = self.tt_ccl.line_reduce_scatter(
                    w1_out, dim=3, cluster_axis=1, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )

                ttnn.synchronize_devices(self.mesh_device)
                ttnn.deallocate(w1_out)
                w1_out_reduced = ttnn.to_memory_config(
                    w1_out_reduced, self.model_config["SHARDED_FF12_PRE_MUL_RING_REDUCE_MEMCFG"]
                )

                # All reduce W3
                w3_out = ttnn.to_memory_config(w3_out, ttnn.DRAM_MEMORY_CONFIG)
                w3_out_reduced = self.tt_ccl.line_reduce_scatter(
                    w3_out, dim=3, cluster_axis=1, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )

                ttnn.synchronize_devices(self.mesh_device)
                ttnn.deallocate(w3_out)
                w3_out_reduced = ttnn.to_memory_config(
                    w3_out_reduced, self.model_config["SHARDED_FF12_PRE_MUL_RING_REDUCE_MEMCFG"]
                )

            else:
                w1_out = tt_all_reduce(
                    w1_out,
                    self.mesh_device,
                    cluster_axis=1,
                    num_all_gather_links=2,
                    sharded=True if mode == "decode" else False,
                    memory_config=self.model_config["FF1_OUT_GATHERED_MEMCFG"] if mode == "decode" else None,
                )
                w3_out = tt_all_reduce(
                    w3_out,
                    self.mesh_device,
                    cluster_axis=1,
                    num_all_gather_links=2,
                    sharded=True if mode == "decode" else False,
                    memory_config=self.model_config["FF1_OUT_GATHERED_MEMCFG"] if mode == "decode" else None,
                )

        w2_in = ttnn.mul(
            w1_out_reduced,
            w3_out_reduced,
            input_tensor_a_activation=ttnn.UnaryOpType.SILU,
            dtype=ttnn.bfloat8_b,
            memory_config=w1_out.memory_config(),
        )
        ttnn.synchronize_devices(self.mesh_device)

        # All reduce W3
        w2_in = ttnn.to_memory_config(w2_in, ttnn.DRAM_MEMORY_CONFIG)
        w2_in_gathered = self.tt_ccl.line_all_gather(
            w2_in, dim=3, cluster_axis=1, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        ttnn.synchronize_devices(self.mesh_device)
        ttnn.deallocate(w2_in)
        w2_in_gathered = ttnn.to_memory_config(w2_in_gathered, self.model_config["FF2_IN_RING_MEMCFG"])

        # print("mul done", w2_in.shape)

        if mode == "decode" and not TG:
            # w2 may use a different core grid, this is a no-op if they already match
            w2_in = ttnn.to_memory_config(w2_in, self.model_config["SHARDED_MLP2_INPUT_MEMCFG"])

        ttnn.deallocate(w3_out)
        ttnn.deallocate(w1_out_reduced)

        # if TG and (self.dim == 8192 or mode == "prefill"):
        # w2_in = ttnn.all_gather(
        #     w2_in,
        #     3,
        #     num_links=2,
        #     cluster_axis=1,
        #     mesh_device=self.mesh_device,
        #     topology=ttnn.Topology.Linear,
        #     memory_config=input_mem_cfg,
        # )
        # w2_in_torch = ttnn.to_torch(
        #     w2_in,
        #     mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(3, 0), mesh_shape=(8, 4)),
        # )
        # #print(w2_in_torch.shape)
        # w2_in_torch_gather = torch.cat([w2_in_torch] * 4, dim=3)
        # #print(w2_in_torch_gather.shape)
        # w2_in = ttnn.as_tensor(
        #     w2_in_torch_gather,
        #     dtype=ttnn.bfloat16,
        #     device=self.mesh_device,
        #     mesh_mapper=ttnn.ShardTensor2dMesh(
        #         mesh_device=self.mesh_device, dims=(3, None), mesh_shape=list(self.mesh_device.shape)
        #     ),
        #     layout=ttnn.TILE_LAYOUT,
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        # )
        # #print(w2_in)
        # if mode == "decode":
        #     w2_in_gathered = ttnn.to_memory_config(w2_in_gathered, self.model_config["FF2_IN_RING_MEMCFG"])

        # print("w2_in", w2_in)
        w2_out = ttnn.linear(
            w2_in_gathered,
            self.w2,
            compute_kernel_config=self.args.compute_kernel_config_hifi2_fp16,
            dtype=self.args.ccl_dtype if TG else ttnn.bfloat16,
            program_config=pc_2,
            memory_config=(self.model_config["FF2_OUT_RING_MEMCFG"] if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG)
            if TG
            else w2_in.memory_config(),
            core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_2 else None,
            global_cb=self.prefetcher_setup.global_circular_buffer
            if self.model_config["USE_PREFETCHER"]
            else None,
        )
        ttnn.synchronize_devices(self.mesh_device)
        ttnn.deallocate(w2_in_gathered)
        # if mode == "decode" and not TG:
        #     w2_out = ttnn.sharded_to_interleaved(w2_out, ttnn.DRAM_MEMORY_CONFIG)
        # w2_out_reduced = tt_all_reduce(
        #     w2_out,
        #     self.mesh_device,
        #     cluster_axis=0,
        #     dim=0 if (TG and self.dim < 8192) else 3,
        #     num_reduce_scatter_links=self.args.num_reduce_scatter_links,
        #     num_all_gather_links=self.args.num_all_gather_links,
        #     sharded=(mode == "decode"),
        #     memory_config=(self.model_config["FF2_OUT_REDUCE_SCATTER_MEMCFG"] if TG else w2_out.memory_config())
        #     if mode == "decode"
        #     else ttnn.DRAM_MEMORY_CONFIG,
        #     dtype=self.args.ccl_dtype,
        #     use_composite=True if self.dim == 8192 else False,
        # )
        w2_out = ttnn.to_memory_config(w2_out, ttnn.DRAM_MEMORY_CONFIG)
        w2_out_reduced = self.tt_ccl.line_all_reduce(
            w2_out, cluster_axis=0, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        ttnn.synchronize_devices(self.mesh_device)
        ttnn.deallocate(w2_out)

        # Ensure dim 0 and 1 are 1
        # original_shape = w2_out_reduced.shape
        # w2_out_reduced = ttnn.reshape(
        #     w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        # )
        # if mode == "decode":
        #     w2_out_reduced = ttnn.to_memory_config(
        #         w2_out_reduced,
        #         self.model_config["SHARDED_ATTN_INPUT_MEMCFG"] if TG else self.model_config["DECODE_RESIDUAL_MEMCFG"],
        #     )

        # ttnn.deallocate(w2_out)
        return w2_out_reduced
