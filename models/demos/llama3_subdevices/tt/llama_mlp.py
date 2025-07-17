# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule
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
            cache_name = lambda name: weight_cache_path / (state_dict_prefix + f".{name}" + "prefetcher")

        w1_w3_mem_config = self.model_config[
            "W1W3_RING_MEMCFG"
        ]  # args.create_dram_sharded_mem_config(args.dim, args.hidden_dim // args.num_devices)
        w2_mem_config = self.model_config[
            "W2_RING_MEMCFG"
        ]  # args.create_dram_sharded_mem_config(args.hidden_dim // args.num_devices, args.dim)
        as_sharded_tensor = lambda name, type, dim: ttnn.as_tensor(
            torch_weight(name[:2]).unsqueeze(0).unsqueeze(0),  # Grab only the wX part of the name
            dtype=type,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=dim, mesh_shape=args.cluster_shape),
            layout=ttnn.TILE_LAYOUT,
            memory_config=w2_mem_config if "w2" in name else w1_w3_mem_config,
            cache_file_name=cache_name(name),
        )

        as_interleaved_tensor = lambda name, type, dim: ttnn.as_tensor(
            torch_weight(name[:2]).unsqueeze(0).unsqueeze(0),  # Grab only the wX part of the name
            dtype=type,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=dim, mesh_shape=args.cluster_shape),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(name),
        )

        self.four_bit_mlp = args.optimizations.bfp4_mlp

        # Sharded weights
        w1_dim = (-1, -2) if args.is_galaxy else (-2, -1)
        w2_dim = (-2, -1) if args.is_galaxy else (-1, -2)

        # sharded
        self.w1 = as_sharded_tensor(
            "w1_sharded", ttnn.bfloat4_b if self.four_bit_mlp else ttnn.bfloat8_b, dim=w1_dim
        )  # bfp4 normally ok here but sub .99 pcc for llama 3.1 weights
        self.w2 = as_sharded_tensor("w2_sharded", ttnn.bfloat8_b, dim=w2_dim)
        self.w3 = as_sharded_tensor("w3_sharded", ttnn.bfloat4_b if self.four_bit_mlp else ttnn.bfloat8_b, dim=w1_dim)

        self.w1_interleaved = as_interleaved_tensor(
            "w1_interleaved", ttnn.bfloat4_b if self.four_bit_mlp else ttnn.bfloat8_b, dim=w1_dim
        )
        self.w2_interleaved = as_interleaved_tensor("w2_interleaved", ttnn.bfloat8_b, dim=w2_dim)
        self.w3_interleaved = as_interleaved_tensor(
            "w3_interleaved", ttnn.bfloat4_b if self.four_bit_mlp else ttnn.bfloat8_b, dim=w1_dim
        )

        if tt_ccl.mode == "decode":
            self.prefetch(prefetcher_setup, tt_ccl)

    def prefetch(self, prefetcher_setup, tt_ccl):
        self.prefetcher_setup = prefetcher_setup
        if tt_ccl.mode == "decode":
            self.prefetcher_setup.insert_tensor(self.w1)
            self.prefetcher_setup.insert_tensor(self.w3)
            self.prefetcher_setup.insert_tensor(self.w2)
        self.tt_ccl = tt_ccl

    def forward(self, x: ttnn.Tensor, mode) -> ttnn.Tensor:
        if mode == "prefill":
            return self.forward_prefill(x, mode)

        pc_1 = self.model_config["FF1_3_TG_RING_PROGCFG"]
        pc_2 = self.model_config["FF2_TG_RING_PROGCFG"]
        pc_3 = self.model_config["FF1_3_TG_RING_PROGCFG"]

        w1_out = ttnn.linear(
            x,
            self.w1,
            compute_kernel_config=self.args.compute_kernel_config_lofi
            if self.four_bit_mlp
            else self.args.compute_kernel_config_hifi2,
            dtype=ttnn.bfloat8_b,
            program_config=pc_1,
            memory_config=self.model_config["SHARDED_FF12_OUT_RING_MEMCFG"],
            global_cb=self.prefetcher_setup.global_circular_buffer if self.model_config["USE_PREFETCHER"] else None,
            sub_device_id=self.prefetcher_setup.worker_sub_device_id if mode == "decode" else None,
        )
        w1_out_reduced, w3_out = self.tt_ccl.matmul_line_reduce_scatter(
            x,
            self.w3,
            w1_out,
            cluster_axis=1,
            num_links=self.model_config["GALAXY_NUM_LINKS"],
            RS_memory_config=self.model_config["REDUCE_SCATTER_OUT_MEMCFG"],
            compute_kernel_config=self.args.compute_kernel_config_lofi
            if self.four_bit_mlp
            else self.args.compute_kernel_config_hifi2,
            dtype=ttnn.bfloat8_b,
            program_config=pc_3,
            memory_config=self.model_config["SHARDED_FF12_OUT_RING_MEMCFG"],
            global_cb=self.prefetcher_setup.global_circular_buffer if self.model_config["USE_PREFETCHER"] else None,
            sub_device_id=self.prefetcher_setup.worker_sub_device_id if mode == "decode" else None,
            use_noc1_only=False,
        )
        ttnn.deallocate(x)
        try:
            w3_out_reduced = self.tt_ccl.line_reduce_scatter(
                w3_out,
                cluster_axis=1,
                num_links=self.model_config["GALAXY_NUM_LINKS"],
                memory_config=self.model_config["REDUCE_SCATTER_OUT_MEMCFG"],
                use_noc1_only=False,
            )
        except Exception as e:
            print(e)
            self.tt_ccl.close()

        ff1ff3 = ttnn.mul(
            w1_out_reduced,
            w3_out_reduced,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat8_b,
            memory_config=self.model_config["REDUCE_SCATTER_OUT_MEMCFG"],
        )

        # print("eltwise mul", w2_in)

        ttnn.deallocate(w3_out_reduced)
        ttnn.deallocate(w1_out_reduced)

        w2_in = self.tt_ccl.line_all_gather(
            ff1ff3,
            dim=3,
            cluster_axis=1,
            num_links=self.model_config["GALAXY_NUM_LINKS"],
            memory_config=self.model_config["FF2_IN_RING_MEMCFG"],
            buffer_key="BINARY_MUL",
            use_optimal_ccl_for_llama=False if mode == "prefill" else True,
        )
        ttnn.deallocate(ff1ff3)

        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            compute_kernel_config=self.args.compute_kernel_config_hifi2,
            dtype=ttnn.bfloat8_b,
            program_config=pc_2,
            memory_config=self.model_config["FF2_OUT_RING_MEMCFG"],
            core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_2 else None,
            global_cb=self.prefetcher_setup.global_circular_buffer if self.model_config["USE_PREFETCHER"] else None,
            sub_device_id=self.prefetcher_setup.worker_sub_device_id if mode == "decode" else None,
        )

        w2_out_reduced = self.tt_ccl.line_all_reduce(
            w2_out,
            cluster_axis=0,
            num_links=self.model_config["GALAXY_NUM_LINKS"],
            memory_config=self.model_config["DECODE_RESIDUAL_MEMCFG"],
            use_optimal_ccl_for_llama=True,
        )

        ttnn.deallocate(w2_out)

        return w2_out_reduced

    def forward_prefill(self, x: ttnn.Tensor, mode) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        seq_len = x.shape[-2]
        use_w1_w3_interleaved = seq_len >= 4096 or seq_len == 128
        pc_1 = self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG"](seq_len, use_w1_w3_interleaved)
        pc_2 = self.model_config["PREFILL_MLP_W2_PRG_CONFIG"](seq_len)
        pc_3 = self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG"](seq_len, use_w1_w3_interleaved)

        if 1024 <= seq_len < 4096:
            x = ttnn.reshape(x, (1, seq_len // 1024, 1024, -1))
        w1_out = ttnn.linear(
            x,
            self.w1_interleaved if use_w1_w3_interleaved else self.w1,
            compute_kernel_config=(
                self.args.compute_kernel_config_lofi
                if self.four_bit_mlp
                else self.args.compute_kernel_config_hifi2_fp16
            ),
            dtype=ttnn.bfloat8_b,
            program_config=pc_1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        w1_out_reduced = self.tt_ccl.line_reduce_scatter(
            w1_out, cluster_axis=1, num_links=3, memory_config=w1_out.memory_config(), buffer_key="FF1", dim=3
        )
        ttnn.deallocate(w1_out)
        w3_out = ttnn.linear(
            x,
            self.w3_interleaved if use_w1_w3_interleaved else self.w3,
            compute_kernel_config=(
                self.args.compute_kernel_config_lofi
                if self.four_bit_mlp
                else self.args.compute_kernel_config_hifi2_fp16
            ),
            dtype=ttnn.bfloat8_b,
            program_config=pc_3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)
        w3_out_reduced = self.tt_ccl.line_reduce_scatter(
            w3_out, cluster_axis=1, num_links=3, memory_config=w3_out.memory_config(), buffer_key="FF3", dim=3
        )
        ttnn.deallocate(w3_out)
        w2_in = ttnn.mul(
            w1_out_reduced,
            w3_out_reduced,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat8_b,
            memory_config=w1_out.memory_config(),
        )
        w2_in_gathered = self.tt_ccl.line_all_gather(
            w2_in, cluster_axis=1, num_links=3, memory_config=w3_out.memory_config(), buffer_key="FF3", dim=3
        )
        ttnn.deallocate(w2_in)
        # ttnn.deallocate(w3_out)
        # ttnn.deallocate(w1_out)

        w2_out = ttnn.linear(
            w2_in_gathered,
            self.w2_interleaved,
            compute_kernel_config=self.args.compute_kernel_config_hifi2_fp16,
            dtype=ttnn.bfloat8_b,
            program_config=pc_2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn.deallocate(w2_in)
        w2_out_reduced = self.tt_ccl.line_all_reduce(
            w2_out, cluster_axis=0, num_links=3, memory_config=ttnn.DRAM_MEMORY_CONFIG, buffer_key="FF2"
        )
        ttnn.deallocate(w2_out)
        if 1024 <= seq_len < 4096:
            original_shape = w2_out_reduced.shape
            w2_out_reduced = ttnn.reshape(
                w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
            )

        # ttnn.deallocate(w2_out)
        return w2_out_reduced
