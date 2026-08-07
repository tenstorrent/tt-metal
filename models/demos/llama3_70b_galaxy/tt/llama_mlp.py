# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

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
        self.layer_num = layer_num
        self.args = args
        self.dim = args.dim
        self.model_config = model_config
        # Single source of truth for the prefetcher gate (Wormhole/TG True, Blackhole bring-up False).
        self.use_prefetcher = args.use_prefetcher
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
            dtype=type if not args.is_qwen else ttnn.bfloat16,
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
        w1_dim = (-1, -2)
        w2_dim = (-2, -1)

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

        if tt_ccl.mode == "decode" and self.use_prefetcher:
            self.prefetch(prefetcher_setup, tt_ccl)

    def prefetch(self, prefetcher_setup, tt_ccl):
        self.prefetcher_setup = prefetcher_setup
        if tt_ccl.mode == "decode" and self.prefetcher_setup is not None:
            self.prefetcher_setup.insert_tensor(self.w1)
            self.prefetcher_setup.insert_tensor(self.w3)
            self.prefetcher_setup.insert_tensor(self.w2)
        self.tt_ccl = tt_ccl

    def forward(self, x: ttnn.Tensor, mode, batch_size=1, return_intermediates=False):
        if mode == "prefill":
            return self.forward_prefill(x, batch_size=batch_size)

        intermediates = {} if return_intermediates else None

        pc_1_3 = self.model_config["FF1_3_TG_RING_PROGCFG"]
        pc_2 = self.model_config["FF2_TG_RING_PROGCFG"]

        if self.args.is_blackhole and not self.use_prefetcher:
            # The MLP input arrives width-sharded on the prefetcher ring cores (core column x=6).
            # The auto-selected 1D matmul compute grid does not span that column, so current main
            # rejects the sharded input ("Tensor shard spec grid ... must lie within compute grid").
            # Move the input to DRAM interleaved first (mirrors the FF2 no-prefetch path below).
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            # BH Qwen no-prefetch: emit FF1/FF3 to DRAM interleaved (program_config=None) instead of
            # the prefetcher L1 ring memcfg. The ring memcfg's shard grid uses core column x=6, which
            # is outside the auto-selected 1D-matmul compute grid; current main validates this
            # (matmul_device_operation.cpp: "output shard grid ... must lie within extent") and fails.
            # Going through DRAM also matches the attention QKV / FF2 no-prefetch paths and avoids the
            # padded-L1 bf8 readback garbage that capped MLP PCC. The following device reduce_scatter
            # consumes DRAM interleaved cleanly.
            w1_out = ttnn.linear(
                x,
                self.w1_interleaved,
                compute_kernel_config=self.args.compute_kernel_config_lofi
                if self.four_bit_mlp
                else self.args.compute_kernel_config_hifi2,
                dtype=ttnn.bfloat8_b,
                program_config=None,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                global_cb=None,
                sub_device_id=None,
            )
            w3_out = ttnn.linear(
                x,
                self.w3_interleaved,
                compute_kernel_config=self.args.compute_kernel_config_lofi
                if self.four_bit_mlp
                else self.args.compute_kernel_config_hifi2,
                dtype=ttnn.bfloat8_b,
                program_config=None,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                global_cb=None,
                sub_device_id=None,
            )
            ttnn.deallocate(x)
            w1_out_reduced = self.tt_ccl.line_reduce_scatter(
                w1_out,
                cluster_axis=1,
                num_links=self.model_config["GALAXY_NUM_LINKS"],
                memory_config=self.model_config["REDUCE_SCATTER_OUT_MEMCFG"],
                use_noc1_only=False,
            )
            ttnn.deallocate(w1_out)
        else:
            w1_out_reduced, w3_out = self.tt_ccl.double_matmul_line_reduce_scatter(
                x,
                self.w1,
                self.w3,
                cluster_axis=1,
                num_links=self.model_config["GALAXY_NUM_LINKS"],
                RS_memory_config=self.model_config["REDUCE_SCATTER_OUT_MEMCFG"],
                compute_kernel_config=self.args.compute_kernel_config_lofi
                if self.four_bit_mlp
                else self.args.compute_kernel_config_hifi2,
                dtype=ttnn.bfloat8_b,
                program_config=pc_1_3,
                memory_config=self.model_config["SHARDED_FF12_OUT_RING_MEMCFG"],
                global_cb=self.prefetcher_setup.global_circular_buffer if self.use_prefetcher else None,
                sub_device_id=(
                    self.prefetcher_setup.worker_sub_device_id if mode == "decode" and self.use_prefetcher else None
                ),
                use_noc1_only=False,
            )
            ttnn.deallocate(x)

        w3_out_reduced = self.tt_ccl.line_reduce_scatter(
            w3_out,
            cluster_axis=1,
            num_links=self.model_config["GALAXY_NUM_LINKS"],
            memory_config=self.model_config["REDUCE_SCATTER_OUT_MEMCFG"],
            use_noc1_only=False,
        )
        ttnn.deallocate(w3_out)
        if return_intermediates:
            intermediates["ff1_reduced"] = w1_out_reduced
            intermediates["ff3_reduced"] = w3_out_reduced

        ff1ff3 = ttnn.mul(
            w1_out_reduced,
            w3_out_reduced,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat8_b,
            memory_config=self.model_config["REDUCE_SCATTER_OUT_MEMCFG"],
        )

        if return_intermediates:
            intermediates["activation"] = ff1ff3
        else:
            ttnn.deallocate(w3_out_reduced)
            ttnn.deallocate(w1_out_reduced)

        w2_in = self.tt_ccl.line_all_gather(
            ff1ff3,
            dim=3,
            cluster_axis=1,
            num_links=self.model_config["GALAXY_NUM_LINKS"],
            memory_config=self.model_config["FF2_IN_RING_MEMCFG"],
            buffer_key="BINARY_MUL",
            use_optimal_ccl_for_llama=(mode == "decode" and (self.use_prefetcher or self.args.is_qwen)),
        )
        if return_intermediates:
            intermediates["ff2_input"] = w2_in

        if not return_intermediates:
            ttnn.deallocate(ff1ff3)

        use_bh_decode_no_pf = self.args.is_blackhole and mode == "decode" and not self.use_prefetcher
        w2_in_for_matmul = w2_in
        pc_2_for_matmul = pc_2
        if use_bh_decode_no_pf:
            # BH decode no-prefetch (Qwen and Llama): use the interleaved FF2 path to avoid the
            # sharded cross-device channel ordering assumptions in the W2 matmul. Feeding the
            # ring-sharded all_gather output straight into the interleaved w2 matmul mismatches the
            # per-device channel order and produces garbage (MLP decode PCC ~0), so bring it into
            # DRAM interleaved first.
            w2_in_for_matmul = ttnn.to_memory_config(w2_in, ttnn.DRAM_MEMORY_CONFIG)
            pc_2_for_matmul = None
            if not return_intermediates:
                ttnn.deallocate(w2_in)

        if use_bh_decode_no_pf:
            w2_out = ttnn.linear(
                w2_in_for_matmul,
                self.w2_interleaved,
                compute_kernel_config=self.args.compute_kernel_config_hifi2,
                dtype=ttnn.bfloat8_b,
                program_config=None,
                memory_config=None,
                core_grid=None,
                global_cb=None,
                sub_device_id=None,
            )
        else:
            w2_out = ttnn.linear(
                w2_in_for_matmul,
                self.w2,
                compute_kernel_config=self.args.compute_kernel_config_hifi2,
                dtype=ttnn.bfloat8_b,
                program_config=pc_2_for_matmul,
                memory_config=self.model_config["FF2_OUT_RING_MEMCFG"],
                core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_2_for_matmul else None,
                global_cb=self.prefetcher_setup.global_circular_buffer if self.use_prefetcher else None,
                sub_device_id=(
                    self.prefetcher_setup.worker_sub_device_id if mode == "decode" and self.use_prefetcher else None
                ),
            )
        if use_bh_decode_no_pf and w2_out.memory_config().buffer_type == ttnn.BufferType.DRAM:
            # The FF2 no-prefetch matmul emits to DRAM, but the device all_reduce
            # (ttnn.experimental.all_reduce_async) rejects DRAM input on Blackhole. Bring it into the
            # all_reduce's own L1 layout (same per-device shape as the all_reduce output).
            w2_out = ttnn.to_memory_config(w2_out, self.model_config["DECODE_RESIDUAL_MEMCFG"])
        w2_out_reduced = self.tt_ccl.line_all_reduce(
            w2_out,
            cluster_axis=0,
            num_links=self.model_config["GALAXY_NUM_LINKS"],
            memory_config=self.model_config["DECODE_RESIDUAL_MEMCFG"],
            use_optimal_ccl_for_llama=True,
        )
        if return_intermediates:
            intermediates["ff2_pre_allreduce"] = w2_out
            intermediates["ff2_output"] = w2_out_reduced
            return w2_out_reduced, intermediates
        ttnn.deallocate(w2_out)

        return w2_out_reduced

    def forward_prefill(self, x: ttnn.Tensor, batch_size=1) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        seq_len = x.shape[-2]
        # Wormhole keeps main's fixed prefill link count (3); Blackhole uses the mesh link budget.
        prefill_num_links = self.model_config["GALAXY_NUM_LINKS"] if self.args.is_blackhole else 3
        use_w1_w3_interleaved = (seq_len >= 4096 or seq_len == 128) if not self.args.is_qwen else True
        short_lens_pc_1_3 = self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG"](seq_len, use_w1_w3_interleaved)
        short_lens_pc_2 = self.model_config["PREFILL_MLP_W2_PRG_CONFIG"](seq_len)

        minimal_pc_1_3 = self.model_config["PREFILL_FF1_FF3_MINIMAL_MATMUL_CONFIG"](seq_len)
        minimal_pc_2 = self.model_config["PREFILL_FF2_MINIMAL_MATMUL_CONFIG"](seq_len)

        if 1024 <= seq_len < 4096:
            x = ttnn.reshape(x, (1, seq_len // 1024, 1024, -1))

        # For shorter sequence lengths use the original matmul since it performs better than the minimal matmul
        if seq_len < 4096 or batch_size > 1:
            w1_out = ttnn.linear(
                x,
                self.w1_interleaved if use_w1_w3_interleaved else self.w1,
                compute_kernel_config=(
                    self.args.compute_kernel_config_lofi
                    if self.four_bit_mlp
                    else self.args.compute_kernel_config_hifi2_fp16
                ),
                dtype=ttnn.bfloat8_b,
                program_config=short_lens_pc_1_3,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            w1_out = ttnn.experimental.minimal_matmul(
                input_tensor=x,
                weight_tensor=self.w1_interleaved if use_w1_w3_interleaved else self.w1,
                config=minimal_pc_1_3,
                compute_kernel_config=self.args.compute_kernel_config_lofi,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        w1_out_reduced = self.tt_ccl.line_reduce_scatter(
            w1_out,
            cluster_axis=1,
            num_links=prefill_num_links,
            memory_config=w1_out.memory_config(),
            buffer_key="FF1",
            dim=3,
            batch_size=batch_size,
        )
        ttnn.deallocate(w1_out)

        # For shorter sequence lengths use the original matmul since it performs better than the minimal matmul
        if seq_len < 4096 or batch_size > 1:
            w3_out = ttnn.linear(
                x,
                self.w3_interleaved if use_w1_w3_interleaved else self.w3,
                compute_kernel_config=(
                    self.args.compute_kernel_config_lofi
                    if self.four_bit_mlp
                    else self.args.compute_kernel_config_hifi2_fp16
                ),
                dtype=ttnn.bfloat8_b,
                program_config=short_lens_pc_1_3,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            w3_out = ttnn.experimental.minimal_matmul(
                input_tensor=x,
                weight_tensor=self.w3_interleaved if use_w1_w3_interleaved else self.w3,
                config=minimal_pc_1_3,
                compute_kernel_config=self.args.compute_kernel_config_lofi,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        ttnn.deallocate(x)
        w3_out_reduced = self.tt_ccl.line_reduce_scatter(
            w3_out,
            cluster_axis=1,
            num_links=prefill_num_links,
            memory_config=w3_out.memory_config(),
            buffer_key="FF3",
            dim=3,
            batch_size=batch_size,
        )
        ttnn.deallocate(w3_out)
        w2_in = ttnn.mul(
            w1_out_reduced,
            w3_out_reduced,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat8_b,
            memory_config=w1_out.memory_config(),
        )
        # For shorter sequence lengths use the original matmul since it performs better than the minimal matmul.
        # On Blackhole always use the unfused all_gather + matmul: the fused all_gather_minimal_matmul_async
        # kernel only supports the 1D LowLatencyPacketHeader and fails to compile against the 2D-torus
        # fabric's HybridMeshPacketHeader required on BH.
        if seq_len < 4096 or batch_size > 1 or self.args.is_blackhole:
            w2_in_gathered = self.tt_ccl.line_all_gather(
                w2_in,
                cluster_axis=1,
                num_links=prefill_num_links,
                memory_config=w3_out.memory_config(),
                buffer_key="FF3",
                dim=3,
            )
            ttnn.deallocate(w2_in)
            w2_out = ttnn.linear(
                w2_in_gathered,
                self.w2_interleaved,
                compute_kernel_config=self.args.compute_kernel_config_hifi2_fp16,
                dtype=ttnn.bfloat8_b,
                program_config=short_lens_pc_2,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            w2_out = self.tt_ccl.line_all_gather_matmul(
                w2_in,
                self.w2_interleaved,
                dim=3,
                cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                matmul_config=minimal_pc_2,
                compute_kernel_config=self.args.compute_kernel_config_hifi2_fp16,
                dtype=ttnn.bfloat8_b,
            )
            ttnn.deallocate(w2_in)

        w2_out_reduced = self.tt_ccl.line_all_reduce(
            w2_out,
            cluster_axis=0,
            num_links=prefill_num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            buffer_key="FF2",
            batch_size=batch_size,
        )
        ttnn.deallocate(w2_out)

        if 1024 <= seq_len < 4096:
            original_shape = w2_out_reduced.shape
            w2_out_reduced = ttnn.reshape(
                w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
            )

        return w2_out_reduced
