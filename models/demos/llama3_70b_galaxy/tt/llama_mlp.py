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

        if tt_ccl.mode == "decode":
            self.prefetch(prefetcher_setup, tt_ccl)

    def prefetch(self, prefetcher_setup, tt_ccl):
        self.prefetcher_setup = prefetcher_setup
        if tt_ccl.mode == "decode" and prefetcher_setup is not None:
            self.prefetcher_setup.insert_tensor(self.w1)
            self.prefetcher_setup.insert_tensor(self.w3)
            self.prefetcher_setup.insert_tensor(self.w2)
        self.tt_ccl = tt_ccl

    def _debug_check_mlp(self, name, tensor):
        """Check tensor for Inf/NaN and log stats."""
        import os

        if os.environ.get("DEBUG_DECODE", "0") != "1":
            return
        try:
            import torch
            from loguru import logger

            torch_tensor = ttnn.to_torch(
                tensor,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    self.mesh_device, dims=(1, 3), mesh_shape=self.args.cluster_shape
                ),
            )
            has_inf = torch.isinf(torch_tensor).any().item()
            has_nan = torch.isnan(torch_tensor).any().item()
            max_val = torch_tensor.float().abs().max().item()
            status = "OK" if not (has_inf or has_nan) else "BAD"
            logger.info(
                f"    MLP [{status}] {name}: shape={list(torch_tensor.shape)}, max={max_val:.4e}, Inf={has_inf}, NaN={has_nan}"
            )
        except Exception as e:
            from loguru import logger

            logger.error(f"    MLP [ERROR] {name}: {e}")

    def forward(self, x: ttnn.Tensor, mode, batch_size=1, skip_input_dealloc=False) -> ttnn.Tensor:
        if mode == "prefill":
            return self.forward_prefill(x, batch_size=batch_size, skip_input_dealloc=skip_input_dealloc)

        # Choose program config based on prefetcher status
        # NO_PREFETCH configs have num_global_cb_receivers=1 (don't require global_cb)
        if self.model_config["USE_PREFETCHER"]:
            pc_1_3 = self.model_config["FF1_3_TG_RING_PROGCFG"]
            pc_2 = self.model_config["FF2_TG_RING_PROGCFG"]
        else:
            pc_1_3 = self.model_config.get(
                "FF1_3_TG_RING_PROGCFG_NO_PREFETCH", self.model_config["FF1_3_TG_RING_PROGCFG"]
            )
            pc_2 = self.model_config.get("FF2_TG_RING_PROGCFG_NO_PREFETCH", self.model_config["FF2_TG_RING_PROGCFG"])

        is_olmo = getattr(self.args, "is_olmo", False)

        if not self.model_config["USE_PREFETCHER"]:
            if is_olmo:
                # OLMo decode: ring matmul outputs 3840-padded, slice to 3456 before reduce_scatter.
                # REDUCE_SCATTER_OUT_MEMCFG (L1) is incompatible with OLMo due to L1 constraints,
                # so reduce_scatter outputs to DRAM. We avoid the redundant L1→DRAM push by slicing
                # directly from DRAM after moving the padded ring output to DRAM.
                unpadded_width = self.args.intermediate_dim_per_tp  # 3456

                w1_out = ttnn.linear(
                    x,
                    self.w1,
                    compute_kernel_config=self.args.compute_kernel_config_lofi
                    if self.four_bit_mlp
                    else self.args.compute_kernel_config_hifi2,
                    dtype=ttnn.bfloat8_b,
                    program_config=pc_1_3,
                    memory_config=self.model_config["SHARDED_FF12_OUT_RING_MEMCFG"],
                )
                w1_out_dram = ttnn.to_memory_config(w1_out, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(w1_out)
                w1_out_unpadded = ttnn.slice(w1_out_dram, [0, 0, 0, 0], [1, 1, 32, unpadded_width])
                w1_out_reduced = self.tt_ccl.line_reduce_scatter(
                    w1_out_unpadded,
                    cluster_axis=1,
                    num_links=self.model_config["GALAXY_NUM_LINKS"],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    use_noc1_only=False,
                )
                ttnn.deallocate(w1_out_dram)
                ttnn.deallocate(w1_out_unpadded)

                w3_out = ttnn.linear(
                    x,
                    self.w3,
                    compute_kernel_config=self.args.compute_kernel_config_lofi
                    if self.four_bit_mlp
                    else self.args.compute_kernel_config_hifi2,
                    dtype=ttnn.bfloat8_b,
                    program_config=pc_1_3,
                    memory_config=self.model_config["SHARDED_FF12_OUT_RING_MEMCFG"],
                )
                ttnn.deallocate(x)
                w3_out_dram = ttnn.to_memory_config(w3_out, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(w3_out)
                w3_out_unpadded = ttnn.slice(w3_out_dram, [0, 0, 0, 0], [1, 1, 32, unpadded_width])
                w3_out_reduced = self.tt_ccl.line_reduce_scatter(
                    w3_out_unpadded,
                    cluster_axis=1,
                    num_links=self.model_config["GALAXY_NUM_LINKS"],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    use_noc1_only=False,
                )
                ttnn.deallocate(w3_out_dram)
                ttnn.deallocate(w3_out_unpadded)
            else:
                w1_out = ttnn.linear(
                    x,
                    self.w1,
                    compute_kernel_config=self.args.compute_kernel_config_lofi
                    if self.four_bit_mlp
                    else self.args.compute_kernel_config_hifi2,
                    dtype=ttnn.bfloat8_b,
                    program_config=pc_1_3,
                    memory_config=self.model_config["SHARDED_FF12_OUT_RING_MEMCFG"],
                )
                self._debug_check_mlp("w1_out", w1_out)
                w1_out_reduced = self.tt_ccl.line_reduce_scatter(
                    w1_out,
                    cluster_axis=1,
                    num_links=self.model_config["GALAXY_NUM_LINKS"],
                    memory_config=self.model_config["REDUCE_SCATTER_OUT_MEMCFG"],
                    use_noc1_only=False,
                )
                ttnn.deallocate(w1_out)
                self._debug_check_mlp("w1_out_reduced", w1_out_reduced)

                w3_out = ttnn.linear(
                    x,
                    self.w3,
                    compute_kernel_config=self.args.compute_kernel_config_lofi
                    if self.four_bit_mlp
                    else self.args.compute_kernel_config_hifi2,
                    dtype=ttnn.bfloat8_b,
                    program_config=pc_1_3,
                    memory_config=self.model_config["SHARDED_FF12_OUT_RING_MEMCFG"],
                )
                ttnn.deallocate(x)
                self._debug_check_mlp("w3_out", w3_out)

                w3_out_reduced = self.tt_ccl.line_reduce_scatter(
                    w3_out,
                    cluster_axis=1,
                    num_links=self.model_config["GALAXY_NUM_LINKS"],
                    memory_config=self.model_config["REDUCE_SCATTER_OUT_MEMCFG"],
                    use_noc1_only=False,
                )
                ttnn.deallocate(w3_out)
                self._debug_check_mlp("w3_out_reduced", w3_out_reduced)
        else:
            # Standard path: Use fused double_matmul_line_reduce_scatter with prefetcher
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
                global_cb=self.prefetcher_setup.global_circular_buffer if self.model_config["USE_PREFETCHER"] else None,
                sub_device_id=self.prefetcher_setup.worker_sub_device_id
                if (mode == "decode" and self.prefetcher_setup is not None)
                else None,
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

        ff1ff3_mem_config = ttnn.DRAM_MEMORY_CONFIG if is_olmo else self.model_config["REDUCE_SCATTER_OUT_MEMCFG"]
        # OLMo: use bfloat16 for SwiGLU output (ff1ff3) in both prefill and decode to preserve
        # precision feeding W2 matmul.  bfloat8_b quantizes ff1ff3 per layer causing ~3-8%
        # relative error that compounds to >50% over 64 layers, collapsing hidden-state PCC.
        ff1ff3_dtype = ttnn.bfloat16 if is_olmo else ttnn.bfloat8_b
        ff1ff3 = ttnn.mul(
            w1_out_reduced,
            w3_out_reduced,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ff1ff3_dtype,
            memory_config=ff1ff3_mem_config,
        )
        self._debug_check_mlp("ff1ff3 (silu*gate)", ff1ff3)

        ttnn.deallocate(w3_out_reduced)
        ttnn.deallocate(w1_out_reduced)

        if is_olmo and mode == "decode" and self.model_config.get("USE_AGMM_FF2", False):
            # Fused AllGather + Matmul: replaces line_all_gather + ttnn.linear + to_memory_config
            # ff1ff3 is [32, 864] DRAM; AGMM gathers across 4 TP col-devices then runs matmul
            # w2_interleaved is DRAM; converted to L1-sharded inside olmo_ff2_all_gather_matmul
            # Output: [32, 1280] L1 in FF2_OUT_RING_MEMCFG_OLMO (10 cores) — ready for all_reduce
            w2_out_sharded = self.tt_ccl.olmo_ff2_all_gather_matmul(
                ff1ff3,
                self.w2_interleaved,  # DRAM-interleaved; moved to L1 inside AGMM call
                compute_kernel_config=self.args.compute_kernel_config_hifi2,
                sub_device_id=self.tt_ccl.worker_sub_device_id,
            )
            ttnn.deallocate(ff1ff3)
            w2_out_reduced = self.tt_ccl.line_all_reduce(
                w2_out_sharded,
                cluster_axis=0,
                num_links=self.model_config["GALAXY_NUM_LINKS"],
                memory_config=self.model_config["DECODE_RESIDUAL_MEMCFG"],
                use_optimal_ccl_for_llama=True,
            )
            ttnn.deallocate(w2_out_sharded)
            self._debug_check_mlp("w2_out_reduced", w2_out_reduced)
            return w2_out_reduced

        if is_olmo and mode == "decode":
            # Original path: separate all_gather + linear (fallback when USE_AGMM_FF2=False)
            w2_in = self.tt_ccl.line_all_gather(
                ff1ff3,
                dim=3,
                cluster_axis=1,
                num_links=self.model_config["GALAXY_NUM_LINKS"],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                buffer_key="BINARY_MUL_BF16",
                use_optimal_ccl_for_llama=False,
            )
        elif is_olmo and mode == "prefill":
            # Prefill is not traced so allocating a fresh bfloat16 tensor is safe.
            # buffer_key=None avoids the bfloat8_b-typed BINARY_MUL persistent buffer
            # which would re-quantize the bfloat16 ff1ff3 and defeat the precision gain.
            w2_in = self.tt_ccl.line_all_gather(
                ff1ff3,
                dim=3,
                cluster_axis=1,
                num_links=self.model_config["GALAXY_NUM_LINKS"],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                buffer_key=None,
                use_optimal_ccl_for_llama=False,
            )
        else:
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

        if is_olmo and mode == "decode":
            w2_out = ttnn.linear(
                w2_in,
                self.w2_interleaved,
                compute_kernel_config=self.args.compute_kernel_config_hifi2,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            # w2_in IS the BINARY_MUL_BF16 persistent buffer — do NOT deallocate it.
            self._debug_check_mlp("w2_out", w2_out)
            w2_out_sharded = ttnn.to_memory_config(w2_out, self.model_config["FF2_OUT_RING_MEMCFG_OLMO"])
            ttnn.deallocate(w2_out)
            w2_out_reduced = self.tt_ccl.line_all_reduce(
                w2_out_sharded,
                cluster_axis=0,
                num_links=self.model_config["GALAXY_NUM_LINKS"],
                memory_config=self.model_config["DECODE_RESIDUAL_MEMCFG"],
                use_optimal_ccl_for_llama=True,
            )
            ttnn.deallocate(w2_out_sharded)
        else:
            w2_out = ttnn.linear(
                w2_in,
                self.w2,
                compute_kernel_config=self.args.compute_kernel_config_hifi2,
                dtype=ttnn.bfloat8_b,
                program_config=pc_2,
                memory_config=self.model_config["FF2_OUT_RING_MEMCFG"],
                core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_2 else None,
                global_cb=self.prefetcher_setup.global_circular_buffer if self.model_config["USE_PREFETCHER"] else None,
                sub_device_id=self.prefetcher_setup.worker_sub_device_id
                if (mode == "decode" and self.prefetcher_setup is not None)
                else None,
            )
            self._debug_check_mlp("w2_out", w2_out)
            w2_out_reduced = self.tt_ccl.line_all_reduce(
                w2_out,
                cluster_axis=0,
                num_links=self.model_config["GALAXY_NUM_LINKS"],
                memory_config=self.model_config["DECODE_RESIDUAL_MEMCFG"],
                use_optimal_ccl_for_llama=True,
            )
            ttnn.deallocate(w2_out)
        self._debug_check_mlp("w2_out_reduced", w2_out_reduced)

        return w2_out_reduced

    def forward_prefill(self, x: ttnn.Tensor, batch_size=1, skip_input_dealloc=False) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        seq_len = x.shape[-2]
        is_olmo = getattr(self.args, "is_olmo", False)
        if is_olmo:
            # OLMo: always use w1_interleaved (DRAM) in prefill.
            # The sharded w1 (W1W3_RING_MEMCFG) combined with MatmulMultiCoreReuseMultiCast
            # is incompatible with OLMo's intermediate_dim (3456 ≠ Llama's 3584) for
            # seqlens 1024-3072, causing ~0 PCC (garbage) output.
            use_w1_w3_interleaved = True
        else:
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
            # For seq_len == 4096 the ring_reduce_scatter uses a persistent bfloat8_b buffer
            # (support_seqlens = [128, 1024, 2048, 4096]); the input dtype must match.
            # For seq_len >= 8192 there is no persistent buffer → dynamic alloc works with
            # the default (bfloat16 from input); forcing bfloat8_b hangs the ring path.
            w1_minimal_dtype = ttnn.bfloat8_b if seq_len <= 4096 else None
            w1_out = ttnn.experimental.minimal_matmul(
                input_tensor=x,
                weight_tensor=self.w1_interleaved if use_w1_w3_interleaved else self.w1,
                config=minimal_pc_1_3,
                dtype=w1_minimal_dtype,
                compute_kernel_config=self.args.compute_kernel_config_lofi,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        w1_out_reduced = self.tt_ccl.line_reduce_scatter(
            w1_out,
            cluster_axis=1,
            num_links=self.model_config["GALAXY_NUM_LINKS"],
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
            w3_minimal_dtype = ttnn.bfloat8_b if seq_len <= 4096 else None
            w3_out = ttnn.experimental.minimal_matmul(
                input_tensor=x,
                weight_tensor=self.w3_interleaved if use_w1_w3_interleaved else self.w3,
                config=minimal_pc_1_3,
                dtype=w3_minimal_dtype,
                compute_kernel_config=self.args.compute_kernel_config_lofi,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        if not skip_input_dealloc:
            ttnn.deallocate(x)
        w3_out_reduced = self.tt_ccl.line_reduce_scatter(
            w3_out,
            cluster_axis=1,
            num_links=self.model_config["GALAXY_NUM_LINKS"],
            memory_config=w3_out.memory_config(),
            buffer_key="FF3",
            dim=3,
            batch_size=batch_size,
        )
        ttnn.deallocate(w3_out)
        # OLMo: bfloat8_b for ff1ff3 introduces ~3-8% relative error per layer that
        # compounds to >50% over 64 layers.  Use bfloat16 to preserve precision.
        ff1ff3_dtype = ttnn.bfloat16 if is_olmo else ttnn.bfloat8_b
        w2_in = ttnn.mul(
            w1_out_reduced,
            w3_out_reduced,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ff1ff3_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if is_olmo:
            # Use the pre-allocated bfloat16 persistent buffer (FF3_BF16) so the
            # all_gather preserves the bfloat16 ff1ff3 values.  Using "FF3" (bfloat8_b)
            # would re-quantize ff1ff3 and defeat the precision gain.
            w2_in_gathered = self.tt_ccl.line_all_gather(
                w2_in,
                cluster_axis=1,
                num_links=self.model_config["GALAXY_NUM_LINKS"],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                buffer_key="FF3_BF16",
                dim=3,
            )
        else:
            w2_in_gathered = self.tt_ccl.line_all_gather(
                w2_in,
                cluster_axis=1,
                num_links=self.model_config["GALAXY_NUM_LINKS"],
                memory_config=w3_out.memory_config(),
                buffer_key="FF3",
                dim=3,
            )
        ttnn.deallocate(w2_in)

        if seq_len < 4096 or batch_size > 1:
            w2_out = ttnn.linear(
                w2_in_gathered,
                self.w2_interleaved,
                compute_kernel_config=self.args.compute_kernel_config_hifi2_fp16,
                dtype=ttnn.bfloat8_b,
                program_config=short_lens_pc_2,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            # FF2 persistent buffer also only exists for seq_len <= 4096.
            w2_minimal_dtype = ttnn.bfloat8_b if seq_len <= 4096 else None
            w2_out = ttnn.experimental.minimal_matmul(
                input_tensor=w2_in_gathered,
                weight_tensor=self.w2_interleaved,
                config=minimal_pc_2,
                dtype=w2_minimal_dtype,
                compute_kernel_config=self.args.compute_kernel_config_hifi2_fp16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        w2_out_reduced = self.tt_ccl.line_all_reduce(
            w2_out,
            cluster_axis=0,
            num_links=self.model_config["GALAXY_NUM_LINKS"],
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
