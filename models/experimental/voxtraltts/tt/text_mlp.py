# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Voxtral text-model MLP: tt_transformers ``MLP`` + the interleaved-weight / fused-SiLU
decode optimizations, kept OUT of the shared tt_transformers file so it stays untouched.

The class is intentionally named ``MLP`` (the base is imported as ``_BaseMLP``) so
``self.__class__.__name__`` still resolves the state-dict prefix ("MLP" -> "feed_forward").
There is no MLP injection hook in tt_transformers' decoder, so this is swapped onto each
decoder block post-construction (see text_model.py ``_swap_text_mlps``).

``__init__`` / ``forward`` are verbatim copies of the optimized tt_transformers methods, with
the single change that ``__init__``'s ``super().__init__()`` (which would re-run the base MLP
build) is redirected to ``LightweightModule.__init__`` since the base is now the real ``MLP``.
"""
from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.ccl import tt_all_reduce
from models.tt_transformers.tt.common import Mode, pad_to_size
from models.tt_transformers.tt.mlp import MLP as _BaseMLP
from models.tt_transformers.tt.model_config import OpGroup, TensorGroup


class MLP(_BaseMLP):
    def __init__(
        self,
        mesh_device,
        tt_ccl,
        args,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        model_config,
        state_dict_prefix=None,
        prefetcher=None,
    ):
        LightweightModule.__init__(self)

        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.args = args
        self.dim = args.dim
        self.model_config = model_config
        self.layer_num = layer_num

        # Define the prefetcher object
        self.prefetcher = prefetcher

        state_dict_prefix = state_dict_prefix or args.get_state_dict_prefix(self.__class__.__name__, layer_num)
        torch_weight = lambda name: torch.transpose(state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)
        pad_hidden_dim = lambda tensor, dim: pad_to_size(tensor, dim=dim, size=args.hidden_dim)
        # If padding was applied (e.g. via env var), add the unpadded hidden dim to the cache name to avoid loading incorrect weights
        hidden_dim_string = f".hidden_dim_{args.hidden_dim}" if args.hidden_dim != args.unpadded_hidden_dim else ""

        if args.dummy_weights:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}{hidden_dim_string}"

        # Opt-in (Voxtral mlp_interleaved_weights): w1/w3 DRAM-INTERLEAVED instead of bank-sharded,
        # so decode can use a 1D mcast matmul. getattr-gated → every other model unchanged.
        if getattr(args, "mlp_interleaved_weights", False):
            w1_w3_mem_config = ttnn.DRAM_MEMORY_CONFIG
        else:
            w1_w3_mem_config = args.create_dram_sharded_mem_config(args.dim, args.hidden_dim // args.num_devices)
        # Opt-in (Voxtral mlp_ff2_interleaved_weights): w2 DRAM-INTERLEAVED instead of bank-sharded,
        # so decode can use a 1D mcast w2 matmul. getattr-gated → every other model unchanged.
        if getattr(args, "mlp_ff2_interleaved_weights", False):
            w2_mem_config = ttnn.DRAM_MEMORY_CONFIG
        else:
            w2_mem_config = args.create_dram_sharded_mem_config(args.hidden_dim // args.num_devices, args.dim)

        # TODO Clean up this code. With sharding, we load the normal weights and then shard them
        # Note: unsqueeze(0).unsqueeze(0) makes weights 4D [1, 1, H, W] to match attention weights
        # This is required for the dram_prefetcher to correctly interpret all weights
        def as_sharded_tensor(name, type, dims):
            # First get the raw weight and transpose it
            raw_weight = torch_weight(name[:2])  # This is 2D: [H, W]
            # Pad if needed
            padded_weight = pad_hidden_dim(raw_weight, dims[0] if args.is_galaxy else dims[-1])
            # Make 4D: [1, 1, H, W] - CRITICAL for prefetcher to work correctly
            torch_tensor = padded_weight.unsqueeze(0).unsqueeze(0)

            result = ttnn.as_tensor(
                torch_tensor,
                dtype=type,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=dims, mesh_shape=args.cluster_shape),
                layout=ttnn.TILE_LAYOUT,
                memory_config=(
                    ttnn.DRAM_MEMORY_CONFIG if args.is_galaxy else w2_mem_config if "w2" in name else w1_w3_mem_config
                ),
                cache_file_name=cache_name(name),
            )
            return result

        # Sharded weights
        w1_dims = (-1, -2) if args.is_galaxy else (-2, -1)
        w2_dims = (-2, -1) if args.is_galaxy else (-1, -2)

        layer_num = max(layer_num, 0)  # cross_block uses the configuration of the first decoder

        # When prefetcher is enabled, use consistent dtypes across all layers to avoid
        # race conditions caused by different block sizes
        use_prefetcher = prefetcher is not None

        self.decoders_optimizations = self.args.decoders_optimizations

        ff1_3_dtype = self.decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.FF1_FF3, prefetcher=use_prefetcher
        )
        ff2_dtype = self.decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.FF2, prefetcher=use_prefetcher
        )

        # Layout-aware cache tag: interleaved w1/w3 must NOT reuse the DRAM-sharded cache entry
        # (as_tensor loads the cached tensor's layout and ignores memory_config on a cache hit).
        _ff13_tag = "w{}_interleaved" if getattr(args, "mlp_interleaved_weights", False) else "w{}_sharded"
        self.w1 = as_sharded_tensor(
            _ff13_tag.format(1), ff1_3_dtype, dims=w1_dims
        )  # bfp4 normally ok here but sub .99 pcc for llama 3.1 weights
        # Layout-aware tag: interleaved w2 must NOT reuse the DRAM-sharded cache entry (as_tensor
        # loads the cached tensor's layout and ignores memory_config on a cache hit).
        _ff2_tag = "w2_interleaved" if getattr(args, "mlp_ff2_interleaved_weights", False) else "w2_sharded"
        self.w2 = as_sharded_tensor(_ff2_tag, ff2_dtype, dims=w2_dims)
        self.w3 = as_sharded_tensor(_ff13_tag.format(3), ff1_3_dtype, dims=w1_dims)

        # Default activation is SILU
        self.activation_type = (
            args.mlp_activation_type if hasattr(args, "mlp_activation_type") else ttnn.UnaryOpType.SILU
        )

        # Insert the tensors into the prefetcher if it is used
        if self.prefetcher is not None:

            def register_weights():
                self.prefetcher.insert_tensor(self.w1)
                self.prefetcher.insert_tensor(self.w3)
                self.prefetcher.insert_tensor(self.w2)

            self.prefetcher.register_callback(register_weights)

    def forward(self, x: ttnn.Tensor, mode: Mode) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        seq_len = x.shape[-2]
        TG = self.args.is_galaxy
        layer_num = max(self.layer_num, 0)  # cross_block uses the configuration of the first decoder
        activation_dtype = self.decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.ACTIVATION
        )
        li_ff1_3_compute_kernel_cfg = self.decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_FF1_FF3, configuration=self.args
        )

        if mode == Mode.PREFILL and seq_len >= self.args.prefill_len_cutoff:  # 512 if Blackhole, 1024 if Wormhole
            # Reshape input to to fit on device and parallelize computation
            x = ttnn.reshape(x, [1, seq_len // self.args.prefill_len_cutoff, self.args.prefill_len_cutoff, -1])

        # In decode mode (seqlen <= 32) do DRAM sharded matmuls
        # These use HiFi2; this drops 1 bit of the activations but would be FLOP-bound on 12 cores with HiFi4
        w1_fuse_silu_decode = (
            mode == Mode.DECODE
            and getattr(self.args, "mlp_w1_fuse_silu_decode", False)
            and hasattr(self.args, "get_mlp_ff1_w1_prg_config")
        )
        pc_1 = (
            self.args.get_mlp_ff1_w1_prg_config(mode, seq_len, self.prefetcher)
            if w1_fuse_silu_decode
            else self.args.get_mlp_ff1_3_prg_config(mode, seq_len, self.prefetcher)
        )
        pc_2 = self.args.get_mlp_ff2_prg_config(mode, seq_len, self.prefetcher)
        pc_3 = self.args.get_mlp_ff1_3_prg_config(mode, seq_len, self.prefetcher)

        w1_out = ttnn.linear(
            x,
            self.w1,
            dtype=ttnn.bfloat8_b if TG else activation_dtype or ttnn.bfloat16,
            core_grid=None,  # FIXME: validate on TG ttnn.CoreGrid(y=8, x=8) if not pc_1 else None,
            compute_kernel_config=li_ff1_3_compute_kernel_cfg,
            program_config=pc_1,
            memory_config=self.args.get_mlp_ff1_3_mem_config(mode, self.prefetcher),
            global_cb=self.prefetcher.global_cb if self.prefetcher is not None and mode == Mode.DECODE else None,
            sub_device_id=self.prefetcher.worker_sub_device_id
            if self.prefetcher is not None and mode == Mode.DECODE
            else None,
        )
        w3_out = ttnn.linear(
            x,
            self.w3,
            dtype=ttnn.bfloat8_b if TG else activation_dtype or ttnn.bfloat16,
            core_grid=None,  # FIXME: validate on TG ttnn.CoreGrid(y=8, x=8) if not pc_3 else None,
            compute_kernel_config=li_ff1_3_compute_kernel_cfg,
            program_config=pc_3,
            memory_config=self.args.get_mlp_ff1_3_mem_config(mode, self.prefetcher),
            global_cb=self.prefetcher.global_cb if self.prefetcher is not None and mode == Mode.DECODE else None,
            sub_device_id=self.prefetcher.worker_sub_device_id
            if self.prefetcher is not None and mode == Mode.DECODE
            else None,
        )
        ttnn.deallocate(x)

        if TG:
            # if mode == "decode" and self.dim!=8192:
            #     w1_out = ttnn.to_memory_config(w1_out, ttnn.DRAM_MEMORY_CONFIG)
            #     w3_out = ttnn.to_memory_config(w3_out, ttnn.DRAM_MEMORY_CONFIG)
            if self.dim == 8192 or mode == Mode.PREFILL:
                input_mem_cfg = w1_out.memory_config()

                cluster_axis = 1
                w1_out = ttnn.experimental.reduce_scatter_minimal_async(
                    w1_out,
                    persistent_output_buffers=None,
                    dim=3,
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
                    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                    num_links=self.tt_ccl.get_num_links(cluster_axis),
                    cluster_axis=cluster_axis,
                    memory_config=self.model_config["FF1_OUT_REDUCE_SCATTER_MEMCFG"] if mode == Mode.DECODE else None,
                    intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=ttnn.Topology.Linear,
                    chunks_per_sync=10,
                    num_workers_per_link=2,
                    num_buffers_per_channel=2,
                )

                w3_out = ttnn.experimental.reduce_scatter_minimal_async(
                    w3_out,
                    persistent_output_buffers=None,
                    dim=3,
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
                    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                    num_links=1,
                    cluster_axis=cluster_axis,
                    memory_config=self.model_config["FF1_OUT_REDUCE_SCATTER_MEMCFG"] if mode == Mode.DECODE else None,
                    intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=ttnn.Topology.Linear,
                    chunks_per_sync=10,
                    num_workers_per_link=2,
                    num_buffers_per_channel=2,
                )
            else:
                # NOTE: In MLP All-reduce hard codes to 2 links, so we do not get the dynamic link count from the CCL class
                # to avoid any performance regressions.
                w1_out = tt_all_reduce(
                    w1_out,
                    self.mesh_device,
                    self.tt_ccl,
                    cluster_axis=1,
                    num_all_gather_links=2,
                    sharded=True if mode == Mode.DECODE else False,
                    topology=self.args.ccl_topology(),
                    memory_config=self.model_config["FF1_OUT_GATHERED_MEMCFG"] if mode == Mode.DECODE else None,
                )
                w3_out = tt_all_reduce(
                    w3_out,
                    self.mesh_device,
                    self.tt_ccl,
                    cluster_axis=1,
                    num_all_gather_links=2,
                    sharded=True if mode == Mode.DECODE else False,
                    topology=self.args.ccl_topology(),
                    memory_config=self.model_config["FF1_OUT_GATHERED_MEMCFG"] if mode == Mode.DECODE else None,
                )

        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[] if w1_fuse_silu_decode else [self.activation_type],
            dtype=activation_dtype or ttnn.bfloat8_b,
            memory_config=w1_out.memory_config(),
        )

        if mode == Mode.DECODE and not TG and self.prefetcher is None:
            # w2 may use a different core grid, this is a no-op if they already match
            w2_in = ttnn.to_memory_config(w2_in, self.args.get_mlp_binary_mult_mem_config(mode))

        ttnn.deallocate(w3_out)
        ttnn.deallocate(w1_out)

        if TG and (self.dim == 8192 or mode == Mode.PREFILL):
            cluster_axis = 1
            w2_in = ttnn.experimental.all_gather_async(
                w2_in,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
                num_links=2,
                cluster_axis=1,
                topology=ttnn.Topology.Linear,
                memory_config=input_mem_cfg,
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )

            if mode == Mode.DECODE:
                w2_in = ttnn.to_memory_config(w2_in, ttnn.L1_MEMORY_CONFIG)

        li_ff2_compute_kernel_cfg = self.decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_FF2, configuration=self.args
        )

        if seq_len > 128 and mode != Mode.DECODE:
            w2_out = ttnn.experimental.minimal_matmul(
                w2_in,
                self.w2,
                compute_kernel_config=li_ff2_compute_kernel_cfg,
                config=pc_2,
            )
        else:
            w2_out = ttnn.linear(
                w2_in,
                self.w2,
                compute_kernel_config=li_ff2_compute_kernel_cfg,
                dtype=self.args.ccl_dtype if TG else activation_dtype or ttnn.bfloat16,
                program_config=pc_2,
                memory_config=self.args.get_mlp_ff2_mem_config(mode, self.prefetcher),
                core_grid=None,  # FIXME: validate on TG ttnn.CoreGrid(y=8, x=8) if not pc_2 else None,
                global_cb=self.prefetcher.global_cb if self.prefetcher is not None and mode == Mode.DECODE else None,
                sub_device_id=self.prefetcher.worker_sub_device_id
                if self.prefetcher is not None and mode == Mode.DECODE
                else None,
            )
        ttnn.deallocate(w2_in)

        w2_out_reduced = tt_all_reduce(
            w2_out,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=0,
            dim=0 if (TG and self.dim < 8192) else 3,
            sharded=(mode == Mode.DECODE),
            memory_config=self.args.get_mlp_ff2_all_reduce_mem_config(mode, w2_out),
            rs_memory_config=self.model_config["MLP_RS_CONFIG"]["rs_memory_config"]
            if mode == Mode.DECODE
            else ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.args.ccl_dtype,
            use_composite=True if self.dim == 8192 else False,
            topology=self.args.ccl_topology(),
            chunks_per_sync=self.model_config["MLP_RS_CONFIG"]["chunks_per_sync"] if mode == Mode.DECODE else 10,
            num_workers_per_link=self.model_config["MLP_RS_CONFIG"]["num_workers_per_link"]
            if mode == Mode.DECODE
            else 2,
            subdevice_id=self.prefetcher.worker_sub_device_id
            if mode == Mode.DECODE and self.prefetcher is not None
            else None,
        )
        # Ensure dim 0 and 1 are 1
        original_shape = w2_out_reduced.shape
        w2_out_reduced = ttnn.reshape(
            w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )

        if mode == Mode.DECODE:
            w2_out_reduced = ttnn.to_memory_config(
                w2_out_reduced,
                self.args.get_mlp_output_mem_config(mode, self.prefetcher),
            )

        return w2_out_reduced
