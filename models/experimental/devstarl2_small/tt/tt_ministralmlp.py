# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""
Tenstorrent SwiGLU FFN for Hugging Face Ministral3 (``Ministral3MLP``).

Implementation follows ``models.tt_transformers.tt.mlp.MLP`` (same ``w1``/``w3``/``w2``
mapping and forward schedule) so Devstral text checkpoints continue to use
``layers.{i}.feed_forward.*`` meta keys without importing that class.
"""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.ccl import tt_all_reduce
from models.tt_transformers.tt.common import Mode, pad_to_size
from models.tt_transformers.tt.model_config import OpGroup, TensorGroup


class TtMinistralMLP(LightweightModule):
    # Prefill FF1/FF3: factory ``ttnn.linear`` + reuse-mcast program blows L1 on 5k×32k-class matmuls.
    # Chunk activations (cap below) and run **minimal_matmul** on non-Galaxy (same idea as FF2 for long seq).
    _PREFILL_MLP_M_CAP = 128

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
        super().__init__()

        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.args = args
        self.dim = args.dim
        self.model_config = model_config
        self.layer_num = layer_num

        self.prefetcher = prefetcher

        state_dict_prefix = state_dict_prefix or args.get_state_dict_prefix("MLP", layer_num)
        torch_weight = lambda name: torch.transpose(state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)
        pad_hidden_dim = lambda tensor, dim: pad_to_size(tensor, dim=dim, size=args.hidden_dim)
        hidden_dim_string = f".hidden_dim_{args.hidden_dim}" if args.hidden_dim != args.unpadded_hidden_dim else ""

        if args.dummy_weights:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}{hidden_dim_string}"

        w1_w3_mem_config = args.create_dram_sharded_mem_config(args.dim, args.hidden_dim // args.num_devices)
        w2_mem_config = args.create_dram_sharded_mem_config(args.hidden_dim // args.num_devices, args.dim)

        def as_sharded_tensor(name, type, dims):
            raw_weight = torch_weight(name[:2])
            padded_weight = pad_hidden_dim(raw_weight, dims[0] if args.is_galaxy else dims[-1])
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

        w1_dims = (-1, -2) if args.is_galaxy else (-2, -1)
        w2_dims = (-2, -1) if args.is_galaxy else (-1, -2)

        layer_num = max(layer_num, 0)

        use_prefetcher = prefetcher is not None

        self.decoders_optimizations = self.args.decoders_optimizations

        ff1_3_dtype = self.decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.FF1_FF3, prefetcher=use_prefetcher
        )
        ff2_dtype = self.decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.FF2, prefetcher=use_prefetcher
        )

        self.w1 = as_sharded_tensor("w1_sharded", ff1_3_dtype, dims=w1_dims)
        self.w2 = as_sharded_tensor("w2_sharded", ff2_dtype, dims=w2_dims)
        self.w3 = as_sharded_tensor("w3_sharded", ff1_3_dtype, dims=w1_dims)

        self.activation_type = (
            args.mlp_activation_type if hasattr(args, "mlp_activation_type") else ttnn.UnaryOpType.SILU
        )

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
        full_seq_len = int(x.shape[-2])
        TG = self.args.is_galaxy
        layer_num = max(self.layer_num, 0)
        activation_dtype = self.decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.ACTIVATION
        )

        li_ff1_3_compute_kernel_cfg = self.decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_FF1_FF3, configuration=self.args
        )

        cfg_seq = full_seq_len
        if mode == Mode.PREFILL:
            max_chunk = min(int(self.args.prefill_len_cutoff), int(self._PREFILL_MLP_M_CAP))
            max_chunk = max(max_chunk, 1)
            chunk = max_chunk
            while chunk > 1 and full_seq_len % chunk != 0:
                chunk -= 1
            if full_seq_len > chunk:
                x = ttnn.reshape(x, [1, full_seq_len // chunk, chunk, -1])
                cfg_seq = chunk

        pc_2 = self.args.get_mlp_ff2_prg_config(mode, cfg_seq, self.prefetcher)

        # ``ttnn.linear`` + reuse-mcast program config blows L1 on wide (~5k × ~32k) prefill matmuls.
        # Use the same minimal matmul path as FF2 for long prefills (smaller static CBs); Galaxy keeps linear.
        x_to_deallocate_after_ff13 = None
        # Non-Galaxy PREFILL: use ``mlp1_3_grid`` and DRAM activations only (same as slab fragments).
        # A ~full-chip grid + L1 input activations overflows per-bank L1 on Wormhole for wide FF1/FF3.
        if mode == Mode.PREFILL and not TG:
            grid = self.args.mlp1_3_grid(cfg_seq)
            mmc_ff13 = ttnn.MinimalMatmulConfig(
                M_block_size=8,
                K_block_size=8,
                N_block_size=8,
                compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
            )
            w1_out = ttnn.experimental.minimal_matmul(
                x,
                self.w1,
                compute_kernel_config=li_ff1_3_compute_kernel_cfg,
                config=mmc_ff13,
                dtype=ttnn.bfloat8_b,
            )
            w3_out = ttnn.experimental.minimal_matmul(
                x,
                self.w3,
                compute_kernel_config=li_ff1_3_compute_kernel_cfg,
                config=mmc_ff13,
                dtype=ttnn.bfloat8_b,
            )
            if x_to_deallocate_after_ff13 is not None:
                ttnn.deallocate(x_to_deallocate_after_ff13)
        elif mode == Mode.DECODE and not TG and self.prefetcher is None:
            # DRAM-sharded ttnn.linear (per_core_N=32 tiles for dim=5120→hidden=32768) overflows L1
            # because TTNN overrides the output MemoryConfig with a computed 80-core layout.
            # Use minimal_matmul with DRAM activations instead — same fix as the prefill path above.
            x_dram = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            grid = self.args.mlp1_3_grid(cfg_seq)
            mmc_ff13 = ttnn.MinimalMatmulConfig(
                M_block_size=8,
                K_block_size=8,
                N_block_size=8,
                compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
            )
            w1_out = ttnn.experimental.minimal_matmul(
                x_dram,
                self.w1,
                compute_kernel_config=li_ff1_3_compute_kernel_cfg,
                config=mmc_ff13,
                dtype=ttnn.bfloat8_b,
            )
            w3_out = ttnn.experimental.minimal_matmul(
                x_dram,
                self.w3,
                compute_kernel_config=li_ff1_3_compute_kernel_cfg,
                config=mmc_ff13,
                dtype=ttnn.bfloat8_b,
            )
            ttnn.deallocate(x_dram)
        else:
            pc_1 = self.args.get_mlp_ff1_3_prg_config(mode, cfg_seq, self.prefetcher)
            pc_3 = self.args.get_mlp_ff1_3_prg_config(mode, cfg_seq, self.prefetcher)
            w1_out = ttnn.linear(
                x,
                self.w1,
                dtype=ttnn.bfloat8_b if TG else activation_dtype or ttnn.bfloat16,
                core_grid=None,
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
                core_grid=None,
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
            input_tensor_a_activations=[self.activation_type],
            dtype=activation_dtype or ttnn.bfloat8_b,
            memory_config=w1_out.memory_config(),
        )

        if mode == Mode.DECODE and not TG and self.prefetcher is None:
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

        if cfg_seq > 128 and mode != Mode.DECODE:
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
                core_grid=None,
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


__all__ = ["TtMinistralMLP"]
