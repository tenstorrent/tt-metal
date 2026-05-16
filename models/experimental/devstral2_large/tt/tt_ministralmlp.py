# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""
Tenstorrent SwiGLU FFN for Devstral-2 123B (``Ministral3MLP`` in HF).

Copied from ``models.experimental.devstarl2_small.tt.tt_ministralmlp`` with Devstral-2–large
mitigations: on **multi-device** meshes (non-Galaxy) with wide hidden dim (~12k), **Blackhole**
and **Wormhole T3K** use **interleaved DRAM** for FFN weights and wide FFN activations where the
shared path would otherwise hit static circular-buffer limits during tilize / wide matmuls.
Dense 123B Devstral differs from MoE GPT-OSS (~120B): every token uses full FFN width, so the
same L1 pressure appears on WH multi-chip as on BH multi-chip.
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.devstral2_large.tt.model_utils import (
    devstral_to_memory_if_needed,
    devstral2_large_multi_device_dram_mitigation,
)
from models.tt_transformers.tt.ccl import tt_all_reduce
from models.tt_transformers.tt.common import Mode, pad_to_size
from models.tt_transformers.tt.model_config import OpGroup, TensorGroup


class TtDevstral2LargeMLP(LightweightModule):
    """SwiGLU FFN (``w1``/``w3`` gate×up, ``w2`` down); wide + multi-device uses DRAM-heavy configs."""

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
        use_dram_weights = args.is_galaxy or devstral2_large_multi_device_dram_mitigation(mesh_device, args)

        def as_sharded_tensor(name, weight_dtype, dims):
            raw_weight = torch_weight(name[:2])
            padded_weight = pad_hidden_dim(raw_weight, dims[0] if args.is_galaxy else dims[-1])
            torch_tensor = padded_weight.unsqueeze(0).unsqueeze(0)

            memory_config = (
                ttnn.DRAM_MEMORY_CONFIG if use_dram_weights else (w2_mem_config if "w2" in name else w1_w3_mem_config)
            )

            result = ttnn.as_tensor(
                torch_tensor,
                dtype=weight_dtype,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=dims, mesh_shape=args.cluster_shape),
                layout=ttnn.TILE_LAYOUT,
                memory_config=memory_config,
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

    def _dram_intermediates(self) -> bool:
        return devstral2_large_multi_device_dram_mitigation(self.mesh_device, self.args)

    def _ff1_3_out_mem(self, mode: Mode):
        # Decode uses ``dram_matmul_config`` (``get_mlp_ff1_3_prg_config``): output mem must be
        # **sharded**; interleaved ``DRAM_MEMORY_CONFIG`` fails validation (``output_mem_config.is_sharded()``).
        # DRAM mitigation still applies to **prefill** and weight placement; decode FFN matmul I/O
        # must match the DRAM-sharded matmul contract.
        if self._dram_intermediates() and mode != Mode.DECODE:
            return ttnn.DRAM_MEMORY_CONFIG
        return self.args.get_mlp_ff1_3_mem_config(mode, self.prefetcher)

    def _ff2_out_mem(self, mode: Mode):
        if self._dram_intermediates() and mode != Mode.DECODE:
            return ttnn.DRAM_MEMORY_CONFIG
        return self.args.get_mlp_ff2_mem_config(mode, self.prefetcher)

    def _ff2_ar_out_mem(self, mode: Mode, w2_out: ttnn.Tensor):
        if self._dram_intermediates() and mode != Mode.DECODE:
            return ttnn.DRAM_MEMORY_CONFIG
        return self.args.get_mlp_ff2_all_reduce_mem_config(mode, w2_out)

    def _binary_mult_mem(self, mode: Mode):
        return self.args.get_mlp_binary_mult_mem_config(mode)

    def forward(self, x: ttnn.Tensor, mode: Mode) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        seq_len = int(x.shape[-2])
        TG = self.args.is_galaxy
        layer_num = max(self.layer_num, 0)
        activation_dtype = self.decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.ACTIVATION
        )
        li_ff1_3_compute_kernel_cfg = self.decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_FF1_FF3, configuration=self.args
        )

        # ``get_mlp_ff1_3_prg_config(DECODE)`` uses ``dram_matmul_config``, which requires **sharded**
        # activations. Post-attention RMSNorm on BH wide models returns **interleaved** L1 (see
        # ``tt_ministralrmsnorm._forward_bh_wide_local_rmsnorm``), so re-shard before w1/w3.
        if mode == Mode.DECODE and not TG and self.prefetcher is None and not x.is_sharded():
            mlp_in = self.args.get_mlp_input_mem_config(mode, self.prefetcher)
            x = devstral_to_memory_if_needed(x, mlp_in)

        # Interleaved DRAM ``w1``/``w3`` are incompatible with ``dram_matmul_config`` (requires
        # width-sharded in1). Use ``ttnn.linear`` **without** that program config for gate/up only.
        # Decode ``w2`` uses the same generic-linear path (``pc_2`` requires width-sharded DRAM ``w2``).
        use_dram_decode_linear_gate_up = (
            mode == Mode.DECODE and not TG and self.prefetcher is None and self._dram_intermediates()
        )

        if mode == Mode.PREFILL and seq_len >= self.args.prefill_len_cutoff:
            x = ttnn.reshape(x, [1, seq_len // self.args.prefill_len_cutoff, self.args.prefill_len_cutoff, -1])

        pc_1 = self.args.get_mlp_ff1_3_prg_config(mode, seq_len, self.prefetcher)
        pc_2 = self.args.get_mlp_ff2_prg_config(mode, seq_len, self.prefetcher)
        pc_3 = self.args.get_mlp_ff1_3_prg_config(mode, seq_len, self.prefetcher)

        use_minimal_matmul_w2 = mode != Mode.DECODE and (seq_len > 128 or self._dram_intermediates())
        if (
            mode == Mode.DECODE
            and not use_minimal_matmul_w2
            and not use_dram_decode_linear_gate_up
            and self._dram_intermediates()
        ):
            nc = int(os.environ.get("DEVSTRAL2_DECODE_FFN_DRAM_CORES", "16"))
            nc = max(1, min(nc, int(self.args.mlp2_core_grid.num_cores)))
            try:
                pc_2 = self.args.dram_matmul_config(
                    self.args.tile_padded_batch_rows,
                    self.args.hidden_dim // self.args.cluster_shape[1],
                    self.args.dim,
                    num_cores=nc,
                )
            except AssertionError:
                pass

        ff1_3_mem = self._ff1_3_out_mem(mode)

        if use_dram_decode_linear_gate_up:
            dram_act = ttnn.DRAM_MEMORY_CONFIG
            w1_out = ttnn.linear(
                x,
                self.w1,
                dtype=activation_dtype or ttnn.bfloat16,
                compute_kernel_config=li_ff1_3_compute_kernel_cfg,
                memory_config=dram_act,
            )
            w3_out = ttnn.linear(
                x,
                self.w3,
                dtype=activation_dtype or ttnn.bfloat16,
                compute_kernel_config=li_ff1_3_compute_kernel_cfg,
                memory_config=dram_act,
            )
        else:
            w1_out = ttnn.linear(
                x,
                self.w1,
                dtype=ttnn.bfloat8_b if TG else activation_dtype or ttnn.bfloat16,
                core_grid=None,
                compute_kernel_config=li_ff1_3_compute_kernel_cfg,
                program_config=pc_1,
                memory_config=ff1_3_mem,
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
                memory_config=ff1_3_mem,
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

        mul_out_mem = (
            ttnn.DRAM_MEMORY_CONFIG
            if use_dram_decode_linear_gate_up
            else (ttnn.L1_MEMORY_CONFIG if self._dram_intermediates() else w1_out.memory_config())
        )
        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[self.activation_type],
            dtype=ttnn.bfloat16 if use_dram_decode_linear_gate_up else (activation_dtype or ttnn.bfloat8_b),
            memory_config=mul_out_mem,
        )

        if mode == Mode.DECODE and not TG and self.prefetcher is None:
            w2_in = devstral_to_memory_if_needed(w2_in, self._binary_mult_mem(mode))

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
                w2_in = devstral_to_memory_if_needed(w2_in, ttnn.L1_MEMORY_CONFIG)

        li_ff2_compute_kernel_cfg = self.decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_FF2, configuration=self.args
        )

        ff2_mem = self._ff2_out_mem(mode)

        # ``get_mlp_ff2_prg_config`` uses a wide reuse matmul for prefill when ``seq_len <= 128``; on
        # wide FFN + multi-device (BH or WH T3K) that program's **static** L1 circular buffers can
        # exceed ~1.5 MiB for BF16 FFN (failure at ``ttnn.linear`` compile). The ``seq_len > 128``
        # branch already switches to ``minimal_matmul`` + ``MinimalMatmulConfig``; use that same path
        # whenever we take the DRAM mitigations for wide FFN. This is not a host deallocation
        # issue—buffers are fixed at kernel compile time from ``program_config``.
        # ``dram_matmul_config`` decode FF2 (``pc_2`` + interleaved DRAM ``w2``) requires width-sharded
        # weights; decode still uses ``pc_2`` until FF2 weights can be DRAM-sharded without static CB
        # overflow at load, or matmul supports interleaved DRAM ``B``.
        if use_minimal_matmul_w2:
            grid = self.args.mlp2_grid(seq_len)
            m_blk = int(os.environ.get("DEVSTRAL2_MINIMAL_MM_M_BLOCK", "8"))
            if m_blk not in (4, 8, 16):
                m_blk = 16
            # Explicit 2×2 subblocks (default ctor used 1×1 here). Device perf report recommended
            # a larger output subblock product for this matmul shape; larger M blocks when divisible.
            pc_2_minimal = ttnn.MinimalMatmulConfig(
                M_block_size=m_blk,
                K_block_size=8,
                N_block_size=8,
                subblock_h=2,
                subblock_w=2,
                compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
            )
            w2_out = ttnn.experimental.minimal_matmul(
                w2_in,
                self.w2,
                compute_kernel_config=li_ff2_compute_kernel_cfg,
                config=pc_2_minimal,
            )
        elif use_dram_decode_linear_gate_up:
            w2_out = ttnn.linear(
                w2_in,
                self.w2,
                dtype=activation_dtype or ttnn.bfloat16,
                compute_kernel_config=li_ff2_compute_kernel_cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            w2_out = ttnn.linear(
                w2_in,
                self.w2,
                compute_kernel_config=li_ff2_compute_kernel_cfg,
                dtype=self.args.ccl_dtype if TG else activation_dtype or ttnn.bfloat16,
                program_config=pc_2,
                memory_config=ff2_mem,
                core_grid=None,
                global_cb=self.prefetcher.global_cb if self.prefetcher is not None and mode == Mode.DECODE else None,
                sub_device_id=self.prefetcher.worker_sub_device_id
                if self.prefetcher is not None and mode == Mode.DECODE
                else None,
            )
        ttnn.deallocate(w2_in)

        ar_out_mem = self._ff2_ar_out_mem(mode, w2_out)
        w2_out_reduced = tt_all_reduce(
            w2_out,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=0,
            dim=0 if (TG and self.dim < 8192) else 3,
            sharded=(mode == Mode.DECODE),
            memory_config=ar_out_mem,
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
            w2_out_reduced = devstral_to_memory_if_needed(
                w2_out_reduced,
                self.args.get_mlp_output_mem_config(mode, self.prefetcher),
            )

        return w2_out_reduced


TtMinistralMLP = TtDevstral2LargeMLP

__all__ = [
    "TtDevstral2LargeMLP",
    "TtMinistralMLP",
]
