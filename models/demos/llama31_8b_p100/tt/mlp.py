# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from models.ttt_compat.tt.mlp import *  # noqa: F401,F403  # isort: skip
import torch

import models.ttt_compat.tt.mlp as _tt
import ttnn
from models.demos.llama31_8b_p100.tt import wiring
from models.ttt_compat.tt.ccl import tt_all_reduce
from models.ttt_compat.tt.common import Mode, pad_to_size
from models.ttt_compat.tt.model_config import OpGroup, TensorGroup


class MLP(_tt.MLP):
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
        super().__init__(
            mesh_device,
            tt_ccl,
            args,
            state_dict,
            weight_cache_path,
            layer_num,
            dtype,
            model_config,
            state_dict_prefix=state_dict_prefix,
            prefetcher=prefetcher,
        )
        self._matmul_bh = wiring.on("matmul_bh", args)
        self.w13 = None
        if self._matmul_bh is not None and prefetcher is None:
            prefix = state_dict_prefix or args.get_state_dict_prefix(_tt.MLP.__name__, layer_num)
            hidden = f".hidden_dim_{args.hidden_dim}" if args.hidden_dim != args.unpadded_hidden_dim else ""
            cache = None if args.dummy_weights else weight_cache_path / f"{prefix}.w13_tile_interleaved{hidden}"
            real = self._matmul_bh.mlp_gate_up(args, state_dict, prefix, layer_num, cache)
            gate = pad_to_size(
                torch.transpose(real[0] if real else state_dict[f"{prefix}.w1.weight"], -2, -1),
                dim=-1,
                size=args.hidden_dim,
            )
            up = pad_to_size(
                torch.transpose(real[1] if real else state_dict[f"{prefix}.w3.weight"], -2, -1),
                dim=-1,
                size=args.hidden_dim,
            )
            k, n = gate.shape
            gate_up = torch.stack([gate.reshape(k, n // 32, 32), up.reshape(k, n // 32, 32)], dim=2).reshape(k, 2 * n)
            self.w13 = ttnn.as_tensor(
                gate_up.unsqueeze(0).unsqueeze(0),
                dtype=self.ff1_3_dtype,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(-2, -1), mesh_shape=args.cluster_shape),
                layout=ttnn.TILE_LAYOUT,
                memory_config=args.create_dram_sharded_mem_config(args.dim, 2 * n),
                cache_file_name=cache,
            )

    def forward(self, x: ttnn.Tensor, mode: Mode) -> ttnn.Tensor:
        seq_len = x.shape[-2]
        chunk_size = self.model_config.get("MLP_PREFILL_CHUNK_SIZE", seq_len)
        if mode == Mode.PREFILL and seq_len > chunk_size:
            outputs = [
                self.forward(x[..., start : min(start + chunk_size, seq_len), :], mode)
                for start in range(0, seq_len, chunk_size)
            ]
            ttnn.deallocate(x)
            output = ttnn.concat(outputs, dim=-2, memory_config=outputs[0].memory_config())
            for chunk in outputs:
                ttnn.deallocate(chunk)
            return output
        TG = self.args.is_galaxy
        layer_num = max(self.layer_num, 0)
        activation_dtype = self.decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.ACTIVATION
        )
        li_ff1_3_compute_kernel_cfg = self.decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_FF1_FF3, configuration=self.args
        )

        if mode == Mode.PREFILL and seq_len >= self.args.prefill_len_cutoff:
            x = ttnn.reshape(x, [1, seq_len // self.args.prefill_len_cutoff, self.args.prefill_len_cutoff, -1])

        pc_1 = self.args.get_mlp_ff1_3_prg_config(mode, seq_len, self.prefetcher)
        pc_2 = self.args.get_mlp_ff2_prg_config(mode, seq_len, self.prefetcher)
        pc_3 = self.args.get_mlp_ff1_3_prg_config(mode, seq_len, self.prefetcher)

        use_tg_decode_no_prefetch = TG and mode == Mode.DECODE and self.prefetcher is None
        if use_tg_decode_no_prefetch:
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        w2_in = None
        if self.w13 is not None and mode == Mode.PREFILL and not TG:
            w2_in = wiring.swiglu(
                self._matmul_bh,
                x,
                self.w13,
                dtype=activation_dtype or ttnn.bfloat8_b,
                compute_kernel_config=li_ff1_3_compute_kernel_cfg,
                program_config=pc_1,
                memory_config=self.args.get_mlp_ff1_3_mem_config(mode, self.prefetcher),
            )
            if w2_in is not None:
                ttnn.deallocate(x)
        if w2_in is None:
            w1_out = wiring.linear(
                self._matmul_bh,
                x,
                self.w1,
                dtype=ttnn.bfloat8_b if TG else activation_dtype or ttnn.bfloat16,
                core_grid=None,
                compute_kernel_config=li_ff1_3_compute_kernel_cfg,
                program_config=None if use_tg_decode_no_prefetch else pc_1,
                memory_config=(
                    ttnn.DRAM_MEMORY_CONFIG
                    if use_tg_decode_no_prefetch
                    else self.args.get_mlp_ff1_3_mem_config(mode, self.prefetcher)
                ),
                global_cb=self.prefetcher.global_cb if self.prefetcher is not None and mode == Mode.DECODE else None,
                sub_device_id=self.prefetcher.worker_sub_device_id
                if self.prefetcher is not None and mode == Mode.DECODE
                else None,
            )
            w3_out = wiring.linear(
                self._matmul_bh,
                x,
                self.w3,
                dtype=ttnn.bfloat8_b if TG else activation_dtype or ttnn.bfloat16,
                core_grid=None,
                compute_kernel_config=li_ff1_3_compute_kernel_cfg,
                program_config=None if use_tg_decode_no_prefetch else pc_3,
                memory_config=(
                    ttnn.DRAM_MEMORY_CONFIG
                    if use_tg_decode_no_prefetch
                    else self.args.get_mlp_ff1_3_mem_config(mode, self.prefetcher)
                ),
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
                        memory_config=self.model_config["FF1_OUT_REDUCE_SCATTER_MEMCFG"]
                        if mode == Mode.DECODE
                        else None,
                        intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        topology=self.args.ccl_topology(),
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
                        memory_config=self.model_config["FF1_OUT_REDUCE_SCATTER_MEMCFG"]
                        if mode == Mode.DECODE
                        else None,
                        intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        topology=self.args.ccl_topology(),
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
                        sharded=mode == Mode.DECODE and not use_tg_decode_no_prefetch,
                        topology=self.args.ccl_topology(),
                        memory_config=(
                            ttnn.DRAM_MEMORY_CONFIG
                            if use_tg_decode_no_prefetch
                            else self.model_config["FF1_OUT_GATHERED_MEMCFG"]
                            if mode == Mode.DECODE
                            else None
                        ),
                    )
                    w3_out = tt_all_reduce(
                        w3_out,
                        self.mesh_device,
                        self.tt_ccl,
                        cluster_axis=1,
                        num_all_gather_links=2,
                        sharded=mode == Mode.DECODE and not use_tg_decode_no_prefetch,
                        topology=self.args.ccl_topology(),
                        memory_config=(
                            ttnn.DRAM_MEMORY_CONFIG
                            if use_tg_decode_no_prefetch
                            else self.model_config["FF1_OUT_GATHERED_MEMCFG"]
                            if mode == Mode.DECODE
                            else None
                        ),
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
                topology=self.args.ccl_topology(),
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

        w2_output_dtype = activation_dtype
        if w2_output_dtype is None:
            w2_output_dtype = ttnn.bfloat16

        if mode != Mode.DECODE and self.args.use_minimal_prefill_matmul(seq_len):
            is_qwen3_32b_t3k = self.args.base_model_name == "Qwen3-32B" and self.args.device_name == "T3K"
            if not is_qwen3_32b_t3k:
                w2_output_dtype = None

            w2_out = wiring.minimal_matmul(
                self._matmul_bh,
                w2_in,
                self.w2,
                dtype=w2_output_dtype,
                compute_kernel_config=li_ff2_compute_kernel_cfg,
                config=pc_2,
            )
        else:
            if TG:
                w2_output_dtype = self.args.ccl_dtype

            w2_out = wiring.linear(
                self._matmul_bh,
                w2_in,
                self.w2,
                compute_kernel_config=li_ff2_compute_kernel_cfg,
                dtype=w2_output_dtype,
                program_config=None if use_tg_decode_no_prefetch else pc_2,
                memory_config=(
                    ttnn.DRAM_MEMORY_CONFIG
                    if use_tg_decode_no_prefetch
                    else self.args.get_mlp_ff2_mem_config(mode, self.prefetcher)
                ),
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
            sharded=mode == Mode.DECODE and not use_tg_decode_no_prefetch,
            memory_config=(
                ttnn.DRAM_MEMORY_CONFIG
                if use_tg_decode_no_prefetch
                else self.args.get_mlp_ff2_all_reduce_mem_config(mode, w2_out)
            ),
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
