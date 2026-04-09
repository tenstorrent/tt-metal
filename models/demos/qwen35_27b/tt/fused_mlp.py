# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fused gate+up MLP for Qwen3.5-27B decode optimization.

Concatenates W1 (gate_proj) and W3 (up_proj) into a single weight tensor,
performing one DRAM-sharded matmul instead of two. This halves the number
of DRAM weight reads for the gate/up projections, directly addressing the
DRAM bandwidth bottleneck in decode mode.
"""

import torch

import ttnn
from models.tt_transformers.tt.ccl import tt_all_reduce
from models.tt_transformers.tt.common import Mode, pad_to_size
from models.tt_transformers.tt.mlp import MLP
from models.tt_transformers.tt.model_config import OpGroup, TensorGroup


class Qwen35FusedMLP(MLP):
    """MLP with fused W1+W3 (gate+up) projection for Qwen3.5-27B.

    On P150 (non-Galaxy, no prefetcher), the MLP decode path does two
    separate DRAM-sharded matmuls for gate and up projections. Since both
    read the same input activation, fusing them into a single matmul
    saves one full DRAM weight pass per layer (64 layers total).
    """

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
            state_dict_prefix,
            prefetcher,
        )

        self.hidden_dim_tp = args.hidden_dim // args.num_devices

        # Fuse W1+W3 in torch, then upload once (avoids DRAM-sharded→interleaved limitation)
        sd_prefix = state_dict_prefix or args.get_state_dict_prefix(self.__class__.__name__, layer_num)
        w1_raw = torch.transpose(state_dict[f"{sd_prefix}.w1.weight"], -2, -1)
        w3_raw = torch.transpose(state_dict[f"{sd_prefix}.w3.weight"], -2, -1)
        # Pad individually before interleaving (no-op if hidden_dim already aligned)
        w1_raw = pad_to_size(w1_raw, dim=-1, size=args.hidden_dim)
        w3_raw = pad_to_size(w3_raw, dim=-1, size=args.hidden_dim)
        # Interleave TP shards: each device must get [w1_shard_i | w3_shard_i],
        # not a contiguous slice of [w1_full | w3_full].
        tp = args.num_devices
        w1_shards = w1_raw.reshape(w1_raw.shape[0], tp, -1)  # [dim, tp, hidden_dim/tp]
        w3_shards = w3_raw.reshape(w3_raw.shape[0], tp, -1)
        w1w3_shards = torch.cat([w1_shards, w3_shards], dim=-1)  # [dim, tp, 2*hidden_dim/tp]
        w1w3_raw = w1w3_shards.reshape(w1_raw.shape[0], -1)  # [dim, 2*hidden_dim]
        w1w3_4d = w1w3_raw.unsqueeze(0).unsqueeze(0)

        layer_num_safe = max(layer_num, 0)
        ff1_3_dtype = self.decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num_safe, tensor=TensorGroup.FF1_FF3, prefetcher=prefetcher is not None
        )
        fused_mem_config = args.create_dram_sharded_mem_config(args.dim, 2 * self.hidden_dim_tp)
        hidden_dim_string = f".hidden_dim_{args.hidden_dim}" if args.hidden_dim != args.unpadded_hidden_dim else ""
        cache_name = None if args.dummy_weights else weight_cache_path / f"{sd_prefix}.w1w3_fused{hidden_dim_string}"

        self.w1w3 = ttnn.as_tensor(
            w1w3_4d,
            dtype=ff1_3_dtype,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=args.cluster_shape),
            layout=ttnn.TILE_LAYOUT,
            memory_config=fused_mem_config,
            cache_file_name=cache_name,
        )

        # Free separate weights — fused tensor replaces them
        ttnn.deallocate(self.w1)
        ttnn.deallocate(self.w3)
        self.w1 = None
        self.w3 = None

        # Pre-compute fused matmul program configs
        self._fused_decode_pc = args.dram_matmul_config(
            m=args.tile_padded_batch_rows,
            k=args.dim,
            n=2 * self.hidden_dim_tp,
            num_cores=args.mlp_core_grid.num_cores,
        )

    def forward(self, x: ttnn.Tensor, mode: Mode) -> ttnn.Tensor:
        seq_len = x.shape[-2]
        layer_num = max(self.layer_num, 0)

        activation_dtype = self.decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.ACTIVATION
        )
        li_ff1_3_compute_cfg = self.decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_FF1_FF3, configuration=self.args
        )

        if mode == Mode.PREFILL and seq_len >= self.args.prefill_len_cutoff:
            x = ttnn.reshape(x, [1, seq_len // self.args.prefill_len_cutoff, self.args.prefill_len_cutoff, -1])

        # ── Fused W1+W3 matmul ──────────────────────────────────────────
        if mode == Mode.DECODE:
            pc = self._fused_decode_pc
            mem = self.args.get_mlp_ff1_3_mem_config(mode, self.prefetcher)
        else:
            pc = self.args.matmul_config(
                m=min(seq_len, self.args.prefill_len_cutoff),
                k=self.args.dim // self.args.cluster_shape[0],
                n=2 * self.hidden_dim_tp,
                grid_size=self.args.mlp1_3_grid(seq_len),
            )
            mem = None

        w1w3_out = ttnn.linear(
            x,
            self.w1w3,
            dtype=activation_dtype or ttnn.bfloat16,
            compute_kernel_config=li_ff1_3_compute_cfg,
            program_config=pc,
            memory_config=mem,
        )
        ttnn.deallocate(x)

        # ── Split gate and up projections ────────────────────────────────
        # Move to interleaved DRAM for cheap slicing (same pattern as GDN)
        w1w3_out = ttnn.to_memory_config(w1w3_out, ttnn.DRAM_MEMORY_CONFIG)

        if mode == Mode.DECODE:
            B = w1w3_out.shape[-2]
            w1_out = ttnn.slice(w1w3_out, (0, 0, 0, 0), (1, 1, B, self.hidden_dim_tp))
            w3_out = ttnn.slice(w1w3_out, (0, 0, 0, self.hidden_dim_tp), (1, 1, B, 2 * self.hidden_dim_tp))
        else:
            S = w1w3_out.shape[-2]
            R = w1w3_out.shape[-3]
            w1_out = ttnn.slice(w1w3_out, (0, 0, 0, 0), (1, R, S, self.hidden_dim_tp))
            w3_out = ttnn.slice(w1w3_out, (0, 0, 0, self.hidden_dim_tp), (1, R, S, 2 * self.hidden_dim_tp))
        ttnn.deallocate(w1w3_out)

        # ── SiLU(gate) * up ─────────────────────────────────────────────
        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[self.activation_type],
            dtype=activation_dtype or ttnn.bfloat8_b,
            memory_config=w1_out.memory_config(),
        )
        ttnn.deallocate(w3_out)
        ttnn.deallocate(w1_out)

        # ── Reshard for W2 (decode only, different core grid) ────────────
        if mode == Mode.DECODE and self.prefetcher is None:
            w2_in = ttnn.to_memory_config(w2_in, self.args.get_mlp_binary_mult_mem_config(mode))

        # ── W2 (down projection) ────────────────────────────────────────
        li_ff2_compute_cfg = self.decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_FF2, configuration=self.args
        )
        pc_2 = self.args.get_mlp_ff2_prg_config(mode, seq_len, self.prefetcher)

        if seq_len > 128 and mode != Mode.DECODE:
            w2_out = ttnn.experimental.minimal_matmul(
                w2_in,
                self.w2,
                compute_kernel_config=li_ff2_compute_cfg,
                config=pc_2,
            )
        else:
            w2_out = ttnn.linear(
                w2_in,
                self.w2,
                compute_kernel_config=li_ff2_compute_cfg,
                dtype=activation_dtype or ttnn.bfloat16,
                program_config=pc_2,
                memory_config=self.args.get_mlp_ff2_mem_config(mode, self.prefetcher),
            )
        ttnn.deallocate(w2_in)

        # ── All-reduce across TP devices ─────────────────────────────────
        w2_out_reduced = tt_all_reduce(
            w2_out,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            sharded=(mode == Mode.DECODE),
            memory_config=self.args.get_mlp_ff2_all_reduce_mem_config(mode, w2_out),
            rs_memory_config=self.model_config["MLP_RS_CONFIG"]["rs_memory_config"]
            if mode == Mode.DECODE
            else ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.args.ccl_dtype,
            topology=self.args.ccl_topology(),
            chunks_per_sync=self.model_config["MLP_RS_CONFIG"]["chunks_per_sync"] if mode == Mode.DECODE else 10,
            num_workers_per_link=self.model_config["MLP_RS_CONFIG"]["num_workers_per_link"]
            if mode == Mode.DECODE
            else 2,
        )

        original_shape = w2_out_reduced.shape
        w2_out_reduced = ttnn.reshape(
            w2_out_reduced,
            (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1]),
        )

        if mode == Mode.DECODE:
            w2_out_reduced = ttnn.to_memory_config(
                w2_out_reduced,
                self.args.get_mlp_output_mem_config(mode, self.prefetcher),
            )

        return w2_out_reduced
