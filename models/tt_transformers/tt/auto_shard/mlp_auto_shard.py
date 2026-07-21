# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Auto-sharded MLP for a 2D mesh.

MLP is the same column-parallel -> row-parallel pattern as attention, so it is driven by the same
Sharding descriptor (sharding.select_sharding):

    w1, w3 [dim, hidden_dim]   column-parallel (like wqkv): intermediate split on the head axis,
                               dim (the contraction) split on the hidden axis. Each is followed by
                               an all-reduce over the hidden axis.
    w2     [hidden_dim, dim]   row-parallel (like wo): intermediate (the contraction) split on the
                               head axis, dim (the output) split on the hidden axis. Followed by an
                               all-reduce over the head axis.

One code path runs decode and prefill on any mesh; there is no galaxy / prefetcher special-casing.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.auto_shard.ccl_auto_shard import (
    all_reduce,
    core_grid_program_config,
    log_default_grid,
    matmul_reduce_scatter,
)
from models.tt_transformers.tt.common import Mode, pad_to_size
from models.tt_transformers.tt.model_config import OpGroup, TensorGroup
from models.tt_transformers.tt.auto_shard.cost_model.sharding import MLPShapes, cache_tag, select_sharding, workload_from_config


class MLP(LightweightModule):
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
        print("Auto Sharding MLP")
        super().__init__()

        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.args = args
        self.dim = args.dim
        self.hidden_dim = args.hidden_dim
        self.model_config = model_config
        self.layer_num = layer_num
        self.ccl_dtype = args.ccl_dtype
        self.ccl_topology = args.ccl_topology()

        # One sharding descriptor fixes the w1/w3/w2 mesh placements and the two all-reduce axes.
        # Rank for the real run's workload (same source as attention), params.py fallback otherwise.
        shapes = MLPShapes(self.dim, self.hidden_dim)
        prefill_len, decode_steps = workload_from_config(args)
        self.sharding = select_sharding(
            mesh_device, shapes, tuple(args.cluster_shape), prefill_len=prefill_len, decode_steps=decode_steps
        )

        layer = max(layer_num, 0)  # cross_block uses the configuration of the first decoder
        decoders_optimizations = args.decoders_optimizations
        ff1_3_dtype = decoders_optimizations.get_tensor_dtype(decoder_id=layer, tensor=TensorGroup.FF1_FF3)
        ff2_dtype = decoders_optimizations.get_tensor_dtype(decoder_id=layer, tensor=TensorGroup.FF2)
        self.activation_dtype = decoders_optimizations.get_tensor_dtype(decoder_id=layer, tensor=TensorGroup.ACTIVATION)
        self.li_ff1_3_compute_kernel_cfg = decoders_optimizations.get_math_fidelity(
            decoder_id=layer, op=OpGroup.LI_FF1_FF3, configuration=args
        )
        self.li_ff2_compute_kernel_cfg = decoders_optimizations.get_math_fidelity(
            decoder_id=layer, op=OpGroup.LI_FF2, configuration=args
        )

        state_dict_prefix = state_dict_prefix or args.get_state_dict_prefix(self.__class__.__name__, layer_num)

        # Tagged with the mesh shape AND the placement (see sharding.cache_tag): the fused gate_up
        # interleave below is built around num_intermediate_shards, so its bytes are placement-
        # specific and a mesh-only tag would let two layouts collide on one filename. Without a cache
        # every layer redoes the transpose/pad/chunk-cat plus the tilize+quantize+shard on every run.
        tag = cache_tag(self.sharding, args.cluster_shape)
        if args.dummy_weights or (weight_cache_path is None):
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / (f"{state_dict_prefix}.{name}_{tag}")

        def as_sharded_tensor(name, dtype, dims, pad_dim):
            # stored weight is [out, in]; transpose to [in, out], pad the hidden_dim axis, make 4D.
            w = state_dict[f"{state_dict_prefix}.{name}.weight"].transpose(-2, -1)
            w = pad_to_size(w, dim=pad_dim, size=args.hidden_dim)
            w = w.unsqueeze(0).unsqueeze(0)
            return ttnn.as_tensor(
                w,
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=dims, mesh_shape=args.cluster_shape),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_name(f"{name}_sharded"),
            )

        # --- unfused path (kept for reference) ---
        # w1, w3 are column-parallel (col_dims); w2 is row-parallel (row_dims). hidden_dim is the
        # last axis of w1/w3 ([dim, hidden_dim] -> pad -1) and the second-to-last of w2
        # ([hidden_dim, dim] -> pad -2).
        # self.w1 = as_sharded_tensor("w1", ff1_3_dtype, self.sharding.col_dims, pad_dim=-1)
        # self.w3 = as_sharded_tensor("w3", ff1_3_dtype, self.sharding.col_dims, pad_dim=-1)

        # --- fused gate+up path ---
        # w1 (gate) and w3 (up) are fused into one column-parallel weight [dim, 2*hidden_dim].
        # The mesh splits the hidden axis into S = num_intermediate_shards chunks, so we interleave
        # per shard -> [gate_0|up_0 | gate_1|up_1 | ...] and device i receives a contiguous
        # [gate_i | up_i]. That lets forward() recover the two halves with one mid-point split.
        def load_col_weight(name):
            # stored [out=hidden, in=dim]; transpose to [dim, hidden] then pad the hidden axis.
            w = state_dict[f"{state_dict_prefix}.{name}.weight"].transpose(-2, -1)
            return pad_to_size(w, dim=-1, size=args.hidden_dim)

        S = self.sharding.num_intermediate_shards
        w1 = load_col_weight("w1")
        w3 = load_col_weight("w3")
        gate_up = torch.cat(
            [
                torch.cat([torch.chunk(w1, S, dim=-1)[i], torch.chunk(w3, S, dim=-1)[i]], dim=-1)
                for i in range(S)
            ],
            dim=-1,
        )
        gate_up = gate_up.unsqueeze(0).unsqueeze(0)  # [1, 1, dim, 2*hidden_dim]
        self.w1_w3 = ttnn.as_tensor(
            gate_up,
            dtype=ff1_3_dtype,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=self.sharding.col_dims, mesh_shape=args.cluster_shape),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("w1_w3_fused"),
        )

        # w2 (down) is row-parallel (row_dims). hidden_dim is the second-to-last axis
        # ([hidden_dim, dim] -> pad -2).
        self.w2 = as_sharded_tensor("w2", ff2_dtype, self.sharding.row_dims, pad_dim=-2)

        self.activation_type = (
            args.mlp_activation_type if hasattr(args, "mlp_activation_type") else ttnn.UnaryOpType.SILU
        )

    def forward(self, x: ttnn.Tensor, mode: Mode) -> ttnn.Tensor:
        """
        w1 -> gate_proj, w3 -> up_proj, w2 -> down_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        seq_len = x.shape[-2]

        # Reshape long prefill sequences to fit on device and parallelize the matmul.
        if mode == Mode.PREFILL and seq_len >= self.args.prefill_len_cutoff:
            x = ttnn.reshape(x, [1, seq_len // self.args.prefill_len_cutoff, self.args.prefill_len_cutoff, -1])

        # === VERSION 1: unfused w1/w3 (kept for reference) ===
        # w1 (gate) and w3 (up): column-parallel. Reduce each over the hidden axis (replicate=True:
        # every chip needs its full intermediate slice for the elementwise product and w2).
        # w1_out = ttnn.linear(
        #     x,
        #     self.w1,
        #     dtype=self.activation_dtype or ttnn.bfloat16,
        #     compute_kernel_config=self.li_ff1_3_compute_kernel_cfg,
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        # )
        # w3_out = ttnn.linear(
        #     x,
        #     self.w3,
        #     dtype=self.activation_dtype or ttnn.bfloat16,
        #     compute_kernel_config=self.li_ff1_3_compute_kernel_cfg,
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        # )
        # ttnn.deallocate(x)
        # w1_out = all_reduce(w1_out, self.mesh_device, axis=self.sharding.reduce_col_over,
        #                     replicate=True, dtype=self.ccl_dtype, topology=self.ccl_topology)
        # w3_out = all_reduce(w3_out, self.mesh_device, axis=self.sharding.reduce_col_over,
        #                     replicate=True, dtype=self.ccl_dtype, topology=self.ccl_topology)

        # === VERSION 2: fused gate+up matmul, unfused (serialized) all_reduce (kept for reference) ===
        # One matmul against the fused [dim, 2*hidden_dim] weight, then a separate all-reduce over
        # the hidden axis (replicate=True: every chip needs its full intermediate slice).
        # gate_up = ttnn.linear(
        #     x,
        #     self.w1_w3,
        #     dtype=self.activation_dtype or ttnn.bfloat16,
        #     compute_kernel_config=self.li_ff1_3_compute_kernel_cfg,
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        # )
        # ttnn.deallocate(x)
        # gate_up = all_reduce(gate_up, self.mesh_device, axis=self.sharding.reduce_col_over,
        #                      replicate=True, dtype=self.ccl_dtype, topology=self.ccl_topology)

        # === VERSION 3: fused gate+up matmul + fused reduce_scatter (overlapped) ===
        # replicate=True -> only the reduce_scatter half fuses/overlaps with the matmul; the helper
        # still appends a (serialized) all_gather to give every chip the full slice for the product.
        # MLP_FF13_CORES=<n> pins the FF1/FF3 (gate_up) width-sharded matmul grid; unset = ttnn default.
        log_default_grid(x, self.w1_w3, "MLP gate_up")
        ff13_pc = core_grid_program_config(
            self.args, "MLP_FF13_CORES", x.shape[-2], x.shape[-1], self.w1_w3.shape[-1], "MLP gate_up"
        )
        gate_up = matmul_reduce_scatter(
            x,
            self.w1_w3,
            self.mesh_device,
            self.tt_ccl,
            axis=self.sharding.reduce_col_over,
            replicate=True,
            dtype=self.ccl_dtype,
            compute_kernel_config=self.li_ff1_3_compute_kernel_cfg,
            program_config=ff13_pc,
            topology=self.ccl_topology,
            label=f"MLP gate_up [{mode}]",
        )
        ttnn.deallocate(x)

        # Each device holds a contiguous [gate_i | up_i]; slice down the middle to recover both.
        # (ttnn.split's two-chunk kernel is currently broken to build, so use two slices.)
        shape = list(gate_up.shape)
        half = shape[-1] // 2
        ends_gate = shape[:-1] + [half]
        begins_up = [0] * (len(shape) - 1) + [half]
        w1_out = ttnn.slice(gate_up, [0] * len(shape), ends_gate)
        w3_out = ttnn.slice(gate_up, begins_up, shape)
        ttnn.deallocate(gate_up)

        # SILU(w1) * w3
        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[self.activation_type],
            dtype=self.activation_dtype or ttnn.bfloat8_b,
            memory_config=w1_out.memory_config(),
        )
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)

        # w2 (down): row-parallel. Reduce over the head axis (replicate=False: on a line the output
        # stays split along the head axis, matching the fractured residual the decoder expects).

        # --- unfused path (kept for reference) ---
        # w2_out = ttnn.linear(
        #     w2_in,
        #     self.w2,
        #     dtype=self.activation_dtype or ttnn.bfloat16,
        #     compute_kernel_config=self.li_ff2_compute_kernel_cfg,
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        # )
        # ttnn.deallocate(w2_in)
        # w2_out = all_reduce(w2_out, self.mesh_device, axis=self.sharding.reduce_row_over,
        #                     replicate=False, dtype=self.ccl_dtype, topology=self.ccl_topology)

        # --- fused path: matmul + reduce_scatter overlapped (replicate=False -> no trailing
        # all_gather; the width-sharded result is exactly what the decoder's residual expects).
        # Falls back to the unfused numerics on anything but a 1D ring.
        # MLP_FF2_CORES=<n> forces the FF2 (down) matmul core grid; unset = ttnn default.
        log_default_grid(w2_in, self.w2, "MLP down")
        ff2_pc = core_grid_program_config(
            self.args, "MLP_FF2_CORES", w2_in.shape[-2], w2_in.shape[-1], self.w2.shape[-1], "MLP down"
        )
        w2_out = matmul_reduce_scatter(
            w2_in,
            self.w2,
            self.mesh_device,
            self.tt_ccl,
            axis=self.sharding.reduce_row_over,
            replicate=False,
            dtype=self.ccl_dtype,
            compute_kernel_config=self.li_ff2_compute_kernel_cfg,
            program_config=ff2_pc,
            topology=self.ccl_topology,
            label=f"MLP down [{mode}]",
        )
        ttnn.deallocate(w2_in)

        # Collapse the prefill parallelism dims back to [1, 1, seq, dim].
        s = w2_out.shape
        return ttnn.reshape(w2_out, (1, 1, s[-4] * s[-3] * s[-2], s[-1]))
