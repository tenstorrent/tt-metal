# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Molmo2 text decoder SwiGLU MLP — column-parallel (TP) layout.

Follows the tt_transformers LLaMA MLP pattern for T3K (1×8 mesh):
  w1/w3: ShardTensor2dMesh(dims=(-2,-1)) — column-parallel, output dim sharded.
  w2:    ShardTensor2dMesh(dims=(-1,-2)) — row-parallel, input dim sharded.

AllReduce after w2 via two CCL async ops (trace-safe, matches LLaMA decode trace):
  1. reduce_scatter_minimal_async  — sums partial w2 outputs, scatters [S, dim/8]
  2. tt_all_gather(cluster_axis=None) — reassembles [S, dim] replicated

This avoids plain ttnn.all_reduce (broken in decode traces) and plain
ttnn.all_gather (ring-ordering mismatch causing ~7% accuracy regression).

Key difference from standard LLaMA SwiGLU:
  ff_proj [24576, 4096] stores value in FIRST half and gate in SECOND half.
    w1 (gate)  = ff_proj[12288:, :]  → silu
    w3 (value) = ff_proj[:12288, :]  → multiply with silu output
"""

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.ccl import tt_all_gather, tt_all_reduce
from models.tt_transformers.tt.common import Mode


class TtMolmo2TextMLP(LightweightModule):
    """Text decoder SwiGLU MLP — column-parallel w1/w3, row-parallel w2, async AllReduce."""

    def __init__(
        self,
        mesh_device,
        tt_ccl,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        configuration,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.num_devices = configuration.num_devices
        self.intermediate_size = configuration.intermediate_size  # 12288
        self.compute_kernel_config_hifi2 = configuration.compute_kernel_config_hifi2

        layer_name = f"model.transformer.blocks.{layer_num}.mlp"
        cache_name = (
            (lambda n: weight_cache_path / f"{layer_name}.tp8ar.{n}")
            if weight_cache_path and not configuration.dummy_weights
            else (lambda _: None)
        )

        # ------------------------------------------------------------------ #
        # Split fused ff_proj [24576, 4096]:
        #   first half  [:12288] = value (up) → w3
        #   second half [12288:] = gate      → w1
        # Standard SwiGLU: silu(w1(x)) * w3(x)
        # ------------------------------------------------------------------ #
        ff_proj = state_dict[f"{layer_name}.ff_proj.weight"]  # [24576, 4096]
        w_value = ff_proj[: self.intermediate_size]  # [12288, 4096]
        w_gate = ff_proj[self.intermediate_size :]  # [12288, 4096]

        # Match tt_transformers LLaMA T3K dims:
        #   w1/w3 column-parallel: dims=(-2,-1) → shard last dim (hidden) across 8 col devices
        #   w2   row-parallel:     dims=(-1,-2) → shard second-to-last dim (hidden) across 8 col devices
        w1_dims = (-2, -1)
        w2_dims = (-1, -2)

        def _col(w, name):
            """Column-parallel w1/w3: [4096, 12288] → each device [4096, 1536]."""
            return ttnn.as_tensor(
                w.T.unsqueeze(0).unsqueeze(0),  # [1, 1, 4096, 12288]
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=w1_dims, mesh_shape=configuration.cluster_shape),
                cache_file_name=cache_name(name),
            )

        def _row(w, name):
            """Row-parallel w2: [12288, 4096] → each device [1536, 4096]."""
            return ttnn.as_tensor(
                w.T.unsqueeze(0).unsqueeze(0),  # [1, 1, 12288, 4096]
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=w2_dims, mesh_shape=configuration.cluster_shape),
                cache_file_name=cache_name(name),
            )

        self.w1 = _col(w_gate, "w1_gate_col")
        self.w3 = _col(w_value, "w3_value_col")

        ff_out = state_dict[f"{layer_name}.ff_out.weight"]  # [4096, 12288]
        self.w2 = _row(ff_out, "w2_down_row")

    def forward(self, x: ttnn.Tensor, mode: Mode) -> ttnn.Tensor:
        """Column-parallel SwiGLU + row-parallel w2 + async AllReduce.

        AllReduce = reduce_scatter_minimal_async + tt_all_gather(cluster_axis=None).
        Both CCL async ops replay correctly in TTNN decode traces (unlike ttnn.all_reduce).
        """
        seq_len = x.shape[-2]
        prefill_chunk = 1024
        chunked = mode == Mode.PREFILL and seq_len >= prefill_chunk and seq_len % prefill_chunk == 0
        if chunked:
            x = ttnn.reshape(x, [1, seq_len // prefill_chunk, prefill_chunk, -1])

        w1_out = ttnn.linear(
            x,
            self.w1,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        w3_out = ttnn.linear(
            x,
            self.w3,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)

        # silu(gate) * value — each device: [S, intermediate/num_devices]
        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
            memory_config=w1_out.memory_config(),
        )
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)

        if chunked:
            w2_in = ttnn.reshape(w2_in, (1, 1, seq_len, -1))

        # Row-parallel w2: each device contributes [S, dim] partial sum
        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(w2_in)

        # AllReduce via two trace-safe CCL async ops (matches LLaMA decode trace pattern):
        #   Step 1 — reduce_scatter: sums 8 partial [S,dim] outputs, scatters → [S, dim/8] per device
        #   Step 2 — all_gather:     reassembles → [S, dim] replicated on each device
        # For single device: both are no-ops, output == w2_out.
        # For T3K: tt_all_reduce does reduce_scatter (deallocates w2_out internally);
        #          tt_all_gather(cluster_axis=None) does all_gather_async.
        scattered = tt_all_reduce(
            w2_out,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=0,  # T3K path uses index-2 semaphores internally regardless
            dim=3,
            topology=ttnn.Topology.Linear,
        )
        out = tt_all_gather(
            scattered,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=None,  # None avoids the cluster_axis=1 no-op guard for T3K
            dim=3,
        )
        if out is not scattered:
            scattered.deallocate(True)

        # Restore [1, 1, S, dim] shape
        original_shape = out.shape
        if original_shape[0] != 1 or original_shape[1] != 1:
            out = ttnn.reshape(
                out, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
            )
        return out

    def forward_decode(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Optimized single-token decode: all projections in L1, async RS+AG.

        Uses tt_all_reduce + tt_all_gather (async reduce_scatter + all_gather)
        which enables the device CQ to overlap CCL with next-layer compute —
        matching the async pattern from forward() that achieves better pipelining.
        """
        w1_out = ttnn.linear(
            x,
            self.w1,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        w3_out = ttnn.linear(
            x,
            self.w3,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)

        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)

        out = ttnn.linear(
            w2_in,
            self.w2,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(w2_in)

        # Async RS+AG — same as forward(), enables CCL-compute overlap in device CQ
        scattered = tt_all_reduce(
            out,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=ttnn.Topology.Linear,
        )
        out = tt_all_gather(
            scattered,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=None,
            dim=3,
            topology=ttnn.Topology.Linear,
        )
        if out is not scattered:
            scattered.deallocate(True)
        return out
