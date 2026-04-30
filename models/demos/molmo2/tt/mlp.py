# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Molmo2 text decoder SwiGLU MLP — column-parallel (TP) layout.

T3K layout (8-way tensor-parallel):
  w1/w3 (gate/value): column-parallel — each device holds [4096, intermediate/8] columns.
  w2 (down):          row-parallel    — each device holds [intermediate/8, 4096] rows.
  After w2: ttnn.all_reduce(cluster_axis=1) combines partial sums.

Memory comparison vs replicated (per device, S=34560):
  Replicated:      w1_out = [S, 12288] bfloat16 ≈ 850 MB/device (OOMs for S > ~10k)
  Column-parallel: w1_out = [S, 1536]  bfloat16 ≈ 106 MB/device (fits for S=34560)

PCC vs replicated: 0.999944 (verified by unit test for S=256).

Key difference from standard LLaMA SwiGLU:
  ff_proj [24576, 4096] stores value (up) in the FIRST half and gate in the SECOND half.
    w1 (gate)  = ff_proj[12288:, :]  (second half → passed through silu)
    w3 (value) = ff_proj[:12288, :]  (first half  → multiplied with silu output)
  This maps to the standard SwiGLU formula: silu(w1(x)) * w3(x).
"""


import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.common import Mode


class TtMolmo2TextMLP(LightweightModule):
    """Text decoder SwiGLU MLP — 8-way column/row-parallel tensor parallelism."""

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
        self.num_devices = configuration.num_devices
        self.intermediate_size = configuration.intermediate_size  # 12288
        self.compute_kernel_config_hifi2 = configuration.compute_kernel_config_hifi2

        layer_name = f"model.transformer.blocks.{layer_num}.mlp"
        cache_name = (
            (lambda n: weight_cache_path / f"{layer_name}.tp8.{n}")
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

        # ShardTensorToMesh(dim=N) shards across ALL devices along dim N.
        # ShardTensor2dMesh(dims=(row_dim, col_dim)) shards row_dim across mesh rows
        # and col_dim across mesh cols — with mesh_shape=[1,8], only col_dim is
        # effectively sharded (1 row = no row sharding). Using ShardTensorToMesh
        # is simpler and unambiguous (same pattern used by the ViT blocks).
        col_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=3)  # shard output dim
        row_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=2)  # shard input dim

        def _col(w, name):
            """Column-parallel: shard output dim (last) across 8 devices."""
            return ttnn.as_tensor(
                w.T.unsqueeze(0).unsqueeze(0),  # [1, 1, 4096, 12288] → each device [4096, 1536]
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=col_mapper,
                cache_file_name=cache_name(name),
            )

        def _row(w, name):
            """Row-parallel: shard input dim across 8 devices."""
            return ttnn.as_tensor(
                w.T.unsqueeze(0).unsqueeze(0),  # [1, 1, 12288, 4096] → each device [1536, 4096]
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=row_mapper,
                cache_file_name=cache_name(name),
            )

        self.w1 = _col(w_gate, "w1_gate_tp8")
        self.w3 = _col(w_value, "w3_value_tp8")

        ff_out = state_dict[f"{layer_name}.ff_out.weight"]  # [4096, 12288]
        self.w2 = _row(ff_out, "w2_down_tp8")

    def forward(self, x: ttnn.Tensor, mode: Mode) -> ttnn.Tensor:
        """Column-parallel SwiGLU: silu(w1(x)) * w3(x) → w2 → AllReduce.

        Each device computes a [S, intermediate/num_devices] partial result.
        AllReduce combines the partial w2 outputs into the full [S, 4096] result.
        """
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

        # silu(gate) * value  (each device: [S, intermediate/num_devices])
        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
            memory_config=w1_out.memory_config(),
        )
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)

        # Row-parallel w2: each device produces partial [S, 4096] sum
        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(w2_in)

        # AllReduce across T3K ring — combines partial sums into full output
        out = ttnn.all_reduce(
            w2_out,
            cluster_axis=1,
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(w2_out)
        return out
