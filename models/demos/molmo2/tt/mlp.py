# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Molmo2 text decoder SwiGLU MLP — column-parallel (TP) layout.

Mirrors the text-attention AllGather pattern for trace compatibility:
  w1/w3 (gate/value): column-parallel — each device [4096, intermediate/8] columns.
  w2 (down):          replicated      — each device holds the full [intermediate, 4096].

After SwiGLU: ttnn.all_gather combines partial [S, intermediate/8] outputs
into [S, intermediate] on each device, then replicated w2 produces [S, 4096].

This is the same pattern as attention column-parallel QKV + AllGather + replicated wo.
It is trace-safe (AllGather works in decode-trace replay; AllReduce does not).

Memory: AllGather intermediate [S, intermediate] bfloat16 is ≤201 MB at S=8192.
For S > 8192 (e.g. 384-frame videos), Phase 2 should use AllReduce in the
non-traced prefill path with row-parallel w2.

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
    """Text decoder SwiGLU MLP — column-parallel w1/w3 + AllGather + replicated w2."""

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
            (lambda n: weight_cache_path / f"{layer_name}.tp8ag.{n}")
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

        # Column-parallel: shard output (intermediate) dim across devices
        col_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=3)

        def _col(w, name):
            # bfloat16 for sharded w1/w3: each device holds [4096, 1536] = 12 MB.
            # Total across 36 layers × 2 weights: 36×2×4096×1536×2B = 901 MB/device —
            # less than bfloat8_b replicated (5.4 GB/device) and eliminates the 4%
            # accuracy regression from per-shard bfloat8_b quantization noise.
            return ttnn.as_tensor(
                w.T.unsqueeze(0).unsqueeze(0),  # [1, 1, 4096, 12288] → each device [4096, 1536]
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=col_mapper,
                cache_file_name=cache_name(name),
            )

        self.w1 = _col(w_gate, "w1_gate_col")
        self.w3 = _col(w_value, "w3_value_col")

        # w2 replicated: each device holds the full [12288, 4096] weight.
        # After AllGather the SwiGLU output is [S, 12288] on each device,
        # so replicated w2 computes the full [S, 4096] output independently.
        ff_out = state_dict[f"{layer_name}.ff_out.weight"]  # [4096, 12288]
        self.w2 = ttnn.as_tensor(
            ff_out.T.unsqueeze(0).unsqueeze(0),  # [1, 1, 12288, 4096]
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            cache_file_name=cache_name("w2_down_replicated"),
        )

    def forward(self, x: ttnn.Tensor, mode: Mode) -> ttnn.Tensor:
        """Column-parallel SwiGLU + AllGather + replicated w2.

        Same AllGather pattern as attention (column-parallel QKV → AllGather → replicated wo).
        Trace-safe: AllGather replays correctly; AllReduce does not.
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

        # silu(gate) * value  — each device: [S, intermediate/num_devices]
        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
            memory_config=w1_out.memory_config(),
        )
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)

        # AllGather: [S, intermediate/num_devices] → [S, intermediate] on each device.
        # Equivalent to attention's AllGather after per-head computation.
        # Trace-safe (unlike AllReduce).
        w2_in_full = ttnn.all_gather(
            w2_in,
            dim=3,
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(w2_in)

        # Replicated w2 projects [S, intermediate] → [S, dim] independently on each device
        out = ttnn.linear(
            w2_in_full,
            self.w2,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(w2_in_full)
        return out
