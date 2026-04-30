# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Molmo2 text decoder SwiGLU MLP with fused ff_proj (reversed gate/value order).

Phase 1 T3K layout (correctness-first):
  All weights REPLICATED across devices — each device computes the full MLP independently.
  Input/output: [1, 1, S, 4096] replicated on all devices.

Key difference from standard LLaMA SwiGLU:
  ff_proj [24576, 4096] stores value (up) in the FIRST half and gate in the SECOND half.
  At load time:
    w1 (gate) = ff_proj[12288:, :]  (second half → passed through silu)
    w3 (value) = ff_proj[:12288, :] (first half → multiplied with silu output)
  This maps to the standard SwiGLU formula: silu(w1(x)) * w3(x).

Adapted from models/tt_transformers/tt/mlp.py (forward structure).
"""


import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.common import Mode


class TtMolmo2TextMLP(LightweightModule):
    """Text decoder SwiGLU MLP for Molmo2-8B (Phase 1: replicated weights)."""

    def __init__(
        self,
        mesh_device,
        tt_ccl,  # kept for API compatibility, not used in Phase 1
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        configuration,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.intermediate_size = configuration.intermediate_size
        self.compute_kernel_config_hifi2 = configuration.compute_kernel_config_hifi2
        self.tile_size = configuration.tile_size

        layer_name = f"model.transformer.blocks.{layer_num}.mlp"
        if configuration.dummy_weights or weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{layer_name}.{name}"

        # ------------------------------------------------------------------ #
        # Split fused ff_proj [24576, 4096]:
        #   first half  [:12288] = value (up) → w3
        #   second half [12288:] = gate      → w1
        # Standard SwiGLU: silu(w1(x)) * w3(x) — w1=gate, w3=value.
        # ------------------------------------------------------------------ #
        ff_proj = state_dict[f"{layer_name}.ff_proj.weight"]  # [24576, 4096]
        w_value = ff_proj[: self.intermediate_size]  # [12288, 4096]
        w_gate = ff_proj[self.intermediate_size :]  # [12288, 4096]

        # Use bfloat8_b for MLP weights to halve memory footprint.
        # Replicated bf16 MLP weights (3 × 12288 × 4096 × 36 layers) = ~10.9 GB/device,
        # exceeding the 12 GB device DRAM when combined with other weights.
        # bfloat8_b cuts this to ~5.4 GB/device; total fits comfortably.
        mlp_dtype = ttnn.bfloat8_b

        def _tt(weight, name):
            return ttnn.as_tensor(
                weight.T.unsqueeze(0).unsqueeze(0),
                dtype=mlp_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                cache_file_name=cache_name(name),
            )

        self.w1 = _tt(w_gate, "w1_gate_replicated")  # gate → silu
        self.w3 = _tt(w_value, "w3_value_replicated")  # value/up

        ff_out = state_dict[f"{layer_name}.ff_out.weight"]  # [4096, 12288]
        self.w2 = _tt(ff_out, "w2_down_replicated")  # down projection

    def forward(self, x: ttnn.Tensor, mode: Mode) -> ttnn.Tensor:
        """Standard SwiGLU: silu(w1(x)) * w3(x) → w2(result).

        With the swapped assignment (w1=gate, w3=value), this equals
        silu(gate) * value — which is the Molmo2 formula.
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

        # silu(gate) * value
        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
            memory_config=w1_out.memory_config(),
        )
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)

        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(w2_in)

        # Restore [1, 1, S, dim] shape only if we chunked
        if chunked:
            original_shape = w2_out.shape
            w2_out = ttnn.reshape(
                w2_out,
                (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1]),
            )
        return w2_out
