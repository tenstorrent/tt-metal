# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Pixtral vision MLP (SiLU-gated, no bias) — fully on device, no host fallback.

    out = down_proj( silu(gate_proj(x)) * up_proj(x) )

All weights are replicated across the mesh. Total per layer at bf16:
    3 × (1024 × 4096) × 2 B  ≈  25 MB
"""

from __future__ import annotations


import ttnn
from models.experimental.mistral_small_4_119b.tt.mistral4_self_attention import _load_weight
from models.experimental.mistral_small_4_119b.tt.vision_matmul_config import (
    build_ffn_down_preset,
    build_ffn_up_preset,
    vision_linear,
)


class TtPixtralMLP:
    """Gated FFN: gate_proj/up_proj 1024→4096, down_proj 4096→1024."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        state_dict: dict,
        layer_prefix: str,
        compute_kernel_config,
        dtype=ttnn.bfloat16,
    ):
        self.mesh_device = mesh_device
        self.compute_kernel_config = compute_kernel_config

        ffn_prefix = layer_prefix + "feed_forward."
        # HF stores [out, in]; transpose to [in, out] for ttnn.linear.
        self.gate_proj = _load_weight(
            state_dict,
            ffn_prefix + "gate_proj.weight",
            transpose=True,
            dtype=dtype,
            mesh_device=mesh_device,
        )  # [1024, 4096]
        self.up_proj = _load_weight(
            state_dict,
            ffn_prefix + "up_proj.weight",
            transpose=True,
            dtype=dtype,
            mesh_device=mesh_device,
        )  # [1024, 4096]
        self.down_proj = _load_weight(
            state_dict,
            ffn_prefix + "down_proj.weight",
            transpose=True,
            dtype=dtype,
            mesh_device=mesh_device,
        )  # [4096, 1024]

        self.ffn_up_preset = build_ffn_up_preset(mesh_device)
        self.ffn_down_preset = build_ffn_down_preset(mesh_device)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # ffn_norm output is WS L1; gate and up both need it in L1 interleaved
        # (their preset's in0). Convert once here so vision_linear's per-call
        # to_memory_config becomes a no-op for both, saving one shard→intlv op.
        if x.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        # gate/up keep their WS output (skip per-matmul WS→DRAM convert); the
        # multiply runs WS×WS→WS and down's vision_linear reshards in0 to its
        # own (smaller) WS grid in one to_memory_config call.
        gate = vision_linear(
            x,
            self.gate_proj,
            self.ffn_up_preset,
            compute_kernel_config=self.compute_kernel_config,
            activation="silu",
            keep_sharded=True,
        )
        up = vision_linear(
            x,
            self.up_proj,
            self.ffn_up_preset,
            compute_kernel_config=self.compute_kernel_config,
            keep_sharded=True,
        )
        hidden = ttnn.multiply(gate, up, memory_config=gate.memory_config())
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = vision_linear(
            hidden,
            self.down_proj,
            self.ffn_down_preset,
            compute_kernel_config=self.compute_kernel_config,
            output_memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(hidden)
        return out
