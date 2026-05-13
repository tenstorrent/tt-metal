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

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        gate = ttnn.linear(
            x,
            self.gate_proj,
            activation="silu",
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        up = ttnn.linear(
            x,
            self.up_proj,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden = ttnn.multiply(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = ttnn.linear(
            hidden,
            self.down_proj,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(hidden)
        return out
