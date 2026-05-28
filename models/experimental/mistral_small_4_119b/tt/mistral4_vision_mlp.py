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

        # Presets depend on the actual sequence length (num_patches), which
        # isn't known until the first forward. Build them lazily and cache.
        self.ffn_up_preset = None
        self.ffn_down_preset = None
        self._preset_m: int | None = None

    def _ensure_presets(self, m: int) -> None:
        if self._preset_m == m:
            return
        self.ffn_up_preset = build_ffn_up_preset(self.mesh_device, m)
        self.ffn_down_preset = build_ffn_down_preset(self.mesh_device, m)
        self._preset_m = m

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        self._ensure_presets(int(x.shape[-2]))
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
        # ``hidden`` lives on the ffn_up output grid (8×6, per_core_N=4 → 32
        # cores). ffn_down resharards to its own ws grid (8×2=16 cores) — the
        # two shards overlap on cores [0..7, 0..1] and together with the
        # matmul CBs they exceed L1. ``deallocate_input=True`` frees the old
        # shard between the reshard and the matmul launch so only the new
        # shard + CBs need to fit on those cores.
        out = vision_linear(
            hidden,
            self.down_proj,
            self.ffn_down_preset,
            compute_kernel_config=self.compute_kernel_config,
            output_memory_config=ttnn.L1_MEMORY_CONFIG,
            deallocate_input=True,
        )
        return out
