# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MLP (HF `EmbeddingGemma2MLP`): down( gelu_tanh(gate x) * up x ), no biases.

Two paths share this class:
    optimized=False  (vision tower, E2 contract) bf16 weights, HiFi4, default program configs,
                     separate gelu op
    optimized=True   (text tower, E5a) bfp8 weights, `tt.linear.linear` program configs, gelu_tanh
                     fused into the gate matmul's packer

Contract
    input   x [B, 1, S, hidden] bf16 TILE, DRAM interleaved
    output  [B, 1, S, hidden]  bf16 TILE, DRAM interleaved
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.blackhole.embedding_gemma2.tt.config import MLPConfig, act_memory_config, compute_kernel_config_hifi4
from models.demos.blackhole.embedding_gemma2.tt.linear import WEIGHT_DTYPE, gated_product, linear
from models.demos.blackhole.embedding_gemma2.tt.weights import as_weight, linear_weight


class MLP(LightweightModule):
    def __init__(
        self, config: MLPConfig, w_gate: ttnn.Tensor, w_up: ttnn.Tensor, w_down: ttnn.Tensor, optimized: bool = False
    ):
        super().__init__()
        self.optimized = optimized
        self.config = config
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down
        self.compute_kernel_config = compute_kernel_config_hifi4()

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        config: MLPConfig,
        device,
        *,
        cache_dir: Path | None = None,
        name: str = "mlp",
        optimized: bool = False,
    ) -> "MLP":
        """`state_dict` holds the HF `layers.<i>.mlp.` sub-state."""
        H, I = config.hidden_size, config.intermediate_size
        shapes = {"gate_proj": (I, H), "up_proj": (I, H), "down_proj": (H, I)}
        ws = {}
        for k, shp in shapes.items():
            w = state_dict[f"{k}.weight"]
            if tuple(w.shape) != shp:
                raise ValueError(f"{name}.{k}: expected {shp}, got {tuple(w.shape)}")
            ws[k] = as_weight(
                linear_weight(w),
                device,
                cache_dir=cache_dir,
                name=f"{name}.{k}.weight",
                dtype=WEIGHT_DTYPE if optimized else ttnn.bfloat16,
            )
        return cls(config, ws["gate_proj"], ws["up_proj"], ws["down_proj"], optimized=optimized)

    def forward(self, x: ttnn.Tensor, memory_config=None) -> ttnn.Tensor:
        """`memory_config` (optimized path only): output placement; DRAM when None."""
        if self.optimized:
            return self._forward_optimized(x, memory_config or ttnn.DRAM_MEMORY_CONFIG)
        ckc = self.compute_kernel_config
        dram = ttnn.DRAM_MEMORY_CONFIG
        gate = ttnn.linear(x, self.w_gate, compute_kernel_config=ckc, memory_config=dram)
        up = ttnn.linear(x, self.w_up, compute_kernel_config=ckc, memory_config=dram)
        gate = ttnn.gelu(gate, variant=ttnn.GeluVariant.Tanh)
        h = ttnn.multiply(gate, up, memory_config=dram)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = ttnn.linear(h, self.w_down, compute_kernel_config=ckc, memory_config=dram)
        ttnn.deallocate(h)
        return out

    def _forward_optimized(self, x: ttnn.Tensor, out_memory_config) -> ttnn.Tensor:
        B, S = x.shape[0], x.shape[2]
        mem = act_memory_config(x.device(), B, S, self.config.intermediate_size)
        up = linear(x, self.w_up, memory_config=mem)
        h = gated_product(x, self.w_gate, up, memory_config=mem)
        ttnn.deallocate(up)
        out = linear(h, self.w_down, memory_config=out_memory_config)
        ttnn.deallocate(h)
        return out
