# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MLP (HF `EmbeddingGemma2MLP`): down( gelu_tanh(gate x) * up x ), no biases.

Contract
    input   x [B, 1, S, hidden] bf16 TILE, DRAM interleaved
    output  [B, 1, S, hidden]  bf16 TILE, DRAM interleaved
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.blackhole.embedding_gemma2.tt.config import MLPConfig, compute_kernel_config_hifi4
from models.demos.blackhole.embedding_gemma2.tt.weights import as_weight, linear_weight


class MLP(LightweightModule):
    def __init__(self, config: MLPConfig, w_gate: ttnn.Tensor, w_up: ttnn.Tensor, w_down: ttnn.Tensor):
        super().__init__()
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
    ) -> "MLP":
        """`state_dict` holds the HF `layers.<i>.mlp.` sub-state."""
        H, I = config.hidden_size, config.intermediate_size
        shapes = {"gate_proj": (I, H), "up_proj": (I, H), "down_proj": (H, I)}
        ws = {}
        for k, shp in shapes.items():
            w = state_dict[f"{k}.weight"]
            if tuple(w.shape) != shp:
                raise ValueError(f"{name}.{k}: expected {shp}, got {tuple(w.shape)}")
            ws[k] = as_weight(linear_weight(w), device, cache_dir=cache_dir, name=f"{name}.{k}.weight")
        return cls(config, ws["gate_proj"], ws["up_proj"], ws["down_proj"])

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
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
