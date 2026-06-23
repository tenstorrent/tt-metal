# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
MiniMax-M3-VL encoder MLP.

Linear(1280 -> 5120) -> GELU -> Linear(5120 -> 1280).

The HF activation is `gelu` (the accurate, erf-based GELU — config
`hidden_act="gelu"`, not the tanh approximation). We map it to
`ttnn.gelu(..., fast_and_approximate_mode=False)`: a per-op bisection on
the sibling MoonViT tower showed the *fast-approximate* device GELU is the
dominant source of full-tower bf16 error, and the accurate GELU recovers
far more PCC than any attention-kernel tweak. The MLP is not the pipeline
bottleneck (attention is), so the accuracy/speed trade favors accuracy.

Both dims (1280, 5120) are tile-aligned, so no auto-pad subtlety here.

Reference: `MiniMaxM3VLVisionMLP` — attributes:
    self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
    self.activation_fn = gelu
    self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)
"""
from __future__ import annotations

from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.minimax_m3_vl.tt.common import as_linear_bias, as_linear_weight, hifi4_compute_config


class M3VLMLP(LightweightModule):
    """Two-layer MLP with accurate GELU, matching MiniMaxM3VLVisionMLP from HF."""

    def __init__(
        self,
        mesh_device,
        hidden_size: int,
        intermediate_size: int,
        fc1_weight: torch.Tensor,
        fc1_bias: torch.Tensor,
        fc2_weight: torch.Tensor,
        fc2_bias: torch.Tensor,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = mesh_device
        self.dtype = dtype
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)

        # Shape sanity.
        assert tuple(fc1_weight.shape) == (intermediate_size, hidden_size), (
            f"fc1.weight shape {tuple(fc1_weight.shape)} != "
            f"(intermediate={intermediate_size}, hidden={hidden_size})"
        )
        assert tuple(fc2_weight.shape) == (hidden_size, intermediate_size), (
            f"fc2.weight shape {tuple(fc2_weight.shape)} != "
            f"(hidden={hidden_size}, intermediate={intermediate_size})"
        )
        assert fc1_bias.shape[0] == intermediate_size, f"fc1.bias shape {fc1_bias.shape}"
        assert fc2_bias.shape[0] == hidden_size, f"fc2.bias shape {fc2_bias.shape}"

        self.fc1_weight = as_linear_weight(mesh_device, fc1_weight, dtype)
        self.fc1_bias = as_linear_bias(mesh_device, fc1_bias, dtype)
        self.fc2_weight = as_linear_weight(mesh_device, fc2_weight, dtype)
        self.fc2_bias = as_linear_bias(mesh_device, fc2_bias, dtype)

        # HiFi4 + FP32 dst accum for the matmuls (small accuracy boost).
        self.compute_kernel_config = hifi4_compute_config()

    @classmethod
    def from_torch(
        cls,
        mesh_device,
        ref: torch.nn.Module,
        dtype=ttnn.bfloat16,
    ) -> "M3VLMLP":
        """Construct from a torch MiniMaxM3VLVisionMLP reference module.

        Pulls out fc1/fc2 weights+biases and the dimensions implied by their shapes.
        """
        assert hasattr(ref, "fc1") and hasattr(ref, "fc2"), (
            f"expected an MLP-like module with fc1+fc2, got {type(ref).__name__} "
            f"with children {[n for n, _ in ref.named_children()]}"
        )
        intermediate_size, hidden_size = ref.fc1.weight.shape
        return cls(
            mesh_device=mesh_device,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc1_weight=ref.fc1.weight.data,
            fc1_bias=ref.fc1.bias.data,
            fc2_weight=ref.fc2.weight.data,
            fc2_bias=ref.fc2.bias.data,
            dtype=dtype,
        )

    def forward(self, x: ttnn.Tensor, memory_config: Optional["ttnn.MemoryConfig"] = None) -> ttnn.Tensor:
        # fc1: linear up
        hidden = ttnn.linear(
            x,
            self.fc1_weight,
            bias=self.fc1_bias,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Accurate (erf) GELU.
        hidden = ttnn.gelu(hidden, fast_and_approximate_mode=False)
        # fc2: linear down
        out = ttnn.linear(
            hidden,
            self.fc2_weight,
            bias=self.fc2_bias,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(hidden)
        return out
