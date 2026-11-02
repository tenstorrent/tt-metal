# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
MiniMax-M3-VL encoder MLP.

Linear(1280 -> 5120) -> GELU -> Linear(5120 -> 1280) in each encoder layer; the projector runs two more MLPs of the
same form with other widths (projector.py).

The HF activation is `gelu` (the accurate, erf-based GELU — config
`hidden_act="gelu"`, not the tanh approximation). We map it to
`ttnn.gelu(..., fast_and_approximate_mode=False)` (accurate_gelu).

Reference: `MiniMaxM3VLVisionMLP` — attributes:
    self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
    self.activation_fn = gelu
    self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)
"""
from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.minimax_m3.tt.vision.common import as_linear_bias, as_linear_weight, linear, linear_compute_config


def accurate_gelu(x: ttnn.Tensor) -> ttnn.Tensor:
    """The accurate (erf) GELU of HF's `gelu`."""
    return ttnn.gelu(x, fast_and_approximate_mode=False)


class M3VLMLP(LightweightModule):
    """Two-layer MLP with accurate GELU, matching MiniMaxM3VLVisionMLP from HF. The widths are the weights' own."""

    def __init__(
        self,
        mesh_device,
        fc1_weight: torch.Tensor,
        fc1_bias: torch.Tensor,
        fc2_weight: torch.Tensor,
        fc2_bias: torch.Tensor,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()

        # Shape sanity: fc1 [intermediate, in], fc2 [out, intermediate].
        assert fc2_weight.shape[1] == fc1_weight.shape[0], (
            f"fc2.weight shape {tuple(fc2_weight.shape)} does not take the "
            f"{fc1_weight.shape[0]} outputs of fc1.weight {tuple(fc1_weight.shape)}"
        )
        assert fc1_bias.shape[0] == fc1_weight.shape[0], f"fc1.bias shape {fc1_bias.shape}"
        assert fc2_bias.shape[0] == fc2_weight.shape[0], f"fc2.bias shape {fc2_bias.shape}"

        self.fc1_weight = as_linear_weight(mesh_device, fc1_weight, dtype)
        self.fc1_bias = as_linear_bias(mesh_device, fc1_bias, dtype)
        self.fc2_weight = as_linear_weight(mesh_device, fc2_weight, dtype)
        self.fc2_bias = as_linear_bias(mesh_device, fc2_bias, dtype)

        # HiFi4 + FP32 dst accum for the matmuls (small accuracy boost), partial sums accumulated in L1.
        self.compute_kernel_config = linear_compute_config()

    @classmethod
    def from_torch(
        cls,
        mesh_device,
        ref: torch.nn.Module,
        dtype=ttnn.bfloat16,
    ) -> "M3VLMLP":
        """Construct from a torch MLP with fc1 and fc2: VisionMLP of the Torch reference (reference/vision.py)."""
        assert hasattr(ref, "fc1") and hasattr(ref, "fc2"), (
            f"expected an MLP-like module with fc1+fc2, got {type(ref).__name__} "
            f"with children {[n for n, _ in ref.named_children()]}"
        )
        return cls(
            mesh_device=mesh_device,
            fc1_weight=ref.fc1.weight.data,
            fc1_bias=ref.fc1.bias.data,
            fc2_weight=ref.fc2.weight.data,
            fc2_bias=ref.fc2.bias.data,
            dtype=dtype,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        hidden = linear(x, self.fc1_weight, self.fc1_bias, self.compute_kernel_config, activation=accurate_gelu)
        out = linear(hidden, self.fc2_weight, self.fc2_bias, self.compute_kernel_config)
        ttnn.deallocate(hidden)
        return out
