# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
MiniMax-M3-VL encoder MLP.

Linear(1280 -> 5120) -> GELU -> Linear(5120 -> 1280) in each encoder layer; the projector runs two more MLPs of the
same form with other widths (projector.py).

The HF activation is `gelu` (the accurate, erf-based GELU: config
`hidden_act="gelu"`, not the tanh approximation). We map it to
`ttnn.gelu(..., fast_and_approximate_mode=False)` (accurate_gelu), or, with
M3VLMLP.fuse_gelu and where fc1 gets a program config, to the same GELU
inside fc1's matmul (ACCURATE_GELU_FUSED).

Reference: `MiniMaxM3VLVisionMLP`, with the attributes:
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


# The same GELU as a matmul's fused_activation: ttnn.gelu with fast_and_approximate_mode=False runs this unary.
ACCURATE_GELU_FUSED = ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 0.0)


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
        # On: where fc1 gets a program config, its matmul applies the GELU in the fp32 dest register before rounding
        # to bf16; elsewhere, and off, accurate_gelu runs on fc1's bf16 output. Off as built.
        self.fuse_gelu = False

    @classmethod
    def from_linears(
        cls,
        mesh_device,
        fc1: torch.nn.Linear,
        fc2: torch.nn.Linear,
        dtype=ttnn.bfloat16,
    ) -> "M3VLMLP":
        """Construct from two torch Linear layers with bias: fc1, then fc2 on its GELU."""
        return cls(
            mesh_device=mesh_device,
            fc1_weight=fc1.weight.data,
            fc1_bias=fc1.bias.data,
            fc2_weight=fc2.weight.data,
            fc2_bias=fc2.bias.data,
            dtype=dtype,
        )

    @classmethod
    def from_torch(
        cls,
        mesh_device,
        ref: torch.nn.Module,
        dtype=ttnn.bfloat16,
    ) -> "M3VLMLP":
        """Construct from a torch MLP with fc1 and fc2: VisionMLP of the Torch reference (reference/vision.py)."""
        return cls.from_linears(mesh_device, ref.fc1, ref.fc2, dtype=dtype)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        hidden = linear(
            x,
            self.fc1_weight,
            self.fc1_bias,
            self.compute_kernel_config,
            activation=accurate_gelu,
            fused_activation=ACCURATE_GELU_FUSED if self.fuse_gelu else None,
        )
        out = linear(hidden, self.fc2_weight, self.fc2_bias, self.compute_kernel_config)
        ttnn.deallocate(hidden)
        return out
