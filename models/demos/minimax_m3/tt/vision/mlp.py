# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
MiniMax-M3 vision MLP, as Hugging Face's MiniMaxM3VLVisionMLP computes it.

It runs fc1, the exact (erf) GELU, then fc2, both linears with bias. Each encoder layer has one, 1280 -> 5120 -> 1280,
and the projector two more of other widths. With M3VLMLP.fuse_gelu on, fc1's matmul applies the GELU where fc1 gets a
program config.
"""
from __future__ import annotations

from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.minimax_m3.tt.vision.common import (
    DRAM_PLACEMENT,
    BlockPlacement,
    as_linear_bias,
    as_linear_weight,
    linear,
    linear_compute_config,
    memory_config_kwargs,
)


def accurate_gelu(x: ttnn.Tensor, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor:
    """The accurate (erf) GELU of HF's `gelu`, into memory_config, or, when None, where x is (ttnn.gelu gets no
    memory_config)."""
    return ttnn.gelu(x, fast_and_approximate_mode=False, **memory_config_kwargs(memory_config))


# The same GELU as a matmul's fused_activation: ttnn.gelu with fast_and_approximate_mode=False runs this unary.
ACCURATE_GELU_FUSED = ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 0.0)


class M3VLMLP(LightweightModule):
    """Two-layer MLP with the exact GELU, like Hugging Face's MiniMaxM3VLVisionMLP, sized by its weights."""

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
        """Construct from a torch MLP with fc1 and fc2, such as Hugging Face's MiniMaxM3VLVisionMLP."""
        return cls.from_linears(mesh_device, ref.fc1, ref.fc2, dtype=dtype)

    def forward(self, x: ttnn.Tensor, placement: BlockPlacement = DRAM_PLACEMENT) -> ttnn.Tensor:
        """fc2(gelu(fc1(x))), with fc1's, GELU's and fc2's outputs where placement's fc1_out, gelu_out and residual
        put them. The projector's MLPs take the default, every output in DRAM."""
        hidden = linear(
            x,
            self.fc1_weight,
            self.fc1_bias,
            self.compute_kernel_config,
            activation=accurate_gelu,
            fused_activation=ACCURATE_GELU_FUSED if self.fuse_gelu else None,
            memory_config=placement.fc1_out,
            activation_memory_config=placement.gelu_out,
        )
        out = linear(
            hidden, self.fc2_weight, self.fc2_bias, self.compute_kernel_config, memory_config=placement.residual
        )
        ttnn.deallocate(hidden)
        return out
