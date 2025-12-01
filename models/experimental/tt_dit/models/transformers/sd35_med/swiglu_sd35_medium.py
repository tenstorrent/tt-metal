# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
SD3.5 Medium SwiGLU FeedForward Implementation

This module implements SwiGLU (Swish-Gated Linear Unit) feedforward network
for SD3.5 Medium, matching the reference implementation.
"""

import ttnn
from models.experimental.tt_dit.layers.linear import Linear
from models.experimental.tt_dit.layers.module import Module


class SD35MediumSwiGLU(Module):
    """
    SwiGLU FeedForward as used in modern transformers.
    Implements: w2(silu(w1(x)) * w3(x))
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float = None,
        bias: bool = False,
        mesh_device=None,
    ):
        super().__init__()

        # Calculate hidden dimension with same logic as reference
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.mesh_device = mesh_device

        # Three linear layers matching reference
        self.w1 = Linear(
            dim,
            hidden_dim,
            bias=bias,
            mesh_device=mesh_device,
        )

        self.w2 = Linear(
            hidden_dim,
            dim,
            bias=bias,
            mesh_device=mesh_device,
        )

        self.w3 = Linear(
            dim,
            hidden_dim,
            bias=bias,
            mesh_device=mesh_device,
        )

        # Compute kernel config for better performance
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass: w2(silu(w1(x)) * w3(x))

        Args:
            x: Input tensor [1, B, seq_len, dim]

        Returns:
            Output tensor [1, B, seq_len, dim]
        """
        # w1 projection
        w1_out = self.w1(x, compute_kernel_config=self.compute_kernel_config)

        # Apply SiLU activation
        w1_silu = ttnn.silu(w1_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # w3 projection
        w3_out = self.w3(x, compute_kernel_config=self.compute_kernel_config)

        # Element-wise multiplication (gating)
        gated = ttnn.multiply(w1_silu, w3_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # w2 projection (output)
        output = self.w2(gated, compute_kernel_config=self.compute_kernel_config)

        return output
