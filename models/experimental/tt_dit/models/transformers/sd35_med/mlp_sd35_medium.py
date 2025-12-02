# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SD3.5 Medium MLP Implementation

This module implements the MLP (Multi-Layer Perceptron) for SD3.5 Medium,
matching the reference MM-DiT implementation.
"""

import ttnn
from models.experimental.tt_dit.layers.linear import Linear
from models.experimental.tt_dit.layers.module import Module


class SD35MediumMlp(Module):
    """
    MLP as used in SD3.5 Medium MM-DiT.
    Two-layer feedforward network with GELU activation.
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        bias=True,
        mesh_device=None,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.mesh_device = mesh_device

        # First linear layer (fc1)
        self.fc1 = Linear(
            in_features,
            hidden_features,
            bias=bias,
            mesh_device=mesh_device,
            activation_fn="gelu",
        )

        # Second linear layer (fc2)
        self.fc2 = Linear(
            hidden_features,
            out_features,
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
        Forward pass of MLP.
        Args:
            x: Input tensor [1, B, seq_len, in_features]
        Returns:
            Output tensor [1, B, seq_len, out_features]
        """

        # fc1 with GELU activation
        x = self.fc1(x, compute_kernel_config=self.compute_kernel_config)

        # fc2
        x = self.fc2(x, compute_kernel_config=self.compute_kernel_config)

        return x
