# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule


class TtRoutedExpert(LightweightModule):
    """TTNN implementation of a routed expert with SwiGLU-style gated activation.

    This implements the computation: output = silu(x @ gate_proj) * (x @ up_proj) @ down_proj
    Similar to DeepSeek's routed expert architecture.

    Args:
        device: TTNN device
        emb_dim: Input embedding dimension (default: 7*1024)
        hidden_dim: Hidden dimension for intermediate layer (default: 2*1024)
        torch_module: Optional PyTorch module to initialize weights from
        weights_dtype: Data type for weights (default: bfloat16)
        activations_dtype: Data type for activations (default: bfloat16)
        compute_kernel_config: Optional compute kernel config (default: None, uses default config)
    """

    def __init__(
        self,
        device,
        emb_dim: int = 7 * 1024,
        hidden_dim: int = 2 * 1024,
        torch_module=None,
        weights_dtype=ttnn.bfloat16,
        activations_dtype=ttnn.bfloat16,
        compute_kernel_config=None,
    ):
        super().__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.activations_dtype = activations_dtype
        self.compute_kernel_config = compute_kernel_config

        # Initialize weights from PyTorch module if provided, otherwise use random
        if torch_module is not None:
            # Get weights from PyTorch module (float32) and convert to specified dtype
            gate_proj_torch = torch_module.gate_proj.weight.T.unsqueeze(0).unsqueeze(0)  # (1, 1, emb_dim, hidden_dim)
            up_proj_torch = torch_module.up_proj.weight.T.unsqueeze(0).unsqueeze(0)  # (1, 1, emb_dim, hidden_dim)
            down_proj_torch = torch_module.down_proj.weight.T.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_dim, emb_dim)
        else:
            # Random initialization
            gate_proj_torch = torch.randn(1, 1, emb_dim, hidden_dim, dtype=torch.float32)
            up_proj_torch = torch.randn(1, 1, emb_dim, hidden_dim, dtype=torch.float32)
            down_proj_torch = torch.randn(1, 1, hidden_dim, emb_dim, dtype=torch.float32)

        # Three weight matrices: gate_proj, up_proj, down_proj
        # Convert to TTNN tensors with specified dtype
        self.gate_proj = ttnn.from_torch(
            gate_proj_torch,
            dtype=weights_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        self.up_proj = ttnn.from_torch(
            up_proj_torch,
            dtype=weights_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        self.down_proj = ttnn.from_torch(
            down_proj_torch,
            dtype=weights_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass: output = silu(x @ gate_proj) * (x @ up_proj) @ down_proj

        Args:
            x: Input tensor of shape (..., emb_dim)

        Returns:
            Output tensor of shape (..., emb_dim)
        """
        # Verify input shape
        assert x.shape[-1] == self.emb_dim, f"Input last dim {x.shape[-1]} != emb_dim {self.emb_dim}"

        # Convert input to activations dtype if needed
        if x.dtype != self.activations_dtype:
            x = ttnn.typecast(x, self.activations_dtype)

        # Gate and up projections (with compute config if provided)
        matmul_kwargs = {}
        if self.compute_kernel_config is not None:
            matmul_kwargs["compute_kernel_config"] = self.compute_kernel_config

        gate = ttnn.matmul(x, self.gate_proj, **matmul_kwargs)
        up = ttnn.matmul(x, self.up_proj, **matmul_kwargs)

        # Apply SiLU activation to gate and multiply with up
        activated = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        )

        # Down projection (with compute config if provided)
        output = ttnn.matmul(activated, self.down_proj, **matmul_kwargs)

        return output
