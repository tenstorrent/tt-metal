# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Timestep embedding module for diffusion process
Converts scalar timestep to embedding vector using sinusoidal encoding
"""

import ttnn
import torch
import math


class TtTimestepEmbedding:
    """
    Timestep embedding module for diffusion process
    Uses sinusoidal positional encoding + MLP projection
    """

    def __init__(self, embed_dim: int, device: ttnn.Device):
        self.device = device
        self.embed_dim = embed_dim

        # Linear projection layers will be loaded from checkpoint
        self.linear1_weight = None
        self.linear1_bias = None
        self.linear2_weight = None
        self.linear2_bias = None

    def get_sinusoidal_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal positional embeddings for timesteps
        Standard implementation from diffusion models
        """
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.embed_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

        return emb

    def __call__(self, timesteps: torch.Tensor) -> ttnn.Tensor:
        """
        Forward pass: timestep -> sinusoidal embedding -> MLP

        Args:
            timesteps: Tensor of shape (batch_size,) containing timestep indices

        Returns:
            Timestep embeddings of shape (batch_size, embed_dim)
        """
        # Get sinusoidal embeddings on CPU/GPU
        emb = self.get_sinusoidal_embedding(timesteps)

        # Convert to TTNN tensor
        emb_ttnn = ttnn.from_torch(
            emb,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG
        )

        # Linear projection 1 (if weights loaded)
        if self.linear1_weight is not None:
            emb_ttnn = ttnn.linear(
                emb_ttnn,
                self.linear1_weight,
                bias=self.linear1_bias,
                memory_config=ttnn.L1_MEMORY_CONFIG
            )

            # SiLU activation
            emb_ttnn = ttnn.silu(emb_ttnn, memory_config=ttnn.L1_MEMORY_CONFIG)

            # Linear projection 2
            emb_ttnn = ttnn.linear(
                emb_ttnn,
                self.linear2_weight,
                bias=self.linear2_bias,
                memory_config=ttnn.L1_MEMORY_CONFIG
            )

        return emb_ttnn

    def load_weights(self, state_dict: dict, prefix: str = "time_embed"):
        """
        Load weights from PyTorch state dict
        """
        # Load linear layer weights
        linear1_weight = state_dict.get(f"{prefix}.0.weight")
        linear1_bias = state_dict.get(f"{prefix}.0.bias")
        linear2_weight = state_dict.get(f"{prefix}.2.weight")
        linear2_bias = state_dict.get(f"{prefix}.2.bias")

        if linear1_weight is None:
            return  # Weights not available

        # Convert to TTNN tensors
        self.linear1_weight = ttnn.from_torch(
            linear1_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        self.linear1_bias = ttnn.from_torch(
            linear1_bias,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        self.linear2_weight = ttnn.from_torch(
            linear2_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        self.linear2_bias = ttnn.from_torch(
            linear2_bias,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
