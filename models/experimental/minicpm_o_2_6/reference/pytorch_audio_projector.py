# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch reference implementation of Audio Projector for MiniCPM-o-2_6.

Matches the official MultiModalProjector architecture with AvgPool1d and 2-layer MLP.
"""

import torch
import torch.nn as nn
from loguru import logger


class MultiModalProjector(nn.Module):
    """
    Multi-modal projector matching official MiniCPM-o-2_6 architecture.

    Architecture (from modeling_minicpmo.py line 2577-2587):
    - Linear1: in_dim → out_dim with bias and ReLU
    - Linear2: out_dim → out_dim with bias

    Note: Pooling is applied externally in the forward pipeline.
    """

    def __init__(self, in_dim: int, out_dim: int):
        """
        Initialize MultiModalProjector matching official architecture.

        Args:
            in_dim: Input dimension
            out_dim: Output dimension
        """
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_dim, out_features=out_dim, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=out_dim, out_features=out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Projected tensor
        """
        hidden_states = self.relu(self.linear1(x))
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class PyTorchAudioProjector(nn.Module):
    """
    PyTorch reference implementation of audio projector matching official MiniCPM-o-2_6 architecture.

    Official Architecture (from modeling_minicpmo.py):
    - audio_output_dim = int(apm.config.encoder_ffn_dim // 4)  # 4096 // 4 = 1024
    - audio_avg_pooler = nn.AvgPool1d(audio_pool_step, stride=audio_pool_step)  # pool_step=2
    - audio_projection_layer = MultiModalProjector(in_dim=1024, out_dim=3584)

    Official Forward Flow:
    1. audio_embeds = audio_projection_layer(audio_states)  # [B, T, 1024] → [B, T, 3584]
    2. audio_embeds = audio_embeds.transpose(1, 2)  # [B, T, 3584] → [B, 3584, T]
    3. audio_embeds = audio_avg_pooler(audio_embeds)  # [B, 3584, T] → [B, 3584, T/2]
    4. audio_embeds = audio_embeds.transpose(1, 2)  # [B, 3584, T/2] → [B, T/2, 3584]

    Input: [batch, seq_len, 1024] (Whisper encoder output)
    Output: [batch, seq_len//pool_step, 3584] (Qwen2.5 embedding space)
    """

    def __init__(
        self,
        input_dim: int = 1024,  # Whisper hidden size
        output_dim: int = 3584,  # Qwen2.5 hidden size
        pool_step: int = 2,  # Pooling step from config.audio_pool_step
    ):
        """
        Initialize PyTorch Audio Projector matching official MiniCPM-o-2_6 architecture.

        Args:
            input_dim: Input dimension (Whisper: 1024)
            output_dim: Output dimension (Qwen2.5: 3584)
            pool_step: Pooling step (default 2)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pool_step = pool_step

        # Audio pooling (from modeling_minicpmo.py line 113)
        self.audio_avg_pooler = nn.AvgPool1d(kernel_size=self.pool_step, stride=self.pool_step, padding=0)

        # Audio projection layer (from modeling_minicpmo.py line 114)
        # in_dim matches Whisper encoder output dimension (1024)
        self.audio_projection_layer = MultiModalProjector(in_dim=input_dim, out_dim=output_dim)  # 1024  # 3584

        logger.info(
            f"✅ PyTorchAudioProjector initialized: {input_dim}d → {output_dim}d → {output_dim}d (pool step={pool_step})"
        )

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of audio projector matching official MiniCPM-o-2_6 architecture.

        Args:
            audio_features: [batch, seq_len, 1024] Whisper encoder output

        Returns:
            projected_features: [batch, seq_len//pool_step, 3584] Qwen2.5 embedding space
        """
        batch, seq_len, hidden_dim = audio_features.shape

        # Ensure input has correct hidden dimension
        assert hidden_dim == self.input_dim, f"Input hidden dim {hidden_dim} != expected {self.input_dim}"

        # Step 1: Apply 2-layer MLP projection (1024 → 3584 → 3584)
        # This matches self.audio_projection_layer(audio_states) in the official code
        audio_embeds = self.audio_projection_layer(audio_features)  # [B, T, 3584]

        # Step 2: Transpose for pooling [B, T, 3584] → [B, 3584, T]
        audio_embeds = audio_embeds.transpose(1, 2)  # [B, T, 3584] → [B, 3584, T]

        # Step 3: Apply AvgPool1d [B, 3584, T] → [B, 3584, T/pool_step]
        audio_embeds = self.audio_avg_pooler(audio_embeds)

        # Step 4: Transpose back [B, 3584, T/pool_step] → [B, T/pool_step, 3584]
        audio_embeds = audio_embeds.transpose(1, 2)

        return audio_embeds
