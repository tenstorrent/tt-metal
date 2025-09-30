# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Language Model Head - final output projection for language models"""

from dataclasses import dataclass
from typing import Optional

import torch

import ttnn


@dataclass
class LMHeadConfig:
    """Configuration for LM Head module"""

    hidden_size: int
    vocab_size: int
    bias: bool = False
    tie_word_embeddings: bool = False


class LMHead(torch.nn.Module):
    """
    Language Model Head for next token prediction.

    Projects hidden states to vocabulary logits.
    Supports weight tying with input embeddings.
    """

    def __init__(
        self,
        config: LMHeadConfig,
        device: ttnn.Device,
    ):
        super().__init__()
        self.config = config
        self.device = device

        # Output projection weight
        self.weight = None
        self.bias = None if not config.bias else None

    def setup_weight(self, weight: ttnn.Tensor, bias: Optional[ttnn.Tensor] = None):
        """
        Set pre-loaded weight for the LM head.

        Args:
            weight: Output projection weight
            bias: Optional bias term
        """
        self.weight = weight
        if self.config.bias and bias is not None:
            self.bias = bias

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """
        Project hidden states to vocabulary logits.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            logits: Output logits of shape [batch_size, seq_len, vocab_size]
        """
        # Linear projection to vocabulary
        logits = ttnn.matmul(hidden_states, self.weight)

        # Add bias if present
        if self.bias is not None:
            logits = logits + self.bias

        return logits


class DistributedLMHead(torch.nn.Module):
    """
    Distributed Language Model Head for large vocabulary sizes.

    Splits vocabulary across devices for memory efficiency.
    """

    def __init__(
        self,
        config: LMHeadConfig,
        device: ttnn.Device,
        num_devices: int = 1,
        device_idx: int = 0,
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.num_devices = num_devices
        self.device_idx = device_idx

        # Calculate local vocabulary size
        vocab_per_device = config.vocab_size // num_devices
        remainder = config.vocab_size % num_devices

        # Distribute remainder tokens to first devices
        if device_idx < remainder:
            self.local_vocab_size = vocab_per_device + 1
            self.vocab_start_idx = device_idx * (vocab_per_device + 1)
        else:
            self.local_vocab_size = vocab_per_device
            self.vocab_start_idx = remainder * (vocab_per_device + 1) + (device_idx - remainder) * vocab_per_device

        self.vocab_end_idx = self.vocab_start_idx + self.local_vocab_size

        # Local weight shard
        self.weight = None
        self.bias = None if not config.bias else None

    def setup_weight(self, weight: ttnn.Tensor, bias: Optional[ttnn.Tensor] = None):
        """
        Set pre-loaded weight shard for this device.

        Args:
            weight: Local weight shard for vocabulary [hidden_size, local_vocab_size]
            bias: Optional local bias shard
        """
        self.weight = weight
        if self.config.bias and bias is not None:
            self.bias = bias

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """
        Compute local vocabulary logits.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            local_logits: Local logits of shape [batch_size, seq_len, local_vocab_size]
        """
        # Compute local logits
        local_logits = ttnn.matmul(hidden_states, self.weight)

        # Add bias if present
        if self.bias is not None:
            local_logits = local_logits + self.bias

        return local_logits

    def gather_logits(self, local_logits_list: list[ttnn.Tensor]) -> ttnn.Tensor:
        """
        Gather logits from all devices to form complete vocabulary.

        Args:
            local_logits_list: List of local logits from each device

        Returns:
            logits: Complete logits of shape [batch_size, seq_len, vocab_size]
        """
        # Concatenate along vocabulary dimension
        full_logits = ttnn.concat(local_logits_list, dim=-1)
        return full_logits


class ClassificationHead(torch.nn.Module):
    """
    Classification head for sequence classification tasks.

    Projects hidden states to class logits with optional pooling.
    """

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        pooling_type: str = "last",  # Options: "last", "first", "mean", "max"
        dropout: float = 0.0,
        device: ttnn.Device = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.pooling_type = pooling_type
        self.dropout = dropout
        self.device = device

        # Classification weight and bias
        self.weight = None
        self.bias = None

    def setup_weights(self, weight: ttnn.Tensor, bias: ttnn.Tensor):
        """Set pre-loaded weights for classification head"""
        self.weight = weight
        self.bias = bias

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Compute classification logits.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional mask for pooling

        Returns:
            logits: Classification logits of shape [batch_size, num_classes]
        """
        # Apply pooling
        pooled_output = self._pool_hidden_states(hidden_states, attention_mask)

        # Apply dropout if configured
        if self.dropout > 0 and self.training:
            pooled_output = ttnn.dropout(pooled_output, p=self.dropout)

        # Classification projection
        logits = ttnn.matmul(pooled_output, self.weight)
        if self.bias is not None:
            logits = logits + self.bias

        return logits

    def _pool_hidden_states(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor],
    ) -> ttnn.Tensor:
        """Pool sequence hidden states based on pooling type"""
        if self.pooling_type == "last":
            # Use last valid token
            if attention_mask is not None:
                # Find last valid position for each sequence
                seq_lengths = ttnn.sum(attention_mask, dim=1) - 1
                batch_size = hidden_states.shape[0]
                pooled = []
                for i in range(batch_size):
                    last_idx = int(seq_lengths[i].item())
                    pooled.append(hidden_states[i, last_idx, :])
                return ttnn.stack(pooled, dim=0)
            else:
                # Use last position
                return hidden_states[:, -1, :]

        elif self.pooling_type == "first":
            # Use first token (e.g., [CLS])
            return hidden_states[:, 0, :]

        elif self.pooling_type == "mean":
            # Mean pooling over valid tokens
            if attention_mask is not None:
                # Expand mask for hidden size
                mask_expanded = ttnn.unsqueeze(attention_mask, dim=-1)
                mask_expanded = ttnn.broadcast_to(mask_expanded, hidden_states.shape)
                # Sum only valid tokens
                sum_hidden = ttnn.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = ttnn.sum(mask_expanded, dim=1)
                return sum_hidden / sum_mask
            else:
                return ttnn.mean(hidden_states, dim=1)

        elif self.pooling_type == "max":
            # Max pooling over valid tokens
            if attention_mask is not None:
                # Mask invalid tokens with very negative values
                mask_expanded = ttnn.unsqueeze(attention_mask, dim=-1)
                mask_expanded = ttnn.broadcast_to(mask_expanded, hidden_states.shape)
                masked_hidden = ttnn.where(mask_expanded, hidden_states, ttnn.full_like(hidden_states, -1e9))
                return ttnn.max(masked_hidden, dim=1)
            else:
                return ttnn.max(hidden_states, dim=1)

        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
