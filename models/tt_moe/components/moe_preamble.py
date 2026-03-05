# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
MoE Preamble Module - Handles intermediate processing between router and experts.

This module encapsulates the backend-specific transformations needed after routing
but before expert processing, including weight transformations, reshaping, and
preparation for all_to_all operations.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import ttnn


@dataclass
class MoEPreambleConfig:
    """Configuration for MoE preamble operations."""

    backend: str  # "deepseek" or "gptoss"
    hidden_size: int
    num_experts_per_tok: int
    num_experts_per_device: int
    num_devices: int
    num_dispatch_devices: int
    batch_size_per_device: int
    seq_len: int = 1  # 1 for decode, > 1 for prefill

    # DeepSeek specific
    topk_weights_repeat: Optional[Dict] = None

    # GPT-OSS specific
    use_throughput_experts: bool = True


class MoEPreamble:
    """
    Handles backend-specific pre-processing between router and experts.

    This includes:
    - Weight transformations (repeat/permute for DeepSeek)
    - Input reshaping for all_to_all operations
    - Format conversions for backend-specific requirements
    """

    @classmethod
    def forward(
        cls,
        x: ttnn.Tensor,
        topk_weights: ttnn.Tensor,
        topk_indices: ttnn.Tensor,
        config: MoEPreambleConfig,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """
        Apply backend-specific preamble transformations.

        Args:
            x: Input hidden states
            topk_weights: Router output weights
            topk_indices: Router output expert indices
            config: Preamble configuration

        Returns:
            Tuple of (processed_x, processed_weights, processed_indices)
        """
        if config.backend == "deepseek":
            return cls._forward_deepseek(x, topk_weights, topk_indices, config)
        elif config.backend == "gptoss":
            return cls._forward_gptoss(x, topk_weights, topk_indices, config)
        else:
            raise ValueError(f"Unknown backend: {config.backend}")

    @classmethod
    def _forward_deepseek(
        cls,
        x: ttnn.Tensor,
        topk_weights: ttnn.Tensor,
        topk_indices: ttnn.Tensor,
        config: MoEPreambleConfig,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """
        DeepSeek preamble: repeat/permute weights and reshape inputs.
        """
        # Transform weights with repeat and permute
        topk_weights_rm = ttnn.to_layout(topk_weights, ttnn.ROW_MAJOR_LAYOUT)

        if config.topk_weights_repeat:
            topk_weights_rm = ttnn.repeat(topk_weights_rm, **config.topk_weights_repeat)

        # Permute to match expected format for all_to_all
        topk_weights_rm = ttnn.permute(topk_weights_rm, (3, 1, 2, 0))
        topk_weights = ttnn.to_layout(topk_weights_rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(topk_weights_rm)

        # Reshape inputs for all_to_all_dispatch
        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x_reshaped = ttnn.reshape(
            x_rm,
            shape=(config.batch_size_per_device, 1, config.seq_len, config.hidden_size),
        )

        topk_indices_rm = ttnn.to_layout(topk_indices, ttnn.ROW_MAJOR_LAYOUT)
        topk_indices_reshaped = ttnn.reshape(
            topk_indices_rm, shape=(config.batch_size_per_device, 1, config.seq_len, config.num_experts_per_tok)
        )

        return x_reshaped, topk_weights, topk_indices_reshaped

    @classmethod
    def _forward_gptoss(
        cls,
        x: ttnn.Tensor,
        topk_weights: ttnn.Tensor,
        topk_indices: ttnn.Tensor,
        config: MoEPreambleConfig,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """
        GPT-OSS preamble: minimal transformations for ThroughputExperts.
        """
        # GPT-OSS expects different input shapes for ThroughputExperts
        # The router already provides the correct format, but we may need
        # to ensure proper layout

        # Ensure tile layout for efficient operations
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # ThroughputExperts expects:
        # - hidden_states: [batch_per_device, 1, 1, hidden_size] for decode
        #                  [seq_per_device, 1, 1, hidden_size] for prefill
        # - topk_indices: [batch/seq, 1, 1, num_experts_per_tok]
        # - topk_weights: [batch/seq, 1, 1, num_experts_per_tok]

        # No weight transformation needed for GPT-OSS
        # The weights are used directly in the ThroughputExperts module

        return x, topk_weights, topk_indices

    @classmethod
    def create_config(
        cls,
        backend: str,
        hf_config,
        batch_size_per_device: int,
        seq_len: int = 1,
        num_dispatch_devices: int = 1,
        topk_weights_repeat: Optional[Dict] = None,
    ) -> MoEPreambleConfig:
        """
        Create preamble configuration from HF config.

        Args:
            backend: "deepseek" or "gptoss"
            hf_config: HuggingFace model configuration
            batch_size_per_device: Batch size per device
            seq_len: Sequence length (1 for decode)
            num_dispatch_devices: Number of dispatch devices
            topk_weights_repeat: Repeat config for DeepSeek weights

        Returns:
            MoEPreambleConfig instance
        """
        return MoEPreambleConfig(
            backend=backend,
            hidden_size=hf_config.hidden_size,
            num_experts_per_tok=hf_config.num_experts_per_tok,
            num_experts_per_device=getattr(
                hf_config, "num_experts_per_device", hf_config.num_local_experts // num_dispatch_devices
            ),
            num_devices=getattr(hf_config, "num_devices", 32),
            num_dispatch_devices=num_dispatch_devices,
            batch_size_per_device=batch_size_per_device,
            seq_len=seq_len,
            topk_weights_repeat=topk_weights_repeat if backend == "deepseek" else None,
            use_throughput_experts=backend == "gptoss",
        )
