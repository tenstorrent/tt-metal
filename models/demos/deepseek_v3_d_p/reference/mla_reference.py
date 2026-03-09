# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Reference CPU implementation of DeepSeek V3 Multi-Latent Attention (MLA) module.
This module can run with both downloaded weights from HuggingFace or random weights for testing.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from models.demos.deepseek_v3.reference.configuration_deepseek import DeepseekV3Config
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3Attention, DeepseekV3RMSNorm


class MLAReference(nn.Module):
    """
    Reference CPU implementation of Multi-Latent Attention (MLA) for DeepSeek V3.

    This is a wrapper around the DeepseekV3Attention module that provides:
    - Easy initialization with random or pretrained weights
    - Simplified interface for testing
    - Support for both prefill and decode modes

    Args:
        config: DeepseekV3Config with model configuration
        layer_idx: Index of the layer (used for caching)
    """

    def __init__(self, config: DeepseekV3Config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Initialize the attention module
        self.attention = DeepseekV3Attention(config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[any]]:
        """
        Forward pass of MLA module.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask of shape [batch_size, 1, seq_len, seq_len]
            position_ids: Position IDs of shape [batch_size, seq_len]
            past_key_value: Cache for past key/value states
            output_attentions: Whether to output attention weights
            use_cache: Whether to use caching

        Returns:
            Tuple of (output_tensor, attention_weights, updated_cache)
        """
        return self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

    @classmethod
    def from_pretrained(
        cls,
        config: DeepseekV3Config,
        state_dict: dict[str, torch.Tensor],
        layer_idx: int = 0,
        module_path: str = "model.layers.0.self_attn",
    ) -> "MLAReference":
        """
        Create MLA module from pretrained weights.

        Args:
            config: Model configuration
            state_dict: State dict containing pretrained weights
            layer_idx: Layer index (default: 0)
            module_path: Path prefix in state dict (default: "model.layers.0.self_attn")

        Returns:
            MLAReference module with loaded weights
        """
        module = cls(config, layer_idx=layer_idx)

        # Extract module-specific state dict
        if module_path:
            prefix = module_path + "."
            module_state_dict = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
        else:
            module_state_dict = state_dict

        # Load weights
        module.attention.load_state_dict(module_state_dict)
        return module

    @classmethod
    def from_random(
        cls,
        config: DeepseekV3Config,
        layer_idx: int = 0,
        seed: int = 42,
    ) -> "MLAReference":
        """
        Create MLA module with random weights for testing.

        Args:
            config: Model configuration
            layer_idx: Layer index (default: 0)
            seed: Random seed for reproducibility (default: 42)

        Returns:
            MLAReference module with random weights
        """
        # Set seed for reproducibility
        torch.manual_seed(seed)

        # Create module (weights are randomly initialized)
        module = cls(config, layer_idx=layer_idx)

        # Initialize weights using standard initialization
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=config.initializer_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, DeepseekV3RMSNorm):
                if hasattr(m, "weight"):
                    nn.init.ones_(m.weight)

        module.apply(init_weights)
        return module


def create_mla_reference(
    config: DeepseekV3Config,
    state_dict: Optional[dict[str, torch.Tensor]] = None,
    layer_idx: int = 0,
    module_path: str = "model.layers.0.self_attn",
) -> MLAReference:
    """
    Convenient factory function to create MLA reference module.

    Args:
        config: Model configuration
        state_dict: State dict with weights (passed in, either pretrained weights or random weights)
        layer_idx: Layer index (default: 0)
        module_path: Path to module in state dict (default: "model.layers.0.self_attn")

    Returns:
        MLAReference module
    """

    if state_dict is None:
        raise ValueError("state_dict must be provided")
    return MLAReference.from_pretrained(config, state_dict, layer_idx, module_path)
