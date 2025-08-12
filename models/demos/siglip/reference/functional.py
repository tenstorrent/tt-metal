# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Functional stubs for SigLIP Vision modules that match input/output shapes.
These are lightweight implementations for testing and development.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from models.demos.gemma3.tests.common import convert_state_dict


def siglip_attention(
    hidden_states: torch.Tensor,
    state_dict: Dict,
    state_dict_prefix: str,
    weight_cache_path: str,
    dtype: torch.dtype = torch.bfloat16,
    vision_dim: int = 1152,
    num_heads: int = 16,
    patch_size: int = 14,
    dropout: float = 0.0,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SigLIP self-attention mechanism."""
    state_dict = convert_state_dict(state_dict)
    batch_size, seq_length, embed_dim = hidden_states.shape
    head_dim = vision_dim // num_heads

    # Project to Q, K, V
    queries = F.linear(hidden_states, state_dict["q_proj"]["weight"], state_dict["q_proj"].get("bias"))
    keys = F.linear(hidden_states, state_dict["k_proj"]["weight"], state_dict["k_proj"].get("bias"))
    values = F.linear(hidden_states, state_dict["v_proj"]["weight"], state_dict["v_proj"].get("bias"))

    # Reshape to multi-head format
    queries = queries.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
    keys = keys.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
    values = values.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)

    # Compute attention scores
    scale = head_dim**-0.5
    attn_weights = torch.matmul(queries, keys.transpose(-1, -2)) * scale

    # Apply attention mask if provided
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # Apply softmax
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(queries.dtype)

    # Apply dropout if specified
    if dropout > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout, training=False)

    # Apply attention to values
    attn_output = torch.matmul(attn_weights, values)

    # Reshape back to original format
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_length, embed_dim)

    # Final output projection
    attn_output = F.linear(attn_output, state_dict["out_proj"]["weight"], state_dict["out_proj"].get("bias"))

    return attn_output, attn_weights

    """SigLIP for image classification."""
    # Get vision features (without pooling head for classification)
    last_hidden_state, _ = siglip_vision_transformer(
        pixel_values,
        state_dict["vision_model"],
        interpolate_pos_encoding=interpolate_pos_encoding,
        patch_size=patch_size,
        image_size=image_size,
        num_hidden_layers=num_hidden_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        layer_norm_eps=layer_norm_eps,
        hidden_act=hidden_act,
        attention_dropout=attention_dropout,
        use_head=False,  # Don't use the attention pooling head for classification
    )

    # Global average pooling over patch tokens
    sequence_output = torch.mean(last_hidden_state, dim=1)  # [B, hidden_size]

    # Classification head
    logits = F.linear(sequence_output, state_dict["classifier"]["weight"], state_dict["classifier"].get("bias"))

    return logits
