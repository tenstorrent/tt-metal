# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Functional stubs for SigLIP Vision modules that match input/output shapes.
These are lightweight implementations for testing and development.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def siglip_attention(
    mesh_device,
    hidden_states: torch.Tensor,
    state_dict: Dict,
    state_dict_prefix: str,
    weight_cache_path: str,
    dtype: torch.dtype = torch.bfloat16,
    vision_dim: int = 1152,
    num_heads: int = 16,
    dropout: float = 0.0,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SigLIP self-attention mechanism."""
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


class SiglipMLP(nn.Module):
    """Siglip MLP module."""
    
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def gelu_tanh(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu_tanh(x)
        x = self.fc2(x)
        return x
        
def siglip_layer_norm(
    mesh_device,
    hidden_states,
    state_dict,
    dim: int = None,
    eps: float = 1e-05,
    state_dict_prefix: str = "",
    weight_cache_path: str = None,
    weight_memory_config=None,
    dtype=None,
) -> torch.Tensor:
    """Layer normalization."""
    return F.layer_norm(hidden_states, hidden_states.shape[-1:], state_dict["weight"], state_dict["bias"], eps)


def siglip_mlp(x: torch.Tensor, state_dict: Dict, hidden_act: str = "gelu") -> torch.Tensor:
    """SigLIP MLP with configurable activation function."""
    # First linear layer
    hidden_states = F.linear(x, state_dict["fc1"]["weight"], state_dict["fc1"].get("bias"))

    # Activation function
    if hidden_act == "gelu":
        hidden_states = F.gelu(hidden_states)
    elif hidden_act == "relu":
        hidden_states = F.relu(hidden_states)
    elif hidden_act == "silu":
        hidden_states = F.silu(hidden_states)
    else:
        # Default to GELU for SigLIP
        hidden_states = F.gelu(hidden_states)

    # Second linear layer
    hidden_states = F.linear(hidden_states, state_dict["fc2"]["weight"], state_dict["fc2"].get("bias"))

    return hidden_states


def siglip_encoder_layer(
    mesh_device,
    hidden_states: torch.Tensor,
    state_dict: Dict,
    state_dict_prefix: str = "",
    weight_cache_path: str = None,
    dtype: torch.dtype = torch.bfloat16,
    attention_mask: Optional[torch.Tensor] = None,
    vision_dim: int = 1152,
    num_heads: int = 16,
    layer_norm_eps: float = 1e-5,
    hidden_act: str = "gelu",
    attention_dropout: float = 0.0,
) -> torch.Tensor:
    """SigLIP encoder layer with pre-norm architecture."""
    # Self-attention block
    residual = hidden_states

    # Pre-attention layer norm
    hidden_states = siglip_layer_norm(
        mesh_device=mesh_device, hidden_states=hidden_states, state_dict=state_dict["layer_norm1"], eps=layer_norm_eps
    )

    # Self-attention
    hidden_states, attn_weights = siglip_attention(
        mesh_device=mesh_device,
        hidden_states=hidden_states,
        state_dict=state_dict["self_attn"],
        state_dict_prefix=state_dict_prefix,
        weight_cache_path=weight_cache_path,
        dtype=dtype,
        vision_dim=vision_dim,
        num_heads=num_heads,
        dropout=attention_dropout,
        attention_mask=attention_mask,
    )

    # Add residual connection
    hidden_states = residual + hidden_states

    # MLP block
    residual = hidden_states

    # Pre-MLP layer norm
    hidden_states = siglip_layer_norm(
        mesh_device=mesh_device, hidden_states=hidden_states, state_dict=state_dict["layer_norm2"], eps=layer_norm_eps
    )

    # MLP
    hidden_states = siglip_mlp(hidden_states, state_dict["mlp"], hidden_act)

    # Add residual connection
    hidden_states = residual + hidden_states

    return hidden_states
