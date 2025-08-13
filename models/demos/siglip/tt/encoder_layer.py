from typing import Dict, Optional

import torch

import ttnn
from models.demos.siglip.reference.functional import siglip_mlp
from models.demos.siglip.tt.attention import siglip_attention_ttnn
from models.demos.siglip.tt.layer_norm import siglip_layer_norm_ttnn


def siglip_encoder_layer_ttnn(
    mesh_device,
    hidden_states: torch.Tensor,
    state_dict: Dict,
    state_dict_prefix: str = "",
    weight_cache_path: str = None,
    dtype=ttnn.bfloat16,
    attention_mask: Optional[torch.Tensor] = None,
    vision_dim: int = 1152,
    num_heads: int = 16,
    layer_norm_eps: float = 1e-5,
    hidden_act: str = "gelu",
    attention_dropout: float = 0.0,
) -> torch.Tensor:
    # Convert torch to TT here so we can test the whole encoder layer with TT tensors
    if isinstance(hidden_states, torch.Tensor):
        hidden_states = ttnn.from_torch(
            hidden_states,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
    # Self-attention block
    residual = hidden_states

    # Pre-attention layer norm
    # hidden_states = siglip_layer_norm(hidden_states, state_dict["layer_norm1"], layer_norm_eps)
    hidden_states = siglip_layer_norm_ttnn(
        mesh_device, hidden_states, state_dict["layer_norm1"], hidden_states.shape[-1], layer_norm_eps
    )

    # Self-attention
    hidden_states, attn_weights = siglip_attention_ttnn(
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
    # hidden_states = siglip_layer_norm(hidden_states, state_dict["layer_norm2"], layer_norm_eps)
    hidden_states = siglip_layer_norm_ttnn(
        mesh_device, hidden_states, state_dict["layer_norm2"], hidden_states.shape[-1], layer_norm_eps
    )

    # MLP
    # Use torch here until we have TT MLP
    if isinstance(hidden_states, ttnn.Tensor):
        hidden_states = ttnn.to_torch(hidden_states, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).to(
            state_dict["mlp"]["fc1"]["weight"].dtype
        )[0]

    hidden_states = siglip_mlp(hidden_states, state_dict["mlp"], hidden_act)

    # Convert back to TT so we can add the residual connection
    if isinstance(hidden_states, torch.Tensor):
        hidden_states = ttnn.from_torch(
            hidden_states,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    # Add residual connection
    hidden_states = residual + hidden_states

    return ttnn.to_torch(hidden_states, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0]
