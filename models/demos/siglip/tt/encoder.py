from typing import Dict

import torch

import ttnn
from models.demos.siglip.tt.encoder_layer import siglip_encoder_layer_ttnn


def siglip_encoder_ttnn(
    mesh_device,
    hidden_states: ttnn.Tensor,
    state_dict: Dict,
    state_dict_prefix: str = "",
    weight_cache_path: str = None,
    dtype=ttnn.bfloat16,
) -> ttnn.Tensor:
    if isinstance(hidden_states, torch.Tensor):
        hidden_states = ttnn.from_torch(
            hidden_states,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
    for layer in state_dict["layers"]:
        hidden_states = siglip_encoder_layer_ttnn(
            mesh_device,
            hidden_states,
            state_dict["layers"][layer],
            state_dict_prefix,
            weight_cache_path,
            dtype,
        )
        hidden_states = hidden_states[0]
    return hidden_states
