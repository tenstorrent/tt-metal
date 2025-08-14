import torch

import ttnn
from models.tt_transformers.tt.multimodal.llama_layernorm import TtLayerNorm


def siglip_layer_norm_ttnn(
    mesh_device,
    hidden_states,
    state_dict,
    dim: int,
    eps: float = 1e-05,
    state_dict_prefix: str = "",
    weight_cache_path: str = None,
    weight_memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    dtype=ttnn.bfloat16,
):
    layer_norm = TtLayerNorm(
        device=mesh_device,
        dim=dim,
        state_dict=state_dict,
        state_dict_prefix=state_dict_prefix,
        weight_cache_path=weight_cache_path,
        weight_memory_config=weight_memory_config,
        weight_dtype=dtype,
        model_config=None,
        eps=eps,
    )
    if isinstance(hidden_states, torch.Tensor):
        torch_in_type = hidden_states.dtype
        hidden_states = ttnn.from_torch(
            hidden_states,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        output = layer_norm(hidden_states)
        output = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).to(torch_in_type)[0]
    else:
        output = layer_norm(hidden_states)
    return output
