"""Per-layer KV cache buffers for Gemma4ForCausalLM.

Each decoder layer reads from / writes to a persistent K and V tensor
held on the model. Sliding layers use shape `[1, 4, 256, 256]`, full
layers use `[1, 1, 256, 512]` — both BFLOAT16 TILE replicated across
the (1, 4) mesh. These match the slot recipes in `runtime_inputs.py`
that this class supersedes.
"""
import torch

import ttnn

_DRAM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
_SLIDING_SHAPE = (1, 4, 256, 256)
_FULL_SHAPE = (1, 1, 256, 512)


def _zero_kv(shape, mesh_device):
    return ttnn.as_tensor(
        torch.zeros(list(shape), dtype=torch.bfloat16),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=mesh_device,
        memory_config=_DRAM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def cache_shape(layer_type):
    return _SLIDING_SHAPE if layer_type == "sliding" else _FULL_SHAPE


class Gemma4Caches:
    """Holds K and V tensors for all 60 decoder layers, indexed by layer_idx.

    Layers receive their slot from `caches.k_caches[layer_idx]` /
    `caches.v_caches[layer_idx]` at construction. `reset()` rebuilds
    every slot with fresh zeros — callers must re-distribute the new
    references to layers (the model's `reset_kv_caches` does this).
    """

    def __init__(self, mesh_device, layer_types):
        self.mesh_device = mesh_device
        self.layer_types = list(layer_types)
        self.k_caches = [_zero_kv(cache_shape(t), mesh_device) for t in self.layer_types]
        self.v_caches = [_zero_kv(cache_shape(t), mesh_device) for t in self.layer_types]

    def reset(self):
        """Rebuild every K/V slot with fresh zero tensors."""
        for i, t in enumerate(self.layer_types):
            self.k_caches[i] = _zero_kv(cache_shape(t), self.mesh_device)
            self.v_caches[i] = _zero_kv(cache_shape(t), self.mesh_device)
