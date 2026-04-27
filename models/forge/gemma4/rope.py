"""HF-derived RoPE inv_freq setup for the gemma4 prelude consteval cache.

Phase 3 source-of-truth for the prelude-consumed inv_freq tables.
Mirrors `tt_transformers/tt/rope.py:RotarySetup` but much smaller:
gemma4 computes cos/sin online inside the prelude via
`ttnn.matmul + ttnn.cos/sin`, so the consteval cache only stores the
reshaped inv_freq inputs (not pre-computed cos/sin tables).

Derivation table (Phase 3 Task 1 inventory):

    | Side    | Key                  | Source                        | Output                        |
    |---------|----------------------|-------------------------------|-------------------------------|
    | prefill | main_const_eval_487  | sliding_attention_inv_freq    | [1,128,1] f32 TILE replicated |
    | prefill | main_const_eval_129  | full_attention_inv_freq       | [1,256,1] f32 TILE replicated |
    | decode  | main_const_eval_626  | sliding_attention_inv_freq    | [1,128,1] f32 TILE replicated |
    | decode  | main_const_eval_75   | full_attention_inv_freq       | [1,256,1] f32 TILE replicated |

Both sides consume the same HF lifted buffers (computed by
`transformers.Gemma4TextRotaryEmbedding`). The numbering differs between
prefill and decode because each side has its own codegen artifact.
"""
import torch

# Per-side cached_main key map for rope-derived consteval entries.
_ROPE_KEYS = {
    "prefill": {
        "sliding_inv_freq": "main_const_eval_487",
        "full_inv_freq": "main_const_eval_129",
    },
    "decode": {
        "sliding_inv_freq": "main_const_eval_626",
        "full_inv_freq": "main_const_eval_75",
    },
}


class RoPESetup:
    """Holds reshaped inv_freq tables for sliding and full attention.

    Both tables are FLOAT32 TILE-layout, replicated across the (1,4)
    mesh, in INTERLEAVED DRAM. Constructor reshapes the HF inv_freq
    buffers (shape [N]) to [1, N, 1] — the shape the prelude matmul
    expects as the LHS of `inv_freq @ position_ids`.
    """

    def __init__(self, *, sliding_inv_freq, full_inv_freq, mesh_device, is_decode):
        import ttnn

        self.is_decode = is_decode
        self.mesh_device = mesh_device
        self._mem_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
        self.sliding = self._build(sliding_inv_freq.reshape(1, -1, 1))
        self.full = self._build(full_inv_freq.reshape(1, -1, 1))

    def _build(self, torch_t):
        import ttnn

        return ttnn.as_tensor(
            torch_t.to(torch.float32),
            dtype=ttnn.DataType.FLOAT32,
            layout=ttnn.Layout.TILE,
            device=self.mesh_device,
            memory_config=self._mem_cfg,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    @classmethod
    def from_hf(cls, hf, mesh_device, *, is_decode):
        return cls(
            sliding_inv_freq=hf.lifted["model.rotary_emb.sliding_attention_inv_freq"],
            full_inv_freq=hf.lifted["model.rotary_emb.full_attention_inv_freq"],
            mesh_device=mesh_device,
            is_decode=is_decode,
        )

    def populate_cached_main(self, cached_main):
        """Override the prelude-derived consteval keys with HF-equivalent
        ttnn.Tensors. Mutates cached_main in place.
        """
        keys = _ROPE_KEYS["decode" if self.is_decode else "prefill"]
        cached_main[keys["sliding_inv_freq"]] = [self.sliding]
        cached_main[keys["full_inv_freq"]] = [self.full]
