"""Scaled word embedding (Gemma4TextScaledWordEmbedding analog)."""
import ttnn


class ScaledEmbedding:
    """Token-id lookup followed by a scale by sqrt(hidden_size). Mirrors
    `Gemma4TextScaledWordEmbedding` from transformers. Shared between
    prefill and decode (shape-agnostic; the only shape diff is the
    sequence length of the input ids, which propagates through).

    Op sequence is bit-identical to the legacy
    `_embed_scaled(token_ids_tile, embed_weight, embed_scale)` helper.
    Consumes `token_ids_tile`; does NOT deallocate the weights.
    """

    def __init__(self, embed_weight, embed_scale):
        self.embed_weight = embed_weight
        self.embed_scale = embed_scale

    def __call__(self, token_ids_tile):
        as_uint32 = ttnn.typecast(
            token_ids_tile,
            ttnn.DataType.UINT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(token_ids_tile, False)
        as_row_major = ttnn.to_layout(as_uint32, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
        ttnn.deallocate(as_uint32, False)
        looked_up = ttnn.embedding(
            as_row_major,
            self.embed_weight,
            padding_idx=None,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(as_row_major, False)
        scaled = ttnn.multiply(
            looked_up,
            self.embed_scale,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(looked_up, False)
        return scaled

    @classmethod
    def from_consteval(cls, cached_main, embed_weight_ce, embed_scale_ce):
        return cls(
            cached_main[f"main_const_eval_{embed_weight_ce}"][0],
            cached_main[f"main_const_eval_{embed_scale_ce}"][0],
        )

    @classmethod
    def from_state_dict(cls, state_dict, lifted, mesh_device, *, weight_dtype=None, scale_dtype=None):
        """Build ScaledEmbedding from HF state_dict + lifted constants.

        embed_weight comes from `model.language_model.embed_tokens.weight`
        (vocab × hidden, sharded along hidden dim).
        embed_scale comes from the lifted constants
        `model.embed_tokens.embed_scale` (scalar bf16 = sqrt(hidden_size),
        replicated across the mesh).
        """
        import torch
        from gemma4 import weights as gw

        if weight_dtype is None:
            weight_dtype = ttnn.DataType.BFLOAT16
        if scale_dtype is None:
            scale_dtype = ttnn.DataType.BFLOAT16
        torch_w = state_dict["model.language_model.embed_tokens.weight"].to(torch.bfloat16)
        torch_s = lifted["model.embed_tokens.embed_scale"].to(torch.bfloat16)
        # embed_scale is a 0-d scalar; ttnn requires shape [1,1,1] for
        # broadcast-mul operands, so reshape.
        torch_s = torch_s.reshape(1, 1, 1)
        embed_weight = ttnn.as_tensor(
            torch_w,
            dtype=weight_dtype,
            layout=ttnn.Layout.ROW_MAJOR,
            device=mesh_device,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            mesh_mapper=gw.mesh_mapper_for_role("embed_tokens", mesh_device),
        )
        embed_scale = ttnn.as_tensor(
            torch_s,
            dtype=scale_dtype,
            layout=ttnn.Layout.TILE,
            device=mesh_device,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            mesh_mapper=gw.mesh_mapper_for_role("embed_scale", mesh_device),
        )
        return cls(embed_weight, embed_scale)
