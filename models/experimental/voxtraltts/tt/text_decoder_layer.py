# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import ttnn
from models.tt_transformers.tt.decoder import TransformerBlock


def remap_voxtral_text_state_dict(state_dict: dict[str, object]) -> dict[str, object]:
    """Normalize alternate/HF text keys to tt_transformers text key format."""
    remapped: dict[str, object] = {}
    for key, value in state_dict.items():
        new_key = key
        # HF-style projection naming -> tt_transformers text naming.
        new_key = new_key.replace(".self_attn.q_proj.", ".attention.wq.")
        new_key = new_key.replace(".self_attn.k_proj.", ".attention.wk.")
        new_key = new_key.replace(".self_attn.v_proj.", ".attention.wv.")
        new_key = new_key.replace(".self_attn.o_proj.", ".attention.wo.")
        new_key = new_key.replace(".mlp.gate_proj.", ".feed_forward.w1.")
        new_key = new_key.replace(".mlp.down_proj.", ".feed_forward.w2.")
        new_key = new_key.replace(".mlp.up_proj.", ".feed_forward.w3.")
        new_key = new_key.replace(".input_layernorm.", ".attention_norm.")
        new_key = new_key.replace(".post_attention_layernorm.", ".ffn_norm.")
        new_key = new_key.replace("model.embed_tokens.", "tok_embeddings.")
        new_key = new_key.replace("mm_audio_embeddings.tok_embeddings.", "tok_embeddings.")
        new_key = new_key.replace("model.norm.", "norm.")
        new_key = new_key.replace("lm_head.", "output.")
        if new_key.startswith("model.layers."):
            new_key = "layers." + new_key[len("model.layers.") :]
        remapped[new_key] = value
    return remapped


class VoxtralTTTextDecoderLayer:
    """Thin wrapper that directly reuses tt_transformers TransformerBlock."""

    def __init__(self, inner_block: TransformerBlock) -> None:
        self.inner = inner_block

    @classmethod
    def create(
        cls,
        *,
        args,
        mesh_device,
        tt_ccl,
        dtype: ttnn.DataType,
        state_dict: dict[str, object],
        layer_num: int,
        weight_cache_path: Path | None,
        transformation_mats,
        paged_attention_config=None,
        use_paged_kv_cache: bool = False,
        attention_class=None,
        prefetcher=None,
    ) -> "VoxtralTTTextDecoderLayer":
        inner = TransformerBlock(
            args=args,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            dtype=dtype,
            state_dict=remap_voxtral_text_state_dict(state_dict),
            layer_num=layer_num,
            weight_cache_path=weight_cache_path,
            transformation_mats=transformation_mats,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
            attention_class=attention_class,
            prefetcher=prefetcher,
        )
        return cls(inner)

    def __call__(self, *args, **kwargs):
        return self.inner(*args, **kwargs)
