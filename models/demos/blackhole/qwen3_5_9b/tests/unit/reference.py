# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""HF-golden reference builders for qwen3_5_9b unit tests.

The installed transformers ships native Qwen3.5 modeling code, so these tests
compare TT modules against the real HF layer (the gemma4 test_attention.py
pattern) instead of a hand-written torch reference:
* model_path()/text_config() resolve the checkpoint (HF_MODEL or DEFAULT_CKPT)
* hf_rope() builds the HF rotary embedding for the RoPE comparison
* causal_mask() builds the explicit mask the eager attention path needs
Single-layer loads keep host RAM in the low-GB range (loading the full model OOMs).
"""
import os

import torch

# Default reference checkpoint (hub id); HF_MODEL overrides.
DEFAULT_CKPT = "Qwen/Qwen3.6-27B"


def model_path():
    return os.path.expanduser(os.environ.get("HF_MODEL", DEFAULT_CKPT))


def text_config(checkpoint_dir=None):
    """Parsed Qwen3_5TextConfig with the eager attention path selected.

    Eager is required so the HF reference uses the explicit attention_mask we
    pass (sdpa/flash backends ignore or re-derive masks differently).
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(checkpoint_dir or model_path(), trust_remote_code=True).get_text_config()
    cfg._attn_implementation = "eager"
    return cfg


def hf_attention(cfg, layer_state_dict):
    """Qwen3_5Attention with the given raw layer weights, float32.

    q_norm/k_norm weights must be the RAW checkpoint values: the HF module adds
    the zero-centered +1 internally, while the TT loaders bake it into the
    weight — using raw weights on the HF side keeps both conventions honest.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

    attn = Qwen3_5Attention(cfg, layer_idx=0).to(torch.float32)
    attn.load_state_dict({k: v.float() for k, v in layer_state_dict.items()})
    attn.eval()
    return attn


def hf_mlp(cfg, mlp_state_dict):
    """Qwen3_5MLP (SwiGLU) with the given layer weights, float32."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP

    mlp = Qwen3_5MLP(cfg, intermediate_size=cfg.intermediate_size).to(torch.float32)
    mlp.load_state_dict({k: v.float() for k, v in mlp_state_dict.items()})
    mlp.eval()
    return mlp


def hf_rope(cfg):
    """Qwen3_5TextRotaryEmbedding. For text-only position_ids the interleaved
    M-RoPE reduces to standard partial RoPE (all three position streams are
    identical), so this is directly comparable to rope_tp's tables."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    return Qwen3_5TextRotaryEmbedding(cfg)


def causal_mask(seq_len):
    """[1,1,S,S] additive causal mask (upper triangle = -inf) for eager attention."""
    return torch.full((1, 1, seq_len, seq_len), float("-inf")).triu(1)
