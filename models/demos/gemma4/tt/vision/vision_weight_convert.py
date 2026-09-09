# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""HF → Meta weight conversion for the Gemma-4 vision encoder.

Gemma-4 vision uses multidimensional (2D) RoPE: ``head_dim`` is split into
``ndim`` independent blocks (rows / columns), and standard 1D RoPE is applied
to each block separately. The TT kernels apply
``ttnn.experimental.rotary_embedding_llama`` (Meta interleaved convention) per
block, so q/k weights, q/k norms, and cos/sin must be converted to Meta format
*per block* rather than across the full head_dim. The shared LLM helpers only
handle 1D RoPE and would mix the two spatial dimensions.
"""

import torch

from models.tt_transformers.tt.load_checkpoints import map_hf_to_meta_keys

# Rename only the projection submodules to the names the Gemma-4 vision modules
# read. Do not rename the attention container (self_attn) or the layernorms —
# unlike stock LLM ``map_hf_to_meta_keys``.
_VISION_BLOCK_PROJ_RENAMES = [
    (".q_proj.", ".wq."),
    (".k_proj.", ".wk."),
    (".v_proj.", ".wv."),
    (".o_proj.", ".wo."),
    (".gate_proj.", ".w1."),
    (".up_proj.", ".w3."),
    (".down_proj.", ".w2."),
]


def meta_permute_qk_weight(weight, n_heads, head_dim, ndim=2):
    """HF→Meta interleave for q/k projection weights, applied independently per RoPE block."""
    blk = head_dim // ndim
    in_dim = weight.shape[-1]
    weight = weight.view(n_heads, ndim, 2, blk // 2, in_dim)
    weight = weight.transpose(2, 3)
    return weight.reshape(n_heads * head_dim, in_dim)


def meta_permute_norm_weight(weight, head_dim, ndim=2):
    """HF→Meta interleave for q/k RMSNorm weights, applied independently per RoPE block."""
    blk = head_dim // ndim
    weight = weight.view(ndim, 2, blk // 2)
    weight = weight.transpose(1, 2)
    return weight.reshape(head_dim)


def convert_rope_style_hf_to_meta_md(cos, sin, ndim=2):
    """Convert HF cos/sin (per-block half-duplicated) to Meta pairwise-duplicated, per RoPE block."""
    blk = cos.shape[-1] // ndim
    cos_parts, sin_parts = [], []
    for k in range(ndim):
        c = cos[..., k * blk : (k + 1) * blk]
        s = sin[..., k * blk : (k + 1) * blk]
        cos_parts.append(torch.repeat_interleave(c[..., : blk // 2], 2, dim=-1))
        sin_parts.append(torch.repeat_interleave(s[..., : blk // 2], 2, dim=-1))
    return torch.cat(cos_parts, dim=-1), torch.cat(sin_parts, dim=-1)


def convert_vision_attention_hf_to_meta(state_dict, n_heads, n_kv_heads, head_dim, ndim=2):
    """Block-aware HF→Meta conversion for the Gemma-4 vision self-attention weights."""
    converted = {}
    for key, tensor in state_dict.items():
        if "q_proj.linear.weight" in key:
            converted[key] = meta_permute_qk_weight(tensor, n_heads, head_dim, ndim)
        elif "k_proj.linear.weight" in key:
            converted[key] = meta_permute_qk_weight(tensor, n_kv_heads, head_dim, ndim)
        elif "q_norm.weight" in key or "k_norm.weight" in key:
            converted[key] = meta_permute_norm_weight(tensor, head_dim, ndim)
        else:
            converted[key] = tensor
    return map_hf_to_meta_keys(converted)


def convert_vision_block_hf_to_meta(state_dict, n_heads, n_kv_heads, head_dim, ndim=2):
    """Block-aware HF→Meta conversion for a full Gemma-4 vision encoder layer.

    Applies the per-block RoPE permute to the attention q/k weights and q/k norms,
    renames only the projection submodules (q/k/v/o_proj → wq/wk/wv/wo and
    gate/up/down_proj → w1/w3/w2), and preserves the HF container/norm key names
    (self_attn, input_layernorm, post_attention_layernorm, pre/post_feedforward_layernorm)
    that the Gemma-4 vision modules expect.
    """
    converted = {}
    for key, tensor in state_dict.items():
        if "q_proj.linear.weight" in key:
            tensor = meta_permute_qk_weight(tensor, n_heads, head_dim, ndim)
        elif "k_proj.linear.weight" in key:
            tensor = meta_permute_qk_weight(tensor, n_kv_heads, head_dim, ndim)
        elif "q_norm.weight" in key or "k_norm.weight" in key:
            tensor = meta_permute_norm_weight(tensor, head_dim, ndim)
        new_key = key
        for old, new in _VISION_BLOCK_PROJ_RENAMES:
            new_key = new_key.replace(old, new)
        converted[new_key] = tensor
    return converted
