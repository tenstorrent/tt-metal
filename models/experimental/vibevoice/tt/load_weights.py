# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
VibeVoice-1.5B weight loading utilities.

All torch usage here is host-only (reading safetensors, building tensors).
Forward paths in tt/ must not import or call these functions at runtime.
"""

import json
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import load_file as safetensors_load_file


def load_vibevoice_state_dict(model_path: str) -> Dict[str, torch.Tensor]:
    """Load the full VibeVoice model state dict from safetensors (host only)."""
    model_path = Path(model_path)
    index_path = model_path / "model.safetensors.index.json"

    if index_path.exists():
        with open(index_path) as f:
            index_data = json.load(f)
        weight_map = index_data["weight_map"]
        shard_files = set(weight_map.values())
        state_dict: Dict[str, torch.Tensor] = {}
        for shard_file in shard_files:
            shard_path = model_path / shard_file
            state_dict.update(safetensors_load_file(str(shard_path)))
    else:
        single_path = model_path / "model.safetensors"
        if not single_path.exists():
            raise FileNotFoundError(f"No model.safetensors(.index.json) found in {model_path}")
        state_dict = safetensors_load_file(str(single_path))

    return state_dict


# Key prefix constants
_LM_PREFIX = "model.language_model."
_LM_HEAD_PREFIX = "lm_head."
_ACOUSTIC_CONN_PREFIX = "model.acoustic_connector."
_SEMANTIC_CONN_PREFIX = "model.semantic_connector."
_DIFFUSION_HEAD_PREFIX = "model.prediction_head."
_ACOUSTIC_TOK_PREFIX = "model.acoustic_tokenizer."
_SEMANTIC_TOK_PREFIX = "model.semantic_tokenizer."

_SUBMODULE_PREFIXES = {
    "lm": _LM_PREFIX,
    "lm_head": _LM_HEAD_PREFIX,
    "acoustic_connector": _ACOUSTIC_CONN_PREFIX,
    "semantic_connector": _SEMANTIC_CONN_PREFIX,
    "diffusion_head": _DIFFUSION_HEAD_PREFIX,
    "acoustic_tokenizer": _ACOUSTIC_TOK_PREFIX,
    "semantic_tokenizer": _SEMANTIC_TOK_PREFIX,
}


def split_submodule_weights(
    state: Dict[str, torch.Tensor],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Split a full VibeVoice state dict into per-submodule dicts.

    Keys in returned dicts have the submodule prefix stripped.
    """
    result: Dict[str, Dict[str, torch.Tensor]] = {k: {} for k in _SUBMODULE_PREFIXES}
    for full_key, tensor in state.items():
        for module_name, prefix in _SUBMODULE_PREFIXES.items():
            if full_key.startswith(prefix):
                short_key = full_key[len(prefix) :]
                result[module_name][short_key] = tensor
                break
    return result


def remap_lm_keys_to_tt_transformers(vv_lm_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remap VibeVoice language model keys to tt_transformers naming convention.

    VibeVoice LM uses HF Qwen2 naming (layers.N.self_attn.q_proj.weight, etc.).
    tt_transformers standardize_hf_keys converts these to its own format:
      embed_tokens.weight → tok_embeddings.weight
      layers.N.self_attn.q_proj → layers.N.attention.wq
      layers.N.self_attn.k_proj → layers.N.attention.wk
      layers.N.self_attn.v_proj → layers.N.attention.wv
      layers.N.self_attn.o_proj → layers.N.attention.wo
      layers.N.mlp.gate_proj    → layers.N.feed_forward.w1
      layers.N.mlp.up_proj      → layers.N.feed_forward.w3
      layers.N.mlp.down_proj    → layers.N.feed_forward.w2
      layers.N.input_layernorm  → layers.N.attention_norm
      layers.N.post_attention_layernorm → layers.N.ffn_norm
      model.norm.weight         → norm.weight (already stripped of model. prefix by split)
    """
    remap: Dict[str, torch.Tensor] = {}
    for k, v in vv_lm_dict.items():
        # The vv_lm_dict already has the "model.language_model." prefix stripped.
        # Keys look like: embed_tokens.weight, layers.N.self_attn.q_proj.weight, norm.weight
        new_k = k
        new_k = new_k.replace("embed_tokens.", "tok_embeddings.")
        new_k = new_k.replace(".self_attn.q_proj", ".attention.wq")
        new_k = new_k.replace(".self_attn.k_proj", ".attention.wk")
        new_k = new_k.replace(".self_attn.v_proj", ".attention.wv")
        new_k = new_k.replace(".self_attn.o_proj", ".attention.wo")
        new_k = new_k.replace(".mlp.gate_proj", ".feed_forward.w1")
        new_k = new_k.replace(".mlp.up_proj", ".feed_forward.w3")
        new_k = new_k.replace(".mlp.down_proj", ".feed_forward.w2")
        new_k = new_k.replace(".input_layernorm", ".attention_norm")
        new_k = new_k.replace(".post_attention_layernorm", ".ffn_norm")
        remap[new_k] = v
    return remap


def fold_weight_norm(state: Dict[str, torch.Tensor], prefix: str = "") -> Dict[str, torch.Tensor]:
    """Fold weight_norm parametrization (weight_g + weight_v → weight) for conv layers.

    Operates on keys with the given prefix. Returns a new dict with folded weights.
    """
    result: Dict[str, torch.Tensor] = {}
    g_keys = {k for k in state if k.endswith("_g")}
    for k, v in state.items():
        g_key = k[:-2] + "_g" if k.endswith("_v") else None
        if k.endswith("_v") and g_key in g_keys:
            # fold: w = weight_v / ||weight_v|| * weight_g
            weight_v = v
            weight_g = state[g_key]
            norm_v = weight_v.view(weight_v.shape[0], -1).norm(dim=1, keepdim=True)
            # reshape norm_v to broadcast correctly against weight_v shape
            shape = [-1] + [1] * (weight_v.dim() - 1)
            norm_v = norm_v.view(shape)
            g_shape = [-1] + [1] * (weight_g.dim() - 1)
            weight_g = weight_g.view(g_shape)
            result[k[:-2]] = (weight_v / norm_v) * weight_g
        elif not k.endswith("_g"):
            result[k] = v
    return result
