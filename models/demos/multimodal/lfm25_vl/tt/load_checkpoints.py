# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""HF LiquidAI/LFM2.5-VL-1.6B checkpoint -> tt_transformers ("meta style") key conversion.

LFM2.5-VL is a hybrid text backbone (``lfm2``): most decoder layers are depthwise
``ShortConv`` layers, interleaved with a handful of full-attention layers (see
``text_config.layer_types``). The vision tower is a SigLIP2 (NaFlex) encoder and is
converted with the *exact same* generic helpers ``tt_transformers`` already uses for
other SigLIP-based VLMs (Gemma3), since the module structure (patch embedding + encoder
blocks + post layernorm) and naming conventions coincide.

Key renames applied to the text tower (see module docstring on each helper below):
    - ``model.language_model.`` prefix stripped
    - ``embed_tokens`` -> ``tok_embeddings``, ``lm_head`` -> ``output``
    - ``embedding_norm`` (final norm) -> ``norm``
    - ``operator_norm`` (pre-attn/conv norm, present on *every* layer) -> ``attention_norm``
    - ``ffn_norm`` kept as-is
    - Attention layers: ``self_attn`` -> ``attention``, ``q_proj/k_proj/v_proj`` -> ``wq/wk/wv``,
      ``out_proj`` -> ``wo``, ``q_layernorm/k_layernorm`` -> ``q_norm/k_norm`` (RoPE-permuted,
      like every other tt_transformers model with per-head qk-norm)
    - Conv (``ShortConv``) layers: ``conv.in_proj`` / ``conv.out_proj`` / ``conv.conv.weight``
      kept unchanged (no analogue in tt_transformers, read directly by ``TtLfm2ShortConv``)
    - ``feed_forward.gate_proj/up_proj/down_proj`` -> ``feed_forward.w1/w3/w2`` (both layer types)

Vision tower and projector keys keep their original HF prefixes
(``model.vision_tower.vision_model....`` / ``model.multi_modal_projector....``), matching the
convention used by ``models/demos/multimodal/gemma3``.
"""

from models.tt_transformers.tt.load_checkpoints import (
    convert_hf_qkv_to_meta_format,
    map_hf_to_meta_keys_vision_only,
    replace_keys,
    split_hf_keys,
)

LANGUAGE_MODEL_PREFIX = "model.language_model."


def _insert_siglip_vision_model_level(state_dict):
    """Re-insert the SiglipVisionModel `.vision_model` level for transformers >=5.x.

    Mirrors ``models.demos.multimodal.gemma3.tt.load_checkpoints._insert_siglip_vision_model_level``:
    newer transformers versions flatten ``SiglipVisionModel`` so ``embeddings`` / ``encoder`` /
    ``post_layernorm`` become direct attributes of the vision tower instead of being nested under a
    ``.vision_model`` wrapper. Re-insert the level so every downstream prefix stays consistent
    regardless of the installed transformers version. No-op if already present.
    """
    out = {}
    for k, v in state_dict.items():
        if "vision_tower." in k and "vision_tower.vision_model." not in k:
            k = k.replace("vision_tower.", "vision_tower.vision_model.", 1)
        out[k] = v
    return out


def split_lfm_state_dict(state_dict):
    """Split a full Lfm2VlForConditionalGeneration state dict into vision / text / other."""
    vision_state_dict, text_state_dict, other_state_dict = {}, {}, {}
    for k, v in state_dict.items():
        if k.startswith("model.vision_tower"):
            vision_state_dict[k] = v
        elif k.startswith("model.language_model") or k.startswith("lm_head"):
            text_state_dict[k] = v
        else:
            # model.multi_modal_projector.* (kept with its original HF prefix)
            other_state_dict[k] = v
    return vision_state_dict, text_state_dict, other_state_dict


def convert_vision_hf_to_meta(state_dict, head_dim):
    """Convert the SigLIP2 vision-tower keys using the generic tt_transformers vision mapper."""
    state_dict = split_hf_keys(state_dict)
    state_dict = map_hf_to_meta_keys_vision_only(state_dict)
    return state_dict


def convert_lfm_text_to_meta(state_dict, head_dim):
    """Convert the LFM2 hybrid text-tower keys (conv + full_attention layers) to meta style."""
    state_dict = {
        (k[len(LANGUAGE_MODEL_PREFIX) :] if k.startswith(LANGUAGE_MODEL_PREFIX) else k): v
        for k, v in state_dict.items()
    }

    # Tied embeddings fallback (LFM2 ties lm_head to embed_tokens when no separate lm_head is saved).
    if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"]

    # Rename qk-norms *before* the RoPE permutation pass below, which keys off "q_norm"/"k_norm".
    state_dict = replace_keys(state_dict, [("q_layernorm", "q_norm"), ("k_layernorm", "k_norm")])
    # Reverse-permutes q_proj/k_proj weights (and q_norm/k_norm) for tt_transformers' RoPE layout.
    # ShortConv layers have no q_proj/k_proj/q_norm/k_norm keys, so this is a no-op for them.
    state_dict = convert_hf_qkv_to_meta_format(state_dict, head_dim)

    replacements = [
        ("embed_tokens", "tok_embeddings"),
        ("lm_head", "output"),
        ("embedding_norm", "norm"),
        ("operator_norm", "attention_norm"),
        ("self_attn", "attention"),
        ("q_proj", "wq"),
        ("k_proj", "wk"),
        ("v_proj", "wv"),
        # Only attention's out_proj (already renamed to "attention.out_proj" above) is remapped to
        # "wo" here; ShortConv's "conv.out_proj" is deliberately left untouched.
        ("attention.out_proj", "attention.wo"),
        ("gate_proj", "w1"),
        ("down_proj", "w2"),
        ("up_proj", "w3"),
    ]
    state_dict = replace_keys(state_dict, replacements)
    return state_dict


def convert_lfm_hf_to_meta(state_dict, head_dim):
    """Convert a full Lfm2VlForConditionalGeneration HF state dict to tt_transformers meta style."""
    state_dict = _insert_siglip_vision_model_level(state_dict)
    vision_state_dict, text_state_dict, other_state_dict = split_lfm_state_dict(state_dict)
    vision_state_dict = convert_vision_hf_to_meta(vision_state_dict, head_dim)
    text_state_dict = convert_lfm_text_to_meta(text_state_dict, head_dim)
    return {**vision_state_dict, **text_state_dict, **other_state_dict}
