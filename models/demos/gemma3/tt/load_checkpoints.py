# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import re

from models.tt_transformers.tt.load_checkpoints import (
    convert_hf_qkv_to_meta_format,
    convert_meta_qkv_to_hf_format,
    map_hf_to_meta_keys,
    replace_keys,
    split_hf_keys,
    standardize_hf_keys,
)


def standardize_hf_keys_multimodal(state_dict):
    all_keys = tuple(state_dict.keys())
    new_state_dict = {}
    for k in all_keys:
        if "model.visual." in k:
            new_state_dict[k.replace("model.visual.", "visual.")] = state_dict[k]
        elif "model.vision_tower.vision_model." in k:
            new_state_dict[k.replace("model.vision_tower.vision_model.", "visual.")] = state_dict[k]
        elif "model.language_model." in k:
            new_state_dict[k.replace("model.language_model.", "model.")] = state_dict[k]
        else:
            new_state_dict[k] = state_dict[k]

    # Standardize keys used in vision parts of Qwen2.5-VL
    state_dict = standardize_hf_keys(new_state_dict)
    replace_whole_name = lambda pattern, repl: lambda s: re.sub(rf"(^|\.)({pattern})($|\.)", rf"\1{repl}\3", s)
    output = {}
    for k, v in state_dict.items():
        k = replace_whole_name("qkv", "qkv_proj")(k)
        k = replace_whole_name("proj", "o_proj")(k)
        k = replace_whole_name("attn", "self_attn")(k)
        output[k] = v
    return output


def convert_vision_hf_to_meta(state_dict, head_dim):
    state_dict = split_hf_keys(state_dict)
    state_dict = map_vision_hf_to_meta_keys(state_dict, head_dim)

    return state_dict


def map_hf_to_meta_keys_vision_only(state_dict):
    """
    Map Hugging Face checkpoint keys to Meta checkpoint keys.
    You can use this to support other models by adding more mappings.
    See replace_keys for more details on the format of replacements.
    """
    replacements = [
        ("self_attn", "attn"),
        ("q_proj", "wq"),
        ("k_proj", "wk"),
        ("v_proj", "wv"),
        ("o_proj", "wo"),
        ("out_proj", "wo"),
        ("q_norm", "q_norm"),
        ("k_norm", "k_norm"),
        ("fc1", "c_fc"),
        ("fc2", "c_proj"),
        ("layer_norm1", "ln_1"),
        ("layer_norm2", "ln_2"),
        ("post_layernorm", "ln_post"),
        ("embeddings.patch_embedding._linear", "embeddings.patch_embedding"),
        ("embeddings.patch_embedding", "embeddings.patch_embedding._linear"),
        # ("embeddings.position_embedding.positional_embedding.weight", "embeddings.position_embedding.positional_embedding"),
        ("embeddings.position_embedding.weight", "embeddings.position_embedding.positional_embedding"),
    ]

    return replace_keys(state_dict, replacements)


def map_vision_hf_to_meta_keys_split_to_submodels(state_dict):
    vision_state_dict = dict()
    text_state_dict = dict()
    other_state_dict = dict()

    for k, v in state_dict.items():
        if k.startswith("model.vision_tower"):
            selected_dict = vision_state_dict
        elif k.startswith("model.language_model") or k.startswith("lm_head"):
            selected_dict = text_state_dict
        else:
            selected_dict = other_state_dict

        selected_dict[k] = v

    return vision_state_dict, text_state_dict, other_state_dict


def map_vision_hf_to_meta_keys(state_dict, head_dim):
    vision_state_dict, text_state_dict, other_state_dict = map_vision_hf_to_meta_keys_split_to_submodels(state_dict)

    text_state_dict = convert_hf_qkv_to_meta_format(text_state_dict, head_dim)
    text_state_dict = map_hf_to_meta_keys(text_state_dict)

    vision_state_dict = map_hf_to_meta_keys_vision_only(vision_state_dict)

    return {**vision_state_dict, **text_state_dict, **other_state_dict}


def convert_meta_to_hf(state_dict, head_dim):
    state_dict = convert_meta_qkv_to_hf_format(state_dict, head_dim)
    state_dict = map_meta_to_hf_keys(state_dict)
    return state_dict


def convert_vision_meta_to_hf(state_dict, head_dim):
    # state_dict = convert_meta_qkv_to_hf_format(state_dict, head_dim)
    state_dict = map_vision_meta_to_hf_keys(state_dict)
    return state_dict


def map_meta_to_hf_keys(state_dict):
    """
    Map Hugging Face checkpoint keys to Meta checkpoint keys.
    You can use this to support other models by adding more mappings.
    See replace_keys for more details on the format of replacements.
    """
    replacements = [
        ("layers", "model.layers"),
        ("attention_norm", "input_layernorm"),
        ("ffn_norm", "post_attention_layernorm"),
        ("attention", "self_attn"),
        ("wq", "q_proj"),
        ("wk", "k_proj"),
        ("wv", "v_proj"),
        ("wo", "o_proj"),
        ("wqkv", "qkv_proj"),
        ("feed_forward", "mlp"),
        ("w1", "gate_proj"),
        ("w2", "down_proj"),
        ("w3", "up_proj"),
        ("w1_w3", "gate_up_proj"),
        ("emb.weight", "weight"),
        ("tok_embeddings", "model.embed_tokens"),
        ("norm", "model.norm"),
        ("output", "lm_head"),
    ]
    return replace_keys(state_dict, replacements)
