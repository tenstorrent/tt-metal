# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from models.tt_transformers.tt.load_checkpoints import (
    convert_hf_qkv_to_meta_format,
    map_hf_to_meta_keys,
    map_hf_to_meta_keys_vision_only,
    split_hf_keys,
)


def convert_vision_hf_to_meta(state_dict, head_dim):
    state_dict = split_hf_keys(state_dict)
    state_dict = map_vision_hf_to_meta_keys(state_dict, head_dim)

    return state_dict


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


def convert_vision_meta_to_hf(state_dict, head_dim):
    # state_dict = convert_meta_qkv_to_hf_format(state_dict, head_dim)
    state_dict = map_vision_meta_to_hf_keys(state_dict)
    return state_dict
