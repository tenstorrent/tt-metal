# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Gemma 4 checkpoint loading utilities.

Handles the conversion of HuggingFace Gemma 4 state dict keys to the
Meta-style keys used by tt_transformers. Also handles Gemma 4 specific
keys that don't exist in the standard pipeline (per-layer input gating,
embed_tokens_per_layer, etc.).
"""

from models.tt_transformers.tt.load_checkpoints import map_hf_to_meta_keys, split_hf_keys


def separate_gemma4_keys(state_dict):
    """
    Separate Gemma 4 specific keys from the standard text model keys.

    Returns:
        text_state_dict: Standard text model keys (attention, MLP, norms)
        gemma4_state_dict: Gemma 4 specific keys (per-layer gating, etc.)
    """
    text_state_dict = {}
    gemma4_state_dict = {}

    gemma4_prefixes = [
        "per_layer_input_gate",
        "per_layer_projection",
        "post_per_layer_input_norm",
        "layer_scalar",
        "embed_tokens_per_layer",
        "per_layer_model_projection",
        "per_layer_projection_norm",
    ]

    for k, v in state_dict.items():
        is_gemma4 = False
        for prefix in gemma4_prefixes:
            if prefix in k:
                is_gemma4 = True
                break

        if is_gemma4:
            gemma4_state_dict[k] = v
        else:
            text_state_dict[k] = v

    return text_state_dict, gemma4_state_dict


def convert_gemma4_hf_to_meta(state_dict, n_heads=None, n_kv_heads=None):
    """
    Convert Gemma 4 HF state dict to Meta format.

    1. Separate Gemma 4 specific keys
    2. Apply standard HF -> Meta conversion on text keys
    3. Merge back Gemma 4 keys
    """
    text_state_dict, gemma4_state_dict = separate_gemma4_keys(state_dict)

    # Standard conversion for text keys
    text_state_dict = split_hf_keys(text_state_dict, n_heads, n_kv_heads)
    text_state_dict = map_hf_to_meta_keys(text_state_dict)

    # Merge back
    text_state_dict.update(gemma4_state_dict)

    return text_state_dict
