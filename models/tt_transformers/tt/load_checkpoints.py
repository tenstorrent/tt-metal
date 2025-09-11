# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import load_file as safetensors_load_file
from tqdm import tqdm


# TODO Update function for large models: For 1 layer tests we only want to load 1 checkpoint file, instead of all.
def load_hf_state_dict(ckpt_dir):
    # First check if index file exists
    index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        # Multi-file case: Read the index file and load all referenced safetensor files
        with open(index_path, "r") as f:
            index_data = json.load(f)

        # Retrieve the weight file names from the index JSON
        weight_map = index_data["weight_map"]
        safetensor_files = set(weight_map.values())

        # Read each safetensors file mentioned in the index
        loaded_weights = {}
        for file in safetensor_files:
            safetensor_path = os.path.join(ckpt_dir, file)
            weights = safetensors_load_file(safetensor_path)
            loaded_weights.update(weights)  # Merge weights into a single dictionary
    else:
        # Single-file case: Load the single model.safetensors file
        safetensor_path = os.path.join(ckpt_dir, "model.safetensors")
        if not os.path.exists(safetensor_path):
            raise FileNotFoundError(f"Neither model.safetensors.index.json nor model.safetensors found in {ckpt_dir}")
        loaded_weights = safetensors_load_file(safetensor_path)

    return loaded_weights


def standardize_hf_keys(state_dict):
    key_meta = "lm_head.weight"
    key_hf = "model.embed_tokens.weight"

    if not key_meta in state_dict and key_hf in state_dict:
        state_dict[key_meta] = state_dict[key_hf]
        del state_dict[key_hf]

    return state_dict


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


def convert_hf_to_meta(state_dict, head_dim):
    state_dict = split_hf_keys(state_dict)
    state_dict = convert_hf_qkv_to_meta_format(state_dict, head_dim)
    state_dict = map_hf_to_meta_keys(state_dict)
    return state_dict


def convert_vision_hf_to_meta(state_dict, head_dim):
    state_dict = split_hf_keys(state_dict)
    state_dict = map_vision_hf_to_meta_keys(state_dict, head_dim)
    return state_dict


def map_hf_to_meta_keys(loaded_weights):
    hf_to_meta = {
        # Top level mappings
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
        # Layer level mappings
        "input_layernorm.weight": "attention_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        # Attention module mappings
        "self_attn.q_proj.weight": "attention.wq.weight",
        "self_attn.k_proj.weight": "attention.wk.weight",
        "self_attn.v_proj.weight": "attention.wv.weight",
        "self_attn.o_proj.weight": "attention.wo.weight",
        "self_attn.q_proj.bias": "attention.wq.bias",
        "self_attn.k_proj.bias": "attention.wk.bias",
        "self_attn.v_proj.bias": "attention.wv.bias",
        "self_attn.q_norm.weight": "attention.q_norm.weight",
        "self_attn.k_norm.weight": "attention.k_norm.weight",
        "self_attn.o_proj.bias": "attention.wo.bias",
        # Feed forward module mappings
        "mlp.gate_proj.weight": "feed_forward.w1.weight",
        "mlp.up_proj.weight": "feed_forward.w3.weight",
        "mlp.down_proj.weight": "feed_forward.w2.weight",
        # MLP bias mappings
        "mlp.gate_proj.bias": "feed_forward.w1.bias",
        "mlp.up_proj.bias": "feed_forward.w3.bias",
        "mlp.down_proj.bias": "feed_forward.w2.bias",
        # === Additional FFN layernorms (Gemma3 specific) ===
        "pre_feedforward_layernorm.weight": "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight": "post_feedforward_layernorm.weight",
        # Direct module mappings
        "gate_proj.weight": "w1.weight",
        "down_proj.weight": "w2.weight",
        "up_proj.weight": "w3.weight",
        "q_proj.weight": "wq.weight",
        "k_proj.weight": "wk.weight",
        "v_proj.weight": "wv.weight",
        "o_proj.weight": "wo.weight",
        "q_proj.bias": "wq.bias",
        "k_proj.bias": "wk.bias",
        "v_proj.bias": "wv.bias",
        "q_norm.weight": "q_norm.weight",
        "k_norm.weight": "k_norm.weight",
        "o_proj.bias": "wo.bias",
        # Direct MLP bias mappings
        "gate_proj.bias": "w1.bias",
        "up_proj.bias": "w3.bias",
        "down_proj.bias": "w2.bias",
        "weight": "emb.weight",  # For host embeddings
        # Full path layer mappings
        "model.layers.{layer}.input_layernorm.weight": "layers.{layer}.attention_norm.weight",
        "model.layers.{layer}.post_attention_layernorm.weight": "layers.{layer}.ffn_norm.weight",
        "model.layers.{layer}.self_attn.q_proj.weight": "layers.{layer}.attention.wq.weight",
        "model.layers.{layer}.self_attn.k_proj.weight": "layers.{layer}.attention.wk.weight",
        "model.layers.{layer}.self_attn.v_proj.weight": "layers.{layer}.attention.wv.weight",
        "model.layers.{layer}.self_attn.o_proj.weight": "layers.{layer}.attention.wo.weight",
        "model.layers.{layer}.self_attn.q_proj.bias": "layers.{layer}.attention.wq.bias",
        "model.layers.{layer}.self_attn.k_proj.bias": "layers.{layer}.attention.wk.bias",
        "model.layers.{layer}.self_attn.v_proj.bias": "layers.{layer}.attention.wv.bias",
        "model.layers.{layer}.self_attn.q_norm.weight": "layers.{layer}.attention.q_norm.weight",
        "model.layers.{layer}.self_attn.k_norm.weight": "layers.{layer}.attention.k_norm.weight",
        "model.layers.{layer}.self_attn.o_proj.bias": "layers.{layer}.attention.wo.bias",
        "model.layers.{layer}.mlp.gate_proj.weight": "layers.{layer}.feed_forward.w1.weight",
        "model.layers.{layer}.mlp.up_proj.weight": "layers.{layer}.feed_forward.w3.weight",
        "model.layers.{layer}.mlp.down_proj.weight": "layers.{layer}.feed_forward.w2.weight",
        # Full path MLP bias mappings
        "model.layers.{layer}.mlp.gate_proj.bias": "layers.{layer}.feed_forward.w1.bias",
        "model.layers.{layer}.mlp.up_proj.bias": "layers.{layer}.feed_forward.w3.bias",
        "model.layers.{layer}.mlp.down_proj.bias": "layers.{layer}.feed_forward.w2.bias",
        "model.layers.{layer}.pre_feedforward_layernorm.weight": "layers.{layer}.pre_feedforward_layernorm.weight",
        "model.layers.{layer}.post_feedforward_layernorm.weight": "layers.{layer}.post_feedforward_layernorm.weight",
    }

    meta_state_dict = {}
    for key, tensor in loaded_weights.items():
        # Remove known prefix if present
        prefix = next((p for p in _get_known_prefixes_mapping().keys() if key.startswith(p)), "")
        key = key.replace(prefix, _get_known_prefixes_mapping().get(prefix, ""), 1)

        new_key = key
        if key in hf_to_meta:
            # Direct match for top-level keys
            new_key = hf_to_meta[key]
        elif key.startswith("model.layers."):
            # Extract layer number and form a template key
            parts = key.split(".")
            layer_num = parts[2]  # e.g. "0" in "model.layers.0.input_layernorm.weight"
            template_key = "model.layers.{layer}." + ".".join(parts[3:])
            if template_key in hf_to_meta:
                new_key = hf_to_meta[template_key].format(layer=layer_num)
            else:
                new_key = key[len("model.") :]  # Remove "model." prefix

        meta_state_dict[new_key] = tensor

    return meta_state_dict


def map_vision_meta_to_hf_keys(loaded_weights):
    language_weights = {
        key[len("language_model.") :]: tensor
        for key, tensor in loaded_weights.items()
        if key.startswith("language_model.")
    }
    mapped_language_weights = map_meta_to_hf_keys(language_weights, language_prefix="language_model.")
    other_weights = {key: tensor for key, tensor in loaded_weights.items() if not key.startswith("language_model.")}
    hf_state_dict = {**mapped_language_weights}
    loaded_weights = {**other_weights}
    meta_to_hf_mappings = {
        # vision MLP
        "c_fc.weight": "fc1.weight",
        "c_fc.bias": "fc1.bias",
        "c_proj.weight": "fc2.weight",
        "c_proj.bias": "fc2.bias",
        # vision attention
        # "wq.weight": "q_proj.weight",
        # "wk.weight": "k_proj.weight",
        # "wv.weight": "v_proj.weight",
        # "wo.weight": "out_proj.weight",
        # "wq.bias": "q_proj.bias",
        # "wk.bias": "k_proj.bias",
        # "wv.bias": "v_proj.bias",
        # "wo.bias": "out_proj.bias",
        # vision encoder block
        "attn.wq.weight": "self_attn.q_proj.weight",
        "attn.wk.weight": "self_attn.k_proj.weight",
        "attn.wv.weight": "self_attn.v_proj.weight",
        "attn.wo.weight": "self_attn.out_proj.weight",
        "attn.wq.bias": "self_attn.q_proj.bias",
        "attn.wk.bias": "self_attn.k_proj.bias",
        "attn.wv.bias": "self_attn.v_proj.bias",
        "attn.wo.bias": "self_attn.out_proj.bias",
        "ln_1.weight": "layer_norm1.weight",
        "ln_1.bias": "layer_norm1.bias",
        "ln_2.weight": "layer_norm2.weight",
        "ln_2.bias": "layer_norm2.bias",
        "mlp.c_fc.weight": "mlp.fc1.weight",
        "mlp.c_fc.bias": "mlp.fc1.bias",
        "mlp.c_proj.weight": "mlp.fc2.weight",
        "mlp.c_proj.bias": "mlp.fc2.bias",
        # vision encoder
        "layers.{layer}.attn.wq.weight": "layers.{layer}.self_attn.q_proj.weight",
        "layers.{layer}.attn.wk.weight": "layers.{layer}.self_attn.k_proj.weight",
        "layers.{layer}.attn.wv.weight": "layers.{layer}.self_attn.v_proj.weight",
        "layers.{layer}.attn.wo.weight": "layers.{layer}.self_attn.out_proj.weight",
        "layers.{layer}.attn.wq.bias": "layers.{layer}.self_attn.q_proj.bias",
        "layers.{layer}.attn.wk.bias": "layers.{layer}.self_attn.k_proj.bias",
        "layers.{layer}.attn.wv.bias": "layers.{layer}.self_attn.v_proj.bias",
        "layers.{layer}.attn.wo.bias": "layers.{layer}.self_attn.out_proj.bias",
        "layers.{layer}.ln_1.weight": "layers.{layer}.layer_norm1.weight",
        "layers.{layer}.ln_1.bias": "layers.{layer}.layer_norm1.bias",
        "layers.{layer}.ln_2.weight": "layers.{layer}.layer_norm2.weight",
        "layers.{layer}.ln_2.bias": "layers.{layer}.layer_norm2.bias",
        "layers.{layer}.mlp.c_fc.weight": "layers.{layer}.mlp.fc1.weight",
        "layers.{layer}.mlp.c_fc.bias": "layers.{layer}.mlp.fc1.bias",
        "layers.{layer}.mlp.c_proj.weight": "layers.{layer}.mlp.fc2.weight",
        "layers.{layer}.mlp.c_proj.bias": "layers.{layer}.mlp.fc2.bias",
        # vision transformer
        "encoder.layers.{layer}.attn.wq.weight": "encoder.layers.{layer}.self_attn.q_proj.weight",
        "encoder.layers.{layer}.attn.wk.weight": "encoder.layers.{layer}.self_attn.k_proj.weight",
        "encoder.layers.{layer}.attn.wv.weight": "encoder.layers.{layer}.self_attn.v_proj.weight",
        "encoder.layers.{layer}.attn.wo.weight": "encoder.layers.{layer}.self_attn.out_proj.weight",
        "encoder.layers.{layer}.attn.wq.bias": "encoder.layers.{layer}.self_attn.q_proj.bias",
        "encoder.layers.{layer}.attn.wk.bias": "encoder.layers.{layer}.self_attn.k_proj.bias",
        "encoder.layers.{layer}.attn.wv.bias": "encoder.layers.{layer}.self_attn.v_proj.bias",
        "encoder.layers.{layer}.attn.wo.bias": "encoder.layers.{layer}.self_attn.out_proj.bias",
        "ln_post.weight": "post_layernorm.weight",
        "ln_post.bias": "post_layernorm.bias",
        # Top level
        "_linear.weight": "weight",  # patch_embedding
        "_linear.bias": "bias",  # patch_embedding
        "positional_embedding": "weight",  # pos_emb
        "visual.embeddings.patch_embedding._linear.weight": "visual.embeddings.patch_embedding.weight",
        "visual.embeddings.patch_embedding._linear.bias": "visual.embeddings.patch_embedding._linear.bias",
        "visual.embeddings.position_embedding.positional_embedding": "visual.embeddings.position_embedding.weight",
        "visual.encoder.layers.{layer}.attn.wq.weight": "visual.encoder.layers.{layer}.self_attn.q_proj.weight",
        "visual.encoder.layers.{layer}.attn.wk.weight": "visual.encoder.layers.{layer}.self_attn.k_proj.weight",
        "visual.encoder.layers.{layer}.attn.wv.weight": "visual.encoder.layers.{layer}.self_attn.v_proj.weight",
        "visual.encoder.layers.{layer}.attn.wo.weight": "visual.encoder.layers.{layer}.self_attn.out_proj.weight",
        "visual.encoder.layers.{layer}.attn.wq.bias": "visual.encoder.layers.{layer}.self_attn.q_proj.bias",
        "visual.encoder.layers.{layer}.attn.wk.bias": "visual.encoder.layers.{layer}.self_attn.k_proj.bias",
        "visual.encoder.layers.{layer}.attn.wv.bias": "visual.encoder.layers.{layer}.self_attn.v_proj.bias",
        "visual.encoder.layers.{layer}.attn.wo.bias": "visual.encoder.layers.{layer}.self_attn.out_proj.bias",
        "visual.encoder.layers.{layer}.ln_1.weight": "visual.encoder.layers.{layer}.layer_norm1.weight",
        "visual.encoder.layers.{layer}.ln_1.bias": "visual.encoder.layers.{layer}.layer_norm1.bias",
        "visual.encoder.layers.{layer}.ln_2.weight": "visual.encoder.layers.{layer}.layer_norm2.weight",
        "visual.encoder.layers.{layer}.ln_2.bias": "visual.encoder.layers.{layer}.layer_norm2.bias",
        "visual.encoder.layers.{layer}.mlp.c_fc.weight": "visual.encoder.layers.{layer}.mlp.fc1.weight",
        "visual.encoder.layers.{layer}.mlp.c_fc.bias": "visual.encoder.layers.{layer}.mlp.fc1.bias",
        "visual.encoder.layers.{layer}.mlp.c_proj.weight": "visual.encoder.layers.{layer}.mlp.fc2.weight",
        "visual.encoder.layers.{layer}.mlp.c_proj.bias": "visual.encoder.layers.{layer}.mlp.fc2.bias",
        "visual.ln_post.weight": "visual.post_layernorm.weight",
        "visual.ln_post.bias": "visual.post_layernorm.bias",
    }

    for key, tensor in loaded_weights.items():
        # Handle full model paths with layer numbers
        if "model.vision_tower.vision_model.encoder.layers." in key:
            parts = key.split(".")
            layer_num = parts[5]
            remainder = ".".join(parts[6:])
            if remainder in meta_to_hf_mappings:
                new_key = f"model.vision_tower.vision_model.encoder.layers.{layer_num}.{meta_to_hf_mappings[remainder]}"
                hf_state_dict[new_key] = tensor
            continue

        # Handle full vision encoder paths with layer numbers
        if "layers." in key:
            parts = key.split(".")
            layer_num = parts[1]  # e.g. "0" in "model.layers.0.input_layernorm.weight"
            template_key = "layers.{layer}." + ".".join(parts[2:])
            if template_key in meta_to_hf_mappings:
                hf_state_dict[meta_to_hf_mappings[template_key].format(layer=layer_num)] = tensor
                continue

        # Try exact matches first
        if key in meta_to_hf_mappings:
            hf_state_dict[meta_to_hf_mappings[key]] = tensor
            continue

        # For submodule state dicts, try matching the end of the key
        matched = False
        for meta_pattern, hf_pattern in meta_to_hf_mappings.items():
            if key.endswith("." + meta_pattern):
                # Replace only the matching part at the end
                prefix = key[: -len(meta_pattern)]
                new_key = prefix + hf_pattern
                hf_state_dict[new_key] = tensor
                matched = True
                break

        # If no mapping found, keep the original key
        if not matched:
            hf_state_dict[key] = tensor

    return hf_state_dict


def map_vision_hf_to_meta_keys(loaded_weights, head_dim):
    hf_to_meta = {
        # vision MLP
        "fc1.weight": "c_fc.weight",
        "fc1.bias": "c_fc.bias",
        "fc2.weight": "c_proj.weight",
        "fc2.bias": "c_proj.bias",
        # vision attention
        "q_proj.weight": "wq.weight",
        "k_proj.weight": "wk.weight",
        "v_proj.weight": "wv.weight",
        "out_proj.weight": "wo.weight",
        "q_proj.bias": "wq.bias",
        "k_proj.bias": "wk.bias",
        "v_proj.bias": "wv.bias",
        "out_proj.bias": "wo.bias",
        # vision encoder
        "self_attn.q_proj.weight": "attn.wq.weight",
        "self_attn.k_proj.weight": "attn.wk.weight",
        "self_attn.v_proj.weight": "attn.wv.weight",
        "self_attn.out_proj.weight": "attn.wo.weight",
        "self_attn.q_proj.bias": "attn.wq.bias",
        "self_attn.k_proj.bias": "attn.wk.bias",
        "self_attn.v_proj.bias": "attn.wv.bias",
        "self_attn.out_proj.bias": "attn.wo.bias",
        "layer_norm1.weight": "ln_1.weight",
        "layer_norm1.bias": "ln_1.bias",
        "layer_norm2.weight": "ln_2.weight",
        "layer_norm2.bias": "ln_2.bias",
        "mlp.fc1.weight": "mlp.c_fc.weight",
        "mlp.fc1.bias": "mlp.c_fc.bias",
        "mlp.fc2.weight": "mlp.c_proj.weight",
        "mlp.fc2.bias": "mlp.c_proj.bias",
        # Top level
        # vision transformer
        "encoder.layers.{layer}.self_attn.q_proj.weight": "encoder.layers.{layer}.attn.wq.weight",
        "encoder.layers.{layer}.self_attn.k_proj.weight": "encoder.layers.{layer}.attn.wk.weight",
        "encoder.layers.{layer}.self_attn.v_proj.weight": "encoder.layers.{layer}.attn.wv.weight",
        "encoder.layers.{layer}.self_attn.out_proj.weight": "encoder.layers.{layer}.attn.wo.weight",
        "encoder.layers.{layer}.self_attn.q_proj.bias": "encoder.layers.{layer}.attn.wq.bias",
        "encoder.layers.{layer}.self_attn.k_proj.bias": "encoder.layers.{layer}.attn.wk.bias",
        "encoder.layers.{layer}.self_attn.v_proj.bias": "encoder.layers.{layer}.attn.wv.bias",
        "encoder.layers.{layer}.self_attn.out_proj.bias": "encoder.layers.{layer}.attn.wo.bias",
        "post_layernorm.weight": "ln_post.weight",
        "post_layernorm.bias": "ln_post.bias",
        "weight": "_linear.weight",
        "bias": "_linear.bias",
        "weight": "positional_embedding",  # pos_emb
        "visual.embeddings.patch_embedding.weight": "visual.embeddings.patch_embedding._linear.weight",
        "visual.embeddings.patch_embedding.bias": "visual.embeddings.patch_embedding._linear.bias",
        "visual.embeddings.position_embedding.weight": "visual.embeddings.position_embedding.positional_embedding",
        "visual.encoder.layers.{layer}.self_attn.q_proj.weight": "visual.encoder.layers.{layer}.attn.wq.weight",
        "visual.encoder.layers.{layer}.self_attn.k_proj.weight": "visual.encoder.layers.{layer}.attn.wk.weight",
        "visual.encoder.layers.{layer}.self_attn.v_proj.weight": "visual.encoder.layers.{layer}.attn.wv.weight",
        "visual.encoder.layers.{layer}.self_attn.out_proj.weight": "visual.encoder.layers.{layer}.attn.wo.weight",
        "visual.encoder.layers.{layer}.self_attn.q_proj.bias": "visual.encoder.layers.{layer}.attn.wq.bias",
        "visual.encoder.layers.{layer}.self_attn.k_proj.bias": "visual.encoder.layers.{layer}.attn.wk.bias",
        "visual.encoder.layers.{layer}.self_attn.v_proj.bias": "visual.encoder.layers.{layer}.attn.wv.bias",
        "visual.encoder.layers.{layer}.self_attn.out_proj.bias": "visual.encoder.layers.{layer}.attn.wo.bias",
        "visual.encoder.layers.{layer}.layer_norm1.weight": "visual.encoder.layers.{layer}.ln_1.weight",
        "visual.encoder.layers.{layer}.layer_norm1.bias": "visual.encoder.layers.{layer}.ln_1.bias",
        "visual.encoder.layers.{layer}.layer_norm2.weight": "visual.encoder.layers.{layer}.ln_2.weight",
        "visual.encoder.layers.{layer}.layer_norm2.bias": "visual.encoder.layers.{layer}.ln_2.bias",
        "visual.encoder.layers.{layer}.mlp.fc1.weight": "visual.encoder.layers.{layer}.mlp.c_fc.weight",
        "visual.encoder.layers.{layer}.mlp.fc1.bias": "visual.encoder.layers.{layer}.mlp.c_fc.bias",
        "visual.encoder.layers.{layer}.mlp.fc2.weight": "visual.encoder.layers.{layer}.mlp.c_proj.weight",
        "visual.encoder.layers.{layer}.mlp.fc2.bias": "visual.encoder.layers.{layer}.mlp.c_proj.bias",
        "visual.post_layernorm.weight": "visual.ln_post.weight",
        "visual.post_layernorm.bias": "visual.ln_post.bias",
    }

    remapped = {}
    for key, tensor in loaded_weights.items():
        if key in hf_to_meta:
            remapped[hf_to_meta[key]] = tensor
        elif "visual.encoder.layers." in key:
            parts = key.split(".")
            layer_num = parts[3]  # e.g. "0" in "visual.encoder.layers.0.layer_norm1.weight"
            template_key = "visual.encoder.layers.{layer}." + ".".join(parts[4:])
            if template_key in hf_to_meta:
                remapped[hf_to_meta[template_key].format(layer=layer_num)] = tensor
        else:
            remapped[key] = tensor

    # Remove language_model keys
    non_text_weights = {k: v for k, v in remapped.items() if not k.startswith("model.")}
    text_weights = {k: v for k, v in loaded_weights.items() if k.startswith("model.") or k.startswith("lm_head.")}
    text_weights = convert_hf_qkv_to_meta_format(text_weights, head_dim)
    remapped_text = map_hf_to_meta_keys(text_weights)
    return {**non_text_weights, **remapped_text}


def load_meta_state_dict(ckpt_dir, n_layers=None, start_layer_idx=0):
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
    is_chunked = any(ckpt.stem.startswith("layers_") for ckpt in checkpoints)
    if is_chunked:
        checkpoints = [ckpt_name for ckpt_name in checkpoints if ckpt_name.stem.startswith("layers_")]
        checkpoint = load_chunked_checkpoints(checkpoints, n_layers, start_layer_idx)
    else:
        checkpoint = load_sharded_checkpoints(checkpoints, n_layers)

    return checkpoint


def load_chunked_checkpoints(checkpoints, n_layers, start_layer_idx):
    checkpoint = {}

    (f"Loading {len(checkpoints)} chunked checkpoint files")
    for ckpt in tqdm(checkpoints):
        if n_layers:
            # Layer range is in the file name, like layers_start-end.pth
            layer_range = ckpt.stem.split("_")[1]
            start_layer, end_layer = map(int, layer_range.split("-"))
            if start_layer > n_layers + start_layer_idx:
                continue
            if end_layer < start_layer_idx:
                continue

        loaded_ckpt = torch.load(ckpt, map_location="cpu")
        checkpoint.update(loaded_ckpt)
    return checkpoint


def is_param_replicated_across_shards(key: str) -> bool:
    """
    Return `True` if the parameter is replicated (i.e., not sharded)
    across checkpoint files and should not be concatenated.
    """
    if key.startswith("vision_model."):
        return any(keyword in key for keyword in ("ln", "gate", "embed", "c_proj.bias"))
    else:
        # for Meta checkpoint keys, key either starts with "text_model." or contains no such prefix; both cases are handled here
        return any(keyword in key for keyword in ("norm", "gate"))


def load_sharded_checkpoints(checkpoints, n_layers):
    checkpoint = {}
    logger.info(f"Loading {len(checkpoints)} sharded checkpoint files")
    for ckpt in tqdm(checkpoints):
        loaded_ckpt = torch.load(ckpt, map_location="cpu")
        for key, value in loaded_ckpt.items():
            if "layers." in key:
                layer_num = int(key.split("layers.")[1].split(".")[0])
                if n_layers and layer_num >= n_layers:
                    continue
            if key in checkpoint:
                checkpoint[key] += [value]
            else:
                checkpoint[key] = [value]
        del loaded_ckpt

    # concat checkpoint values
    for key, value in checkpoint.items():
        if len(value) == 1 or is_param_replicated_across_shards(key):
            checkpoint[key] = value[0]
        else:
            if key.endswith("tok_embeddings.weight") or key.endswith("output.weight"):
                assert value[0].shape[1] == 8192  # FIXME: do we need this hardcoded shape?
                # Concatenate along dimension 0 for llama3 token embeddings weight and lm head
                checkpoint[key] = torch.cat(value, dim=0)
            else:
                # cat_dim is index of the smallest dimension in value[0].shape
                cat_dim = torch.argmin(torch.tensor(value[0].shape))
                checkpoint[key] = torch.cat(value, dim=cat_dim)

    return checkpoint


def split_hf_keys(loaded_weights):
    converted_weights = {}
    for key, tensor in loaded_weights.items():
        if "qkv_proj" in key:
            # split Q, K and V
            q_key = key.replace("qkv_proj", "q_proj")
            k_key = key.replace("qkv_proj", "k_proj")
            v_key = key.replace("qkv_proj", "v_proj")
            q_tensor, k_tensor, v_tensor = torch.split(tensor, tensor.shape[0] // 3, dim=0)
            converted_weights[q_key] = q_tensor
            converted_weights[k_key] = k_tensor
            converted_weights[v_key] = v_tensor
        elif "gate_up_proj" in key:
            # Split Gate and Up
            gate_key = key.replace("gate_up_proj", "gate_proj")
            up_key = key.replace("gate_up_proj", "up_proj")
            gate_tensor, up_tensor = torch.split(tensor, tensor.shape[0] // 2, dim=0)
            converted_weights[gate_key] = gate_tensor
            converted_weights[up_key] = up_tensor
        else:
            # Keep all other weights unchanged
            converted_weights[key] = tensor
    return converted_weights


def convert_hf_qkv_to_meta_format(loaded_weights, head_dim):
    """Convert HuggingFace QKV weights to Meta format for RoPE compatibility."""
    converted_weights = {}
    for key, tensor in loaded_weights.items():
        if "q_proj.weight" in key or "k_proj.weight" in key:
            # For weights: n_heads = tensor.shape[0] // head_dim
            n_heads = tensor.shape[0] // head_dim
            converted_weights[key] = reverse_permute(tensor, n_heads, tensor.shape[0], tensor.shape[1])
        elif "q_proj.bias" in key or "k_proj.bias" in key:
            # For biases: n_heads = tensor.shape[0] // head_dim
            n_heads = tensor.shape[0] // head_dim
            converted_weights[key] = reverse_permute(tensor, n_heads, tensor.shape[0], 1).squeeze(-1)
        elif "q_norm.weight" in key or "k_norm.weight" in key:
            converted_weights[key] = reverse_permute_1d(tensor)
        else:
            # Keep all other weights unchanged
            converted_weights[key] = tensor
    return converted_weights


def fuse_mlp_meta(state_dict):
    key_map = {"w_gate": "w1.weight", "w_up": "w3.weight", "w_gate_up_proj": "w1_w3.weight"}

    wgate_list = sorted(list(filter(lambda x: key_map["w_gate"] in x, state_dict.keys())))
    wproj_list = sorted(list(filter(lambda x: key_map["w_up"] in x, state_dict.keys())))

    for wgate_key, wproj_key in zip(wgate_list, wproj_list):
        wgate = state_dict[wgate_key]
        wproj = state_dict[wproj_key]

        prefix_gate = wgate_key[: -len(key_map["w_gate"])]

        fused_gate_up_proj = torch.vstack((wgate, wproj))
        state_dict[f"{prefix_gate}{key_map['w_gate_up_proj']}"] = fused_gate_up_proj

        del state_dict[wgate_key], state_dict[wproj_key]

    return state_dict


def fuse_qkv_meta(state_dict):
    # Weight keys list
    wq_list = sorted(list(filter(lambda x: "wq.weight" in x, state_dict.keys())))
    wk_list = sorted(list(filter(lambda x: "wk.weight" in x, state_dict.keys())))
    wv_list = sorted(list(filter(lambda x: "wv.weight" in x, state_dict.keys())))
    # Bias keys list
    wq_bias_list = sorted(list(filter(lambda x: "wq.bias" in x, state_dict.keys())))
    wk_bias_list = sorted(list(filter(lambda x: "wk.bias" in x, state_dict.keys())))
    wv_bias_list = sorted(list(filter(lambda x: "wv.bias" in x, state_dict.keys())))

    for wq_key, wk_key, wv_key in zip(wq_list, wk_list, wv_list):
        wq = state_dict[wq_key]
        wk = state_dict[wk_key]
        wv = state_dict[wv_key]

        prefix = wq_key[: -len("wq.weight")]
        fused_qkv_weights = torch.vstack((wq, wk, wv))
        state_dict[f"{prefix}wqkv.weight"] = fused_qkv_weights

        del state_dict[wq_key], state_dict[wk_key], state_dict[wv_key]

    # Checking for bias
    if len(wq_bias_list) > 0:
        for wq_bias_key, wk_bias_key, wv_bias_key in zip(wq_bias_list, wk_bias_list, wv_bias_list):
            wq_bias = state_dict[wq_bias_key]
            wk_bias = state_dict[wk_bias_key]
            wv_bias = state_dict[wv_bias_key]

            prefix = wq_bias_key[: -len("wq.bias")]
            fused_qkv_bias = torch.vstack((wq_bias, wk_bias, wv_bias))
            state_dict[f"{prefix}wqkv.bias"] = fused_qkv_bias

            del state_dict[wq_bias_key], state_dict[wk_bias_key], state_dict[wv_bias_key]

    return state_dict


def convert_meta_to_hf(state_dict, head_dim, fuse_qkv=False, fuse_mlp=False):
    state_dict = convert_meta_qkv_to_hf_format(state_dict, head_dim)
    if fuse_qkv:
        state_dict = fuse_qkv_meta(state_dict)
    if fuse_mlp:
        state_dict = fuse_mlp_meta(state_dict)
    state_dict = map_meta_to_hf_keys(state_dict)
    return state_dict


def replace_keys(state_dict, replacements):
    """
    Replacements are in the form (pattern, replacement).
    Patterns can use ^ to match the start of the string but are otherwise
    matched as whole words. These are not regular expressions, e.g. . is not
    a special character.
    """
    for pattern, replacement in replacements:
        pre = r"^" if pattern.startswith("^") else r"(?=^|\b)"
        post = r"\." if pattern.endswith(".") else r"(?=\b|$)"
        pattern = pattern[1:] if pattern.startswith("^") else pattern
        pattern = pattern[:-1] if pattern.endswith(".") else pattern
        pattern = pre + pattern + post
        state_dict = {re.sub(pattern, replacement, k): v for k, v in state_dict.items()}
    return state_dict


def map_hf_to_meta_keys(loaded_weights):
    """
    Map Hugging Face checkpoint keys to Meta checkpoint keys.
    You can use this to support other models by adding more mappings.
    See replace_keys for more details on the format of replacements.
    """
    replacements = [
        ("^emb.weight", "weight"),
        ("model.language_model.", ""),
        ("model.", ""),
        ("embed_tokens", "tok_embeddings"),
        ("lm_head", "output"),
        ("input_layernorm", "attention_norm"),
        ("post_attention_layernorm", "ffn_norm"),
        ("self_attn", "attention"),
        ("mlp", "feed_forward"),
        ("gate_proj", "w1"),
        ("down_proj", "w2"),
        ("up_proj", "w3"),
        ("q_proj", "wq"),
        ("k_proj", "wk"),
        ("v_proj", "wv"),
        ("o_proj", "wo"),
        ("q_norm", "q_norm"),
        ("k_norm", "k_norm"),
    ]
    return replace_keys(loaded_weights, replacements)


def convert_vision_meta_to_hf(state_dict, head_dim):
    # state_dict = convert_meta_qkv_to_hf_format(state_dict, head_dim)
    state_dict = map_vision_meta_to_hf_keys(state_dict)
    return state_dict


def map_meta_to_hf_keys(loaded_weights, language_prefix=""):
    # Define mappings at each level of the hierarchy
    meta_to_hf_mappings = {
        # Top level
        "tok_embeddings.weight": "model.embed_tokens.weight",
        "norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
        # Layer level
        "attention_norm.weight": "input_layernorm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight": "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight": "post_feedforward_layernorm.weight",
        # Attention module
        "attention.wq.weight": "self_attn.q_proj.weight",
        "attention.wk.weight": "self_attn.k_proj.weight",
        "attention.wv.weight": "self_attn.v_proj.weight",
        "attention.wo.weight": "self_attn.o_proj.weight",
        "attention.wq.bias": "self_attn.q_proj.bias",
        "attention.wk.bias": "self_attn.k_proj.bias",
        "attention.wv.bias": "self_attn.v_proj.bias",
        "attention.q_norm.weight": "self_attn.q_norm.weight",
        "attention.k_norm.weight": "self_attn.k_norm.weight",
        "attention.wo.bias": "self_attn.o_proj.bias",
        "attention.wqkv.weight": "self_attn.qkv_proj.weight",
        "attention.wqkv.bias": "self_attn.qkv_proj.bias",
        # Feed forward module
        "feed_forward.w1.weight": "mlp.gate_proj.weight",
        "feed_forward.w3.weight": "mlp.up_proj.weight",
        "feed_forward.w2.weight": "mlp.down_proj.weight",
        # Feed forward bias mappings
        "feed_forward.w1.bias": "mlp.gate_proj.bias",
        "feed_forward.w3.bias": "mlp.up_proj.bias",
        "feed_forward.w2.bias": "mlp.down_proj.bias",
        "feed_forward.w1_w3.weight": "mlp.gate_up_proj.weight",
        # Direct mappings for when we get just the final components
        "w1_w3.weight": "gate_up_proj.weight",
        "w1.weight": "gate_proj.weight",
        "w2.weight": "down_proj.weight",
        "w3.weight": "up_proj.weight",
        "wq.weight": "q_proj.weight",
        "wk.weight": "k_proj.weight",
        "wv.weight": "v_proj.weight",
        "wo.weight": "o_proj.weight",
        "wq.bias": "q_proj.bias",
        "wk.bias": "k_proj.bias",
        "wv.bias": "v_proj.bias",
        "wo.bias": "o_proj.bias",
        # Direct MLP bias mappings
        "w1.bias": "gate_proj.bias",
        "w3.bias": "up_proj.bias",
        "w2.bias": "down_proj.bias",
        # Fused qkv mapping
        "wqkv.weight": "qkv_proj.weight",
        "wqkv.bias": "qkv_proj.bias",
        # Host embeddings
        "emb.weight": "weight",
    }

    hf_state_dict = {}
    for key, tensor in loaded_weights.items():
        # Handle full model paths with layer numbers
        if "layers." in key:
            parts = key.split(".")
            layer_num = parts[1]
            remainder = ".".join(parts[2:])
            if remainder in meta_to_hf_mappings:
                new_key = f"model.layers.{layer_num}.{meta_to_hf_mappings[remainder]}"
                hf_state_dict[new_key] = tensor
            continue

        # Try exact matches first
        if key in meta_to_hf_mappings:
            hf_state_dict[meta_to_hf_mappings[key]] = tensor
            continue

        # For submodule state dicts, try matching the end of the key
        matched = False
        for meta_pattern, hf_pattern in meta_to_hf_mappings.items():
            if key.endswith("." + meta_pattern):
                # Replace only the matching part at the end
                prefix = key[: -len(meta_pattern)]
                new_key = prefix + hf_pattern
                hf_state_dict[new_key] = tensor
                matched = True
                break

        # If no mapping found, keep the original key
        if not matched:
            hf_state_dict[key] = tensor

    return hf_state_dict


def convert_meta_qkv_to_hf_format(loaded_weights, head_dim):
    """Convert Meta QKV weights back to HuggingFace format."""
    converted_weights = {}
    for key, tensor in loaded_weights.items():
        if "wq.weight" in key or "wk.weight" in key:
            # For weights: n_heads = tensor.shape[0] // head_dim
            n_heads = tensor.shape[0] // head_dim
            converted_weights[key] = permute(tensor, n_heads, tensor.shape[0], tensor.shape[1])
        elif "wq.bias" in key or "wk.bias" in key:
            # For biases: n_heads = tensor.shape[0] // head_dim
            n_heads = tensor.shape[0] // head_dim
            converted_weights[key] = permute(tensor.unsqueeze(-1), n_heads, tensor.shape[0], 1).squeeze(-1)
        elif "q_norm.weight" in key or "k_norm.weight" in key:
            converted_weights[key] = permute_1d(tensor)
        else:
            # Keep all other weights unchanged
            converted_weights[key] = tensor
    return converted_weights


def reverse_permute(tensor, n_heads, dim1, dim2):
    return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def permute(tensor, n_heads, dim1, dim2):
    return tensor.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def reverse_permute_1d(tensor):
    """Convert the last dim of a tensor from separate real and imaginary parts (r1, r2, i1, i2, ...) to interleaved rope format (r1, i1, r2, i2, ...)"""
    shape = tensor.shape
    dim = shape[-1]
    assert dim % 2 == 0, "Last dimension must be even"
    reals = tensor[..., : dim // 2]
    imags = tensor[..., dim // 2 :]
    interleaved = torch.stack((reals, imags), dim=-1).flatten(start_dim=len(shape) - 1)
    return interleaved


def permute_1d(tensor):
    """Convert the last dim of a tensor from interleaved rope format (r1, i1, r2, i2, ...) to separate real and imaginary parts (r1, r2, i1, i2, ...)"""
    shape = tensor.shape
    dim = shape[-1]
    assert dim % 2 == 0, "Last dimension must be even"
    reshaped = tensor.reshape(*shape[:-1], dim // 2, 2)
    reals = reshaped[..., 0]
    imags = reshaped[..., 1]
    return torch.cat((reals, imags), dim=-1)


def convert_rope_style_hf_to_meta(cos_hf: torch.Tensor, sin_hf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Converts RoPE cos/sin tensors from Hugging Face style (half-dim duplicated)
    to Meta style (pairwise duplicated / odd-even interleaved).

    Args:
        cos_hf: Cosine tensor in HF format [..., seq_len, head_dim]
                (e.g., [c0, c1, ..., c_{d/2-1}, c0, c1, ..., c_{d/2-1}])
        sin_hf: Sine tensor in HF format [..., seq_len, head_dim]
                (e.g., [s0, s1, ..., s_{d/2-1}, s0, s1, ..., s_{d/2-1}])

    Returns:
        A tuple containing (cos_meta, sin_meta) in Meta format [..., seq_len, head_dim]
        (e.g., [c0, c0, c1, c1, ..., c_{d/2-1}, c_{d/2-1}],
         [s0, s0, s1, s1, ..., s_{d/2-1}, s_{d/2-1}])
    """
    # Input validation (optional but good practice)
    if cos_hf.shape != sin_hf.shape:
        raise ValueError("cos_hf and sin_hf must have the same shape.")
    if len(cos_hf.shape) < 2:
        raise ValueError("Input tensors must have at least 2 dimensions (seq_len, head_dim).")

    head_dim = cos_hf.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(f"Head dimension ({head_dim}) must be even.")

    half_head_dim = head_dim // 2

    # Select the first half (contains the unique frequencies)
    cos_unique = cos_hf[..., :half_head_dim]
    sin_unique = sin_hf[..., :half_head_dim]

    # Repeat each unique frequency pairwise
    cos_meta = torch.repeat_interleave(cos_unique, repeats=2, dim=-1)
    sin_meta = torch.repeat_interleave(sin_unique, repeats=2, dim=-1)

    return cos_meta, sin_meta
