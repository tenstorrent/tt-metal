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
        elif "model.vision_model." in k:
            new_state_dict[k.replace("model.vision_model.", "vision_model.")] = state_dict[k]
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


def convert_hf_to_meta(state_dict, head_dim, n_heads=None, n_kv_heads=None):
    state_dict = split_hf_keys(state_dict, n_heads, n_kv_heads)
    state_dict = convert_hf_qkv_to_meta_format(state_dict, head_dim)
    state_dict = map_hf_to_meta_keys(state_dict)
    return state_dict


def convert_vision_hf_to_meta(state_dict, head_dim):
    state_dict = split_hf_keys(state_dict)
    state_dict = map_vision_hf_to_meta_keys(state_dict, head_dim)
    return state_dict


def convert_hf_to_meta_mllama(state_dict, head_dim, config):
    state_dict = split_hf_keys(state_dict)
    state_dict = convert_hf_qkv_to_meta_format(state_dict, head_dim)
    state_dict = map_hf_to_meta_keys_mllama(state_dict, config)
    state_dict = convert_pos_embeddings(state_dict)
    state_dict = flatten_conv_linear(state_dict)
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
        ("embeddings.position_embedding.weight", "embeddings.position_embedding.positional_embedding"),
    ]

    return replace_keys(state_dict, replacements)


def map_vision_hf_to_meta_keys_split_to_submodels(state_dict):
    vision_state_dict = dict()
    text_state_dict = dict()
    other_state_dict = dict()

    for k, v in state_dict.items():
        if k.startswith("visual"):
            selected_dict = vision_state_dict
        elif k.startswith("model") or k.startswith("lm_head"):
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


def split_hf_keys(loaded_weights, n_heads=None, n_kv_heads=None):
    converted_weights = {}
    for key, tensor in loaded_weights.items():
        if "qkv_proj" in key:
            # split Q, K and V
            q_key = key.replace("qkv_proj", "q_proj")
            k_key = key.replace("qkv_proj", "k_proj")
            v_key = key.replace("qkv_proj", "v_proj")

            # Handle GQA (Grouped Query Attention) case
            if n_heads is not None and n_kv_heads is not None and n_heads != n_kv_heads:
                # For GQA: Q has n_heads, K and V have n_kv_heads
                head_dim = tensor.shape[0] // (n_heads + 2 * n_kv_heads)
                q_size = n_heads * head_dim
                kv_size = n_kv_heads * head_dim

                q_tensor = tensor[:q_size]
                k_tensor = tensor[q_size : q_size + kv_size]
                v_tensor = tensor[q_size + kv_size : q_size + 2 * kv_size]
            else:
                # Default case: equal split for Q, K, V
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


def _is_hf_llama_vision(config):
    return hasattr(config, "text_config") and hasattr(config.text_config, "cross_attention_layers")


def reindex_layers(state_dict, config):
    """Only for Llama-Vision models
    Same functionality as in https://github.com/huggingface/transformers/blob/41980ce93e775f6c88500c51c8db7946fc6a2add/src/transformers/models/mllama/convert_mllama_weights_to_hf.py#L365-L369
    """

    if not _is_hf_llama_vision(config):
        return state_dict

    new_state_dict = {k: v for k, v in state_dict.items()}
    idx_cross_attn = len(config.text_config.cross_attention_layers) - 1
    idx_self_attn = config.text_config.num_hidden_layers - len(config.text_config.cross_attention_layers) - 1
    for i in range(config.text_config.num_hidden_layers - 1, -1, -1):
        if i in config.text_config.cross_attention_layers:
            keys = [k for k in new_state_dict if f"cross_attention_layers.{idx_cross_attn}." in k]
            for key in keys:
                new_key = key.replace(f"cross_attention_layers.{idx_cross_attn}.", f"layers.{i}.")
                new_state_dict[new_key] = new_state_dict.pop(key)
            idx_cross_attn -= 1
        else:
            keys = [k for k in new_state_dict if f"layers.{idx_self_attn}." in k]
            for key in keys:
                new_key = key.replace(f"layers.{idx_self_attn}.", f"layers.{i}.")
                new_state_dict[new_key] = new_state_dict.pop(key)
            idx_self_attn -= 1
    return new_state_dict


def rename_layers_to_cross_attn(state_dict, config):
    if not _is_hf_llama_vision(config):
        return state_dict

    mapping = {
        "self_attn.q_proj.weight": "cross_attn.q_proj.weight",
        "self_attn.k_proj.weight": "cross_attn.k_proj.weight",
        "self_attn.v_proj.weight": "cross_attn.v_proj.weight",
        "self_attn.o_proj.weight": "cross_attn.o_proj.weight",
        "self_attn.q_proj.bias": "cross_attn.q_proj.bias",
        "self_attn.k_proj.bias": "cross_attn.k_proj.bias",
        "self_attn.v_proj.bias": "cross_attn.v_proj.bias",
        "self_attn.o_proj.bias": "cross_attn.o_proj.bias",
        "self_attn.q_norm.weight": "cross_attn.q_norm.weight",
        "self_attn.k_norm.weight": "cross_attn.k_norm.weight",
    }

    new_state_dict = {}
    for key, tensor in state_dict.items():
        matched = False

        for idx in config.text_config.cross_attention_layers:
            if matched:
                break
            for self_attn, cross_attn in mapping.items():
                self_pattern = f"layers.{idx}.{self_attn}"
                cross_pattern = f"layers.{idx}.{cross_attn}"
                if self_pattern in key:
                    key = key.replace(self_pattern, cross_pattern)
                    new_state_dict[key] = tensor
                    matched = True
                    break

        if not matched:
            new_state_dict[key] = tensor

    return new_state_dict


def convert_meta_to_hf(state_dict, head_dim, fuse_qkv=False, fuse_mlp=False, config=None):
    state_dict = reindex_layers(state_dict, config)
    state_dict = convert_meta_qkv_to_hf_format(state_dict, head_dim)
    if fuse_qkv:
        state_dict = fuse_qkv_meta(state_dict)
    if fuse_mlp:
        state_dict = fuse_mlp_meta(state_dict)

    state_dict = map_meta_to_hf_keys(state_dict)
    state_dict = rename_layers_to_cross_attn(state_dict, config)
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


def map_hf_to_meta_keys_mllama(loaded_weights, config):
    replacements = [
        (r"^model.norm.weight", r"text_model.norm.weight"),
        (r"^lm_head.weight", r"text_model.output.weight"),
        (r"^model.embed_tokens", r"text_model.tok_embeddings"),
        (r"^vision_model.patch_embedding", r"vision_model.conv1._linear"),
        (
            r"^vision_model.(global_transformer|transformer).layers.(\d+).self_attn.q_proj",
            r"vision_model.\1.resblocks.\2.attn.wq",
        ),
        (
            r"^vision_model.(global_transformer|transformer).layers.(\d+).self_attn.k_proj",
            r"vision_model.\1.resblocks.\2.attn.wk",
        ),
        (
            r"^vision_model.(global_transformer|transformer).layers.(\d+).self_attn.v_proj",
            r"vision_model.\1.resblocks.\2.attn.wv",
        ),
        (
            r"^vision_model.(global_transformer|transformer).layers.(\d+).self_attn.o_proj",
            r"vision_model.\1.resblocks.\2.attn.wo",
        ),
        (
            r"^vision_model.(global_transformer|transformer).layers.(\d+).mlp.fc1",
            r"vision_model.\1.resblocks.\2.mlp.c_fc",
        ),
        (
            r"^vision_model.(global_transformer|transformer).layers.(\d+).mlp.fc2",
            r"vision_model.\1.resblocks.\2.mlp.c_proj",
        ),
        (
            r"^vision_model.(global_transformer|transformer).layers.(\d+).input_layernorm",
            r"vision_model.\1.resblocks.\2.ln_1",
        ),
        (
            r"^vision_model.(global_transformer|transformer).layers.(\d+).post_attention_layernorm",
            r"vision_model.\1.resblocks.\2.ln_2",
        ),
        (
            r"^vision_model.global_transformer.layers.(\d+).(gate_ffn|gate_attn)",
            r"vision_model.global_transformer.resblocks.\1.\2",
        ),
        (r"^vision_model.layernorm_(pre|post).(weight|bias)", r"vision_model.ln_\1.\2"),
        (r"^vision_model.gated_positional_embedding.embedding", r"vision_model.positional_embedding"),
        (r"^vision_model.gated_positional_embedding.tile_embedding.weight", r"vision_model.gated_positional_embedding"),
        (r"^vision_model.gated_positional_embedding.gate", r"vision_model.gated_positional_embedding_gate"),
        (r"^vision_model.pre_tile_positional_embedding.embedding.weight", r"vision_model.pre_tile_pos_embed.embedding"),
        (
            r"^vision_model.post_tile_positional_embedding.embedding.weight",
            r"vision_model.post_tile_pos_embed.embedding",
        ),
        (r"^vision_model.pre_tile_positional_embedding.gate", r"vision_model.pre_tile_pos_embed.gate"),
        (r"^vision_model.post_tile_positional_embedding.gate", r"vision_model.post_tile_pos_embed.gate"),
        (r"^vision_model.", r"vision_model.vision_encoder."),
        (r"^model.multi_modal_projector.", r"vision_model.vision_projection."),
    ]

    self_attn_replacements = {
        (r"^model.layers.(\d+).mlp.gate_proj.", r"text_model.layers.\1.feed_forward.w1."),
        (r"^model.layers.(\d+).mlp.down_proj.", r"text_model.layers.\1.feed_forward.w2."),
        (r"^model.layers.(\d+).mlp.up_proj.", r"text_model.layers.\1.feed_forward.w3."),
        (r"^model.layers.(\d+).input_layernorm.weight", r"text_model.layers.\1.attention_norm.weight"),
        (r"^model.layers.(\d+).post_attention_layernorm.weight", r"text_model.layers.\1.ffn_norm.weight"),
        (r"^model.layers.(\d+).self_attn.(q|k|v|o)_proj.weight", r"text_model.layers.\1.attention.w\2.weight"),
    }
    cross_attn_replacements = {
        (r"^model.layers.(\d+).mlp.gate_proj.weight", r"text_model.cross_attention_layers.\1.feed_forward.w1.weight"),
        (r"^model.layers.(\d+).mlp.down_proj.weight", r"text_model.cross_attention_layers.\1.feed_forward.w2.weight"),
        (r"^model.layers.(\d+).mlp.up_proj.weight", r"text_model.cross_attention_layers.\1.feed_forward.w3.weight"),
        (r"^model.layers.(\d+).input_layernorm.weight", r"text_model.cross_attention_layers.\1.attention_norm.weight"),
        (
            r"^model.layers.(\d+).post_attention_layernorm.weight",
            r"text_model.cross_attention_layers.\1.ffn_norm.weight",
        ),
        (r"^model.layers.(\d+).cross_attn_attn_gate", r"text_model.cross_attention_layers.\1.gate_attn"),
        (r"^model.layers.(\d+).cross_attn_mlp_gate", r"text_model.cross_attention_layers.\1.gate_ffwd"),
        (r"^model.layers.(\d+).cross_attn.(q|k|v|o)_proj", r"text_model.cross_attention_layers.\1.attention.w\2"),
        (r"^model.layers.(\d+).cross_attn.(q|k)_norm", r"text_model.cross_attention_layers.\1.attention.\2_norm"),
    }

    idx_cross_attn = 0
    for i in range(config.text_config.num_hidden_layers):
        if i in config.text_config.cross_attention_layers:
            cur_replacements = [
                (
                    k.replace(r"layers.(\d+).", rf"layers.{i}."),
                    v.replace(r"cross_attention_layers.\1.", rf"cross_attention_layers.{idx_cross_attn}.").replace(
                        r"\2", r"\1"
                    ),
                )
                for k, v in cross_attn_replacements
            ]
            idx_cross_attn += 1
        else:
            cur_replacements = [
                (
                    k.replace(r"layers.(\d+).", rf"layers.{i}."),
                    v.replace(r"layers.\1.", rf"layers.{i-idx_cross_attn}.").replace(r"\2", r"\1"),
                )
                for k, v in self_attn_replacements
            ]
        replacements.extend(cur_replacements)

    return replace_keys(loaded_weights, replacements)


def convert_pos_embeddings(state_dict):
    do_convert = lambda key: (
        ("tile_pos_embed.embedding" in key) or (key == "vision_model.vision_encoder.gated_positional_embedding")
    )
    state_dict = {k: invert_pre_compute_positional_embedding(v) if do_convert(k) else v for k, v in state_dict.items()}
    return state_dict


def invert_pre_compute_positional_embedding(precomputed_embeddings):
    """Inverts https://github.com/huggingface/transformers/blob/41980ce93e775f6c88500c51c8db7946fc6a2add/src/transformers/models/mllama/convert_mllama_weights_to_hf.py#L122-L148"""

    # TBD: remove hardcode
    if tuple(precomputed_embeddings.shape) == (9, 5120):
        max_aspect_ratio_id, max_num_tiles, num_patches, hidden_size = 9 - 1, 4, 1, 1280
    elif tuple(precomputed_embeddings.shape) == (9, 8197120):
        max_aspect_ratio_id, max_num_tiles, num_patches, hidden_size = 9 - 1, 4, 1601, 1280
    else:
        raise ValueError(f"Unknown embedding shape: {precomputed_embeddings.shape}")

    precomputed_embeddings = precomputed_embeddings.reshape(
        max_aspect_ratio_id + 1, max_num_tiles, num_patches, hidden_size
    )

    from transformers.models.mllama.image_processing_mllama import get_all_supported_aspect_ratios

    supported_aspect_ratios = get_all_supported_aspect_ratios(max_num_tiles)

    embedding = torch.zeros(max_num_tiles, max_num_tiles, num_patches, hidden_size, dtype=precomputed_embeddings.dtype)

    for i, (height, width) in enumerate(supported_aspect_ratios):
        aspect_ratio_id = i + 1
        current_embedding = precomputed_embeddings[aspect_ratio_id, : height * width]
        embedding[:height, :width] = current_embedding.reshape(height, width, num_patches, hidden_size)

    return embedding


def flatten_conv_linear(state_dict):
    do_flatten = lambda key: (("conv" in key) and ("_linear.weight" in key))
    state_dict = {k: v.flatten(start_dim=1) if do_flatten(k) else v for k, v in state_dict.items()}
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
