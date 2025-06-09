# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import load_file as safetensors_load_file
from tqdm import tqdm


def _get_known_prefixes_mapping():
    return {
        # Llama Vision
        "text_model.": "",
        "vision_model.": "",
        # Gemma3
        "model.language_model.": "model.",
        "model.vision_tower.": "model.",
    }


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

    # Check if the key_meta exists with any known prefix
    if not any(f"{prefix}{key_meta}" in state_dict for prefix in _get_known_prefixes_mapping().keys()):
        # Assume tied to the embeddings if not present
        for prefix in _get_known_prefixes_mapping().keys():
            if f"{prefix}{key_hf}" in state_dict:
                state_dict[f"{prefix}{key_meta}"] = state_dict[f"{prefix}{key_hf}"]
                break

    return state_dict


def convert_hf_to_meta(state_dict, head_dim):
    state_dict = split_hf_keys(state_dict)
    state_dict = convert_hf_qkv_to_meta_format(state_dict, head_dim)
    state_dict = map_hf_to_meta_keys(state_dict)
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
        # Feed forward module mappings
        "mlp.gate_proj.weight": "feed_forward.w1.weight",
        "mlp.up_proj.weight": "feed_forward.w3.weight",
        "mlp.down_proj.weight": "feed_forward.w2.weight",
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
        "model.layers.{layer}.mlp.gate_proj.weight": "layers.{layer}.feed_forward.w1.weight",
        "model.layers.{layer}.mlp.up_proj.weight": "layers.{layer}.feed_forward.w3.weight",
        "model.layers.{layer}.mlp.down_proj.weight": "layers.{layer}.feed_forward.w2.weight",
    }

    meta_state_dict = {}
    for key, tensor in loaded_weights.items():
        # Remove known prefix if present
        prefix = next((p for p in _get_known_prefixes_mapping().keys() if key.startswith(p)), "")
        key = key.replace(prefix, _get_known_prefixes_mapping().get(prefix, ""), 1)

        if key in hf_to_meta:
            # Direct match for top-level keys
            meta_state_dict[hf_to_meta[key]] = tensor
        elif "model.layers." in key:
            # Extract layer number and form a template key
            parts = key.split(".")
            layer_num = parts[2]  # e.g. "0" in "model.layers.0.input_layernorm.weight"
            template_key = "model.layers.{layer}." + ".".join(parts[3:])
            if template_key in hf_to_meta:
                meta_state_dict[hf_to_meta[template_key].format(layer=layer_num)] = tensor

    return meta_state_dict


def load_meta_state_dict(ckpt_dir, n_layers=None, start_layer_idx=0):
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
    is_chunked = "layers_" in str(checkpoints[0])
    if is_chunked:
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
        if "self_attn.qkv_proj" in key:
            # split Q, K and V
            q_key = key.replace("self_attn.qkv_proj", "self_attn.q_proj")
            k_key = key.replace("self_attn.qkv_proj", "self_attn.k_proj")
            v_key = key.replace("self_attn.qkv_proj", "self_attn.v_proj")
            q_tensor, k_tensor, v_tensor = torch.split(tensor, tensor.shape[0] // 3, dim=0)
            converted_weights[q_key] = q_tensor
            converted_weights[k_key] = k_tensor
            converted_weights[v_key] = v_tensor
        elif "mlp.gate_up_proj" in key:
            # Split Gate and Up
            gate_key = key.replace("mlp.gate_up_proj", "mlp.gate_proj")
            up_key = key.replace("mlp.gate_up_proj", "mlp.up_proj")
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


def map_meta_to_hf_keys(loaded_weights):
    # Define mappings at each level of the hierarchy
    meta_to_hf_mappings = {
        # Top level
        "tok_embeddings.weight": "model.embed_tokens.weight",
        "norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
        # Layer level
        "attention_norm.weight": "input_layernorm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
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
        "attention.wqkv.weight": "self_attn.qkv_proj.weight",
        "attention.wqkv.bias": "self_attn.qkv_proj.bias",
        # Feed forward module
        "feed_forward.w1.weight": "mlp.gate_proj.weight",
        "feed_forward.w3.weight": "mlp.up_proj.weight",
        "feed_forward.w2.weight": "mlp.down_proj.weight",
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
        # fused qkv mapping
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
