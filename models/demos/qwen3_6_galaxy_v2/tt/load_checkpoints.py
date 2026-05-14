# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import time

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


def _is_qwen36_state_dict(state_dict):
    """Detect Qwen3.6 HF state-dict by its distinctive ``model.language_model.`` prefix.

    Qwen3.6-27B is a VLM whose language tower lives under ``model.language_model.*``
    (as opposed to llama / Qwen3-32B where it lives under ``model.*``). This makes
    detection unambiguous from the keys alone, so we do not require the caller to
    pass ``model_args``. Mirrors the auto-detection style used elsewhere in the
    galaxy ports (olmo precedent: ``models/demos/olmo_galaxy/tt/load_checkpoints.py``).
    """
    return any(k.startswith("model.language_model.") for k in state_dict)


def _strip_qwen36_vlm_keys(state_dict):
    """Drop the vision tower and MTP blocks; LM-only bring-up keeps only the LM keys.

    Qwen3.6-27B is a VLM; the safetensors contain ``model.visual.*`` and ``mtp.*``
    weights that the LM-only TT model does not consume. Drop them up front so they
    don't waste host memory through the rest of the pipeline.
    """
    return {k: v for k, v in state_dict.items() if not (k.startswith("model.visual.") or k.startswith("mtp."))}


def standardize_hf_keys(state_dict):
    if _is_qwen36_state_dict(state_dict):
        return standardize_hf_keys_qwen36(state_dict)
    if not "lm_head.weight" in state_dict:
        # Assume tied to the embeddings if not present
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
    return state_dict


def standardize_hf_keys_qwen36(state_dict):
    """Rename Qwen3.6 HF keys (``model.language_model.*``) onto the canonical
    ``model.*`` namespace and drop VLM/MTP weights.

    After this pass the state-dict looks like a Qwen3-32B-style HF state-dict for the
    full_attention layers; the ``linear_attn.*`` sub-keys keep their distinctive
    names so :func:`map_hf_to_meta_keys` can route them into the DeltaNet namespace.
    """
    state_dict = _strip_qwen36_vlm_keys(state_dict)
    renamed = {}
    for k, v in state_dict.items():
        if k.startswith("model.language_model."):
            new_key = "model." + k[len("model.language_model.") :]
        else:
            new_key = k
        renamed[new_key] = v
    # Qwen3.6-27B has tie_word_embeddings=False; lm_head.weight is present in the
    # checkpoint and should NOT be aliased to the embedding. Only synthesize a
    # tied head if the checkpoint genuinely omits it.
    if "lm_head.weight" not in renamed and "model.embed_tokens.weight" in renamed:
        renamed["lm_head.weight"] = renamed["model.embed_tokens.weight"]
    return renamed


def convert_hf_to_meta(state_dict, head_dim, is_qwen36=None):
    """Convert an HF state-dict into the canonical meta-style layout.

    Args:
        state_dict: state-dict (already passed through ``standardize_hf_keys``).
        head_dim: head dimension, used by the QKV ``reverse_permute`` step.
        is_qwen36: optional bool. If ``None``, auto-detected from the keys (any
            ``.linear_attn.`` sub-key, or the pre-standardize ``model.language_model.``
            prefix). Pass ``True`` explicitly for the full_attention-only test
            scenario where no DeltaNet keys are present.
    """
    if is_qwen36 is None:
        is_qwen36 = _is_qwen36_state_dict(state_dict) or any(".linear_attn." in k for k in state_dict)
    state_dict = split_hf_keys(state_dict)
    if is_qwen36:
        # Qwen3.6-27B has ``attn_output_gate=True`` (q_proj packs Q and an output
        # gate, doubling the leading dim) and ``partial_rotary_factor=0.25``. The
        # llama "reverse_permute" trick assumes a single Q tensor of
        # ``n_heads * head_dim`` rows that gets a full RoPE — neither holds here.
        # The TT attention block carries an ``is_qwen36`` branch (V2-4
        # ``llama_attention.py``) that consumes the un-permuted HF layout
        # directly, so we skip the permutation step. q_norm/k_norm are also
        # already per-head ``[head_dim]`` vectors and require no permutation.
        # TODO(V2-4): revisit if/when the TT attention path is unified.
        pass
    else:
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
        "model.layers.{layer}.mlp.gate_proj.weight": "layers.{layer}.feed_forward.w1.weight",
        "model.layers.{layer}.mlp.up_proj.weight": "layers.{layer}.feed_forward.w3.weight",
        "model.layers.{layer}.mlp.down_proj.weight": "layers.{layer}.feed_forward.w2.weight",
        # Mappings for models with qk_norm
        "model.layers.{layer}.self_attn.q_norm.weight": "layers.{layer}.attention.q_norm.weight",
        "model.layers.{layer}.self_attn.k_norm.weight": "layers.{layer}.attention.k_norm.weight",
        # ----------------------------------------------------------------
        # Qwen3.6 DeltaNet (``linear_attn.*``) keys — kept under their own
        # ``layers.{i}.linear_attn.*`` namespace so V2-5
        # ``qwen36_delta_attention.py`` can ingest them without colliding with
        # the self-attention namespace. Only present on layers whose
        # ``config.layer_types[i] == "linear_attention"``. The sub-key suffixes
        # (``in_proj_qkv``, ``in_proj_z``, ``in_proj_a``, ``in_proj_b``,
        # ``conv1d``, ``A_log``, ``dt_bias``, ``norm.weight``, ``out_proj``)
        # are preserved verbatim from the HF safetensors so the DeltaNet
        # constructor can keep using the same names its reference code uses.
        # ----------------------------------------------------------------
        "model.layers.{layer}.linear_attn.in_proj_qkv.weight": "layers.{layer}.linear_attn.in_proj_qkv.weight",
        "model.layers.{layer}.linear_attn.in_proj_z.weight": "layers.{layer}.linear_attn.in_proj_z.weight",
        "model.layers.{layer}.linear_attn.in_proj_a.weight": "layers.{layer}.linear_attn.in_proj_a.weight",
        "model.layers.{layer}.linear_attn.in_proj_b.weight": "layers.{layer}.linear_attn.in_proj_b.weight",
        "model.layers.{layer}.linear_attn.conv1d.weight": "layers.{layer}.linear_attn.conv1d.weight",
        "model.layers.{layer}.linear_attn.A_log": "layers.{layer}.linear_attn.A_log",
        "model.layers.{layer}.linear_attn.dt_bias": "layers.{layer}.linear_attn.dt_bias",
        "model.layers.{layer}.linear_attn.norm.weight": "layers.{layer}.linear_attn.norm.weight",
        "model.layers.{layer}.linear_attn.out_proj.weight": "layers.{layer}.linear_attn.out_proj.weight",
    }

    meta_state_dict = {}
    for key, tensor in loaded_weights.items():
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
    is_chunked = any(ckpt.stem.startswith("layers_") for ckpt in checkpoints)
    if is_chunked:
        checkpoints = [ckpt_name for ckpt_name in checkpoints if ckpt_name.stem.startswith("layers_")]
        checkpoint = load_chunked_checkpoints(checkpoints, n_layers, start_layer_idx)
    else:
        checkpoint = load_sharded_checkpoints_optimized(checkpoints, n_layers)

    return checkpoint


def load_chunked_checkpoints(checkpoints, n_layers, start_layer_idx):
    checkpoint = {}
    (f"Loading {len(checkpoints)} checkpoint files")
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


def load_sharded_checkpoints_optimized(checkpoints, n_layers):
    logger.info(f"Loading {len(checkpoints)} checkpoint files")
    start_time = time()

    def load_single_checkpoint(ckpt_path):
        local_data = defaultdict(list)
        loaded_ckpt = torch.load(ckpt_path, map_location="cpu")
        for key, value in loaded_ckpt.items():
            if "layers." in key:
                layer_num = int(key.split("layers.")[1].split(".")[0])
                if n_layers and layer_num >= n_layers:
                    continue
            local_data[key].append(value)
        del loaded_ckpt
        return local_data

    all_checkpoints = []
    with ThreadPoolExecutor(max_workers=min(8, len(checkpoints))) as executor:
        for result in tqdm(executor.map(load_single_checkpoint, checkpoints), total=len(checkpoints)):
            all_checkpoints.append(result)

    # Merge all dictionaries
    checkpoint = defaultdict(list)
    for partial in all_checkpoints:
        for key, values in partial.items():
            checkpoint[key].extend(values)

    # Concatenate checkpoint values
    for key, values in checkpoint.items():
        if len(values) == 1 or "norm" in key:
            checkpoint[key] = values[0]
        elif key in {"tok_embeddings.weight", "output.weight"}:
            assert values[0].shape[1] == 8192  # FIXME: make configurable
            checkpoint[key] = torch.cat(values, dim=0)
        else:
            cat_dim = values[0].shape.index(min(values[0].shape))
            checkpoint[key] = torch.cat(values, dim=cat_dim)

    logger.info(f"Loaded and merged in {time() - start_time:.2f}s")
    return checkpoint


def load_sharded_checkpoints(checkpoints, n_layers):
    checkpoint = {}
    logger.info(f"Loading {len(checkpoints)} checkpoint files")
    for ckpt in tqdm(checkpoints):
        loaded_ckpt = torch.load(ckpt, map_location="cpu")
        for (
            key,
            value,
        ) in loaded_ckpt.items():
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
        if len(value) == 1 or "norm" in key:
            checkpoint[key] = value[0]
        else:
            if key == "tok_embeddings.weight" or key == "output.weight":
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


def convert_meta_to_hf(state_dict, head_dim):
    state_dict = convert_meta_qkv_to_hf_format(state_dict, head_dim)
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
        # Feed forward module
        "feed_forward.w1.weight": "mlp.gate_proj.weight",
        "feed_forward.w3.weight": "mlp.up_proj.weight",
        "feed_forward.w2.weight": "mlp.down_proj.weight",
        # Direct mappings for when we get just the final components
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
            if key.endswith(meta_pattern):
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


def get_qwen36_linear_attention_pattern(ckpt_dir):
    """Return the ``layer_types`` list from a Qwen3.6 HF snapshot's ``config.json``.

    Returns a list of length ``num_hidden_layers`` whose elements are either
    ``"linear_attention"`` (DeltaNet) or ``"full_attention"`` (standard GQA).
    """
    cfg_path = os.path.join(ckpt_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    text_cfg = cfg.get("text_config", cfg)
    pattern = text_cfg.get("layer_types")
    if pattern is None:
        raise ValueError(
            f"Could not find ``text_config.layer_types`` in {cfg_path}; this does "
            f"not look like a Qwen3.6 checkpoint."
        )
    return list(pattern)


def permute_1d(tensor):
    """Convert the last dim of a tensor from interleaved rope format (r1, i1, r2, i2, ...) to separate real and imaginary parts (r1, r2, i1, i2, ...)"""
    shape = tensor.shape
    dim = shape[-1]
    assert dim % 2 == 0, "Last dimension must be even"
    reshaped = tensor.reshape(*shape[:-1], dim // 2, 2)
    reals = reshaped[..., 0]
    imags = reshaped[..., 1]
    return torch.cat((reals, imags), dim=-1)
