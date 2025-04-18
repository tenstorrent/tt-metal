# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger

# Assuming Glm4ModelArgs will be importable here if needed
# from models.demos.glm4.tt.model_config import Glm4ModelArgs


def map_glm4_keys(state_dict):
    """Applies GLM-4 specific key mappings after generic mapping.

    Args:
        state_dict (dict): The state dictionary after generic HF -> Meta mapping.

    Returns:
        dict: State dictionary with GLM-4 keys correctly mapped.
    """
    # Specific mappings needed for GLM-4 post-norms
    glm4_key_map = {
        "post_self_attn_layernorm.weight": "post_attention_layernorm.weight",
        "post_mlp_layernorm.weight": "post_mlp_layernorm.weight",
    }

    remapped_state_dict = {}
    for key, tensor in state_dict.items():
        mapped = False
        # Check for layer-specific keys first
        if "layers." in key:
            parts = key.split(".")
            layer_num = parts[1]
            potential_glm4_key = ".".join(parts[2:])  # e.g., post_self_attn_layernorm.weight

            if potential_glm4_key in glm4_key_map:
                new_key_suffix = glm4_key_map[potential_glm4_key]
                new_key = f"layers.{layer_num}.{new_key_suffix}"
                remapped_state_dict[new_key] = tensor
                mapped = True
                logger.trace(f"Mapped GLM-4 key: {key} -> {new_key}")

        # Check for top-level GLM-4 keys (less likely but possible)
        if not mapped and key in glm4_key_map:
            new_key = glm4_key_map[key]
            remapped_state_dict[new_key] = tensor
            mapped = True
            logger.trace(f"Mapped GLM-4 key: {key} -> {new_key}")

        if not mapped:
            remapped_state_dict[key] = tensor  # Keep non-GLM4 keys

    return remapped_state_dict


def split_glm4_qkv(state_dict, args):
    """Splits the combined query_key_value tensor found in GLM-4 checkpoints.

    Args:
        state_dict (dict): The state dictionary potentially containing combined QKV.
        args (Glm4ModelArgs): Model arguments containing head info.

    Returns:
        dict: State dictionary with QKV potentially split.
    """
    converted_weights = {}
    keys_to_process = list(state_dict.keys())  # Iterate over a copy of keys

    for key in keys_to_process:
        if "self_attn.query_key_value" in key:
            tensor = state_dict[key]
            logger.trace(f"Splitting GLM-4 combined QKV tensor: {key}")
            # GLM-4 style QKV handling - split into Q, K, V
            q_key = key.replace("self_attn.query_key_value", "self_attn.q_proj")
            k_key = key.replace("self_attn.query_key_value", "self_attn.k_proj")
            v_key = key.replace("self_attn.query_key_value", "self_attn.v_proj")

            # Use args passed in for head dimensions
            n_heads = args.n_heads
            n_kv_heads = args.n_kv_heads
            head_dim = args.head_dim

            if n_heads is None or n_kv_heads is None or head_dim is None:
                logger.error(f"Missing head dimension arguments in Glm4ModelArgs for splitting {key}. Cannot split.")
                converted_weights[key] = tensor  # Keep original if args are missing
                continue

            # Calculate splitting sizes
            dim0_is_features = len(tensor.shape) == 2  # Weight tensor
            split_dim = 0 if dim0_is_features else 0  # Also split bias along dim 0

            if dim0_is_features:
                q_size = n_heads * head_dim
                k_size = n_kv_heads * head_dim
                v_size = n_kv_heads * head_dim
            else:  # Bias tensor
                q_size = n_heads * head_dim  # Bias shape might be different, adjust if needed based on model
                k_size = n_kv_heads * head_dim
                v_size = n_kv_heads * head_dim
                # Bias might be just hidden_dim for each Q, K, V part
                # Need to confirm GLM-4 bias structure
                # Assuming bias shape is like [q_hidden_dim + k_hidden_dim + v_hidden_dim]
                # Let's assume hidden_dim is args.dim for bias splitting (needs verification)
                q_size = args.dim
                k_size = args.dim * n_kv_heads // n_heads  # Rough scaling if bias depends on KV ratio
                v_size = args.dim * n_kv_heads // n_heads
                # Fallback to simple split if precise bias dimensions are unknown
                if q_size + k_size + v_size != tensor.shape[split_dim]:
                    logger.warning(
                        f"Could not determine exact GLM-4 bias split sizes for {key}. Using approximate split."
                    )
                    total_size = tensor.shape[split_dim]
                    q_size = total_size // 3
                    k_size = total_size // 3
                    v_size = total_size - q_size - k_size

            # Sanity check split sizes against tensor dimension
            if q_size + k_size + v_size != tensor.shape[split_dim]:
                logger.error(
                    f"Calculated split sizes ({q_size}, {k_size}, {v_size}) do not match tensor dimension {tensor.shape[split_dim]} for key {key}. Skipping split."
                )
                converted_weights[key] = tensor  # Keep original if split sizes mismatch
                continue

            # Split into Q, K, V tensors
            q_tensor, k_tensor, v_tensor = torch.split(tensor, [q_size, k_size, v_size], dim=split_dim)

            converted_weights[q_key] = q_tensor
            converted_weights[k_key] = k_tensor
            converted_weights[v_key] = v_tensor
            # Original combined key is removed by not adding it back
        else:
            # Keep other keys as they are
            if key in state_dict:  # Ensure key still exists (might be removed by previous split)
                converted_weights[key] = state_dict[key]

    return converted_weights


# Add a function to load and process the state dict
# NOTE: This assumes a generic way to load the raw state_dict exists,
# similar to load_llama_state_dict used in the llama demo template.
# If no such generic loader exists, this function might need to handle file loading directly.


def load_and_process_glm4_state_dict(ckpt_dir, args, n_layers=None):
    """Loads the state dict, splits QKV, and maps GLM-4 keys.

    Args:
        ckpt_dir (str): Path to the checkpoint directory.
        args (Glm4ModelArgs): Model arguments for head dimensions etc.
        n_layers (int, optional): Number of layers to load. Defaults to None (load all).

    Returns:
        dict: The processed state dictionary ready for model loading.
    """
    logger.info(f"Loading and processing GLM-4 state dict from: {ckpt_dir}")

    # --- Step 1: Load raw state dict ---
    # Placeholder: Replace with actual state dict loading mechanism
    # This might involve finding .safetensors or .bin files in ckpt_dir
    # Example structure borrowed from llama demo:
    # state_dict = load_llama_state_dict(ckpt_dir, n_layers=n_layers)
    # For now, assume state_dict is loaded somehow:
    # We need a function like `load_raw_state_dict(ckpt_dir)`
    # If it doesn't exist, we might need to implement basic file loading here.
    # Let's assume a hypothetical generic loader for now.

    # Hypothetical loading function:
    def load_raw_state_dict(directory):
        # Simplified loader: finds .safetensors and loads them
        # Needs safetensors library: pip install safetensors
        # Needs to handle sharded checkpoints correctly
        from safetensors import safe_open
        import os

        state_dict = {}
        safetensor_files = sorted([f for f in os.listdir(directory) if f.endswith(".safetensors")])

        if not safetensor_files:
            raise FileNotFoundError(f"No .safetensors files found in {directory}")

        logger.info(f"Loading weights from {len(safetensor_files)} safetensors files...")
        for filename in safetensor_files:
            file_path = os.path.join(directory, filename)
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
            logger.trace(f"Loaded {filename}")
        return state_dict

    try:
        raw_state_dict = load_raw_state_dict(ckpt_dir)
    except Exception as e:
        logger.error(f"Failed to load raw state dict from {ckpt_dir}: {e}")
        raise

    logger.info(f"Raw state dict loaded. Found {len(raw_state_dict)} keys.")

    # --- Step 2: Split combined QKV weights ---
    logger.info("Splitting combined QKV weights...")
    state_dict_after_qkv_split = split_glm4_qkv(raw_state_dict, args)
    keys_after_split = len(state_dict_after_qkv_split)
    logger.info(f"State dict keys after QKV split: {keys_after_split}")

    # --- Step 3: Map GLM-4 specific keys ---
    # Note: The map_glm4_keys function seems designed to run *after* a generic HF->Meta mapping.
    # If the raw checkpoint keys are already in the GLM-4 HF format, this map might not be needed
    # or needs adjustment. Assuming for now it works on the raw/split keys.
    logger.info("Applying GLM-4 specific key mappings...")
    processed_state_dict = map_glm4_keys(state_dict_after_qkv_split)
    keys_after_map = len(processed_state_dict)
    logger.info(f"State dict keys after GLM-4 mapping: {keys_after_map}")

    # Optional: Filter layers if n_layers is specified (needs key format check)
    if n_layers is not None:
        logger.warning("Layer filtering based on n_layers is not implemented in this loader yet.")
        # Add logic here to filter keys like `layers.{i}.*` if i >= n_layers

    logger.info("GLM-4 state dict processing complete.")
    return processed_state_dict
