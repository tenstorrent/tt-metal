# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0


# This script updates the weights in a msgpack file containing GPT-2 weights
# To use it you need to export tt-train gpt2s weights to msgpack file
# and then run this script to update the weights with the new GPT-2 weights
# from the Hugging Face model.

import argparse
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import msgpack
import msgpack_numpy  # required for serializing numpy arrays
import os
import json
import shutil

# Patch msgpack to support numpy arrays.
msgpack_numpy.patch()

# Set of substrings for GPT2 keys that should be transposed.
TRANSPOSED_KEYS = {"attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"}


def update_key(existing_state, key, new_value):
    """
    Updates a key in existing_state with new_value.

    All stored values are linear (1D) arrays. For special keys (token embedding,
    fc head weight and fc bias) we allow padding:

      - For embedding keys: new_value is a 2D array of shape (n_tokens, emb_dim).
        The stored value is a flat array representing a matrix with shape
        (expected_n_tokens, emb_dim). If new_value has fewer tokens, pad with zeros.

      - For all other keys, the flattened new_value must match the stored volume.
    """
    print(f"Updating key: {key}")
    # In your file, the weight is stored as a linear array in the second element.
    old_value = existing_state[key][1]
    old_array = np.array(old_value) if isinstance(old_value, list) else old_value
    new_value = new_value.astype(np.float32)
    # Define keys that require special treatment.
    embedding_keys = {
        "transformer/transformer/tok_emb/weight/value/data",
        "transformer/transformer/fc/weight/value/data",
    }

    if key in embedding_keys:
        # new_value is expected to be 2D: (n_tokens, emb_dim)
        if new_value.ndim != 2:
            raise ValueError(f"Expected new_value for key '{key}' to be 2D, got shape {new_value.shape}")
        emb_dim = new_value.shape[1]
        # Determine expected number of tokens from the stored flat array.
        if old_array.size % emb_dim != 0:
            raise ValueError(f"Stored size for key '{key}' ({old_array.size}) is not divisible by emb_dim ({emb_dim}).")
        expected_tokens = old_array.size // emb_dim
        print("Expected tokens:", expected_tokens)
        print("New value shape:", new_value.shape)
        if new_value.shape[0] < expected_tokens:
            pad_tokens = expected_tokens - new_value.shape[0]
            pad = np.zeros((pad_tokens, emb_dim), dtype=new_value.dtype)
            new_value = np.concatenate([new_value, pad], axis=0)
            print("Padded new value shape:", new_value.shape)
        elif new_value.shape[0] > expected_tokens:
            raise ValueError(
                f"New value for key '{key}' has more tokens than expected: new {new_value.shape[0]} vs expected {expected_tokens}"
            )

        new_value_flat = new_value.flatten()
        existing_state[key][1] = new_value_flat.tolist()
        return

    # For all other keys, require an exact match in total number of elements.
    new_value_flat = new_value.flatten()
    if old_array.size != new_value_flat.size:
        raise ValueError(
            f"Mismatch in size for key '{key}': existing volume {old_array.size} != new volume {new_value_flat.size}"
        )
    print("New value shape:", new_value.shape)
    existing_state[key][1] = new_value_flat.tolist()


def print_transformer(data, prefix=""):
    """
    Recursively prints keys from a nested dictionary (or list)
    that start with 'transformer' and end with 'data'.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(key, str) and key.startswith("transformer") and key.endswith("data"):
                print(f"{prefix}{key}")
            print_transformer(value, prefix=prefix + "  ")
    elif isinstance(data, list):
        for item in data:
            print_transformer(item, prefix=prefix)


def load_and_update(existing_file, output_file):
    # Load the existing state dictionary from the msgpack file.
    with open(existing_file, "rb") as f:
        existing_state = msgpack.unpack(f, raw=False)

    # Load GPT-2 small model and its state dictionary.
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    state_dict = model.state_dict()
    print("GPT2 state_dict keys:")
    print(state_dict.keys())

    # We'll build a set of expected file keys from the GPT2 side.
    expected_keys = set()

    # Update each transformer block (0 to 11).
    for i in range(12):
        prefix = f"transformer.h.{i}"
        layer_keys = {
            f"transformer/transformer/gpt_block_{i}/ln1/gamma/value/data",
            f"transformer/transformer/gpt_block_{i}/ln1/beta/value/data",
            f"transformer/transformer/gpt_block_{i}/attention/qkv_linear/weight/value/data",
            f"transformer/transformer/gpt_block_{i}/attention/qkv_linear/bias/value/data",
            f"transformer/transformer/gpt_block_{i}/attention/out_linear/weight/value/data",
            f"transformer/transformer/gpt_block_{i}/attention/out_linear/bias/value/data",
            f"transformer/transformer/gpt_block_{i}/ln2/gamma/value/data",
            f"transformer/transformer/gpt_block_{i}/ln2/beta/value/data",
            f"transformer/transformer/gpt_block_{i}/mlp/fc1/weight/value/data",
            f"transformer/transformer/gpt_block_{i}/mlp/fc1/bias/value/data",
            f"transformer/transformer/gpt_block_{i}/mlp/fc2/weight/value/data",
            f"transformer/transformer/gpt_block_{i}/mlp/fc2/bias/value/data",
        }
        expected_keys.update(layer_keys)

        # For each key, if it exists in the file then update it.
        # ln1.gamma
        key_name = f"transformer/transformer/gpt_block_{i}/ln1/gamma/value/data"
        if key_name in existing_state:
            gpt2_key = f"{prefix}.ln_1.weight"
            new_val = state_dict[gpt2_key].cpu().numpy()
            update_key(existing_state, key_name, new_val)
        # ln1.beta
        key_name = f"transformer/transformer/gpt_block_{i}/ln1/beta/value/data"
        if key_name in existing_state:
            gpt2_key = f"{prefix}.ln_1.bias"
            new_val = state_dict[gpt2_key].cpu().numpy()
            update_key(existing_state, key_name, new_val)
        # attn.qkv_linear.weight (from attn.c_attn.weight, transposed)
        key_name = f"transformer/transformer/gpt_block_{i}/attention/qkv_linear/weight/value/data"
        if key_name in existing_state:
            gpt2_key = f"{prefix}.attn.c_attn.weight"
            new_val = state_dict[gpt2_key].cpu().numpy()
            if any(x in gpt2_key for x in TRANSPOSED_KEYS):
                print("transposing:", gpt2_key)
                new_val = new_val.T
            update_key(existing_state, key_name, new_val)
        # attn.qkv_linear.bias
        key_name = f"transformer/transformer/gpt_block_{i}/attention/qkv_linear/bias/value/data"
        if key_name in existing_state:
            gpt2_key = f"{prefix}.attn.c_attn.bias"
            new_val = state_dict[gpt2_key].cpu().numpy()
            update_key(existing_state, key_name, new_val)
        # attn.out_linear.weight (from attn.c_proj.weight, transposed)
        key_name = f"transformer/transformer/gpt_block_{i}/attention/out_linear/weight/value/data"
        if key_name in existing_state:
            gpt2_key = f"{prefix}.attn.c_proj.weight"
            new_val = state_dict[gpt2_key].cpu().numpy()
            if any(x in gpt2_key for x in TRANSPOSED_KEYS):
                print("transposing:", gpt2_key)
                new_val = new_val.T
            update_key(existing_state, key_name, new_val)
        # attn.out_linear.bias
        key_name = f"transformer/transformer/gpt_block_{i}/attention/out_linear/bias/value/data"
        if key_name in existing_state:
            gpt2_key = f"{prefix}.attn.c_proj.bias"
            new_val = state_dict[gpt2_key].cpu().numpy()
            update_key(existing_state, key_name, new_val)
        # ln2.gamma
        key_name = f"transformer/transformer/gpt_block_{i}/ln2/gamma/value/data"
        if key_name in existing_state:
            gpt2_key = f"{prefix}.ln_2.weight"
            new_val = state_dict[gpt2_key].cpu().numpy()
            update_key(existing_state, key_name, new_val)
        # ln2.beta
        key_name = f"transformer/transformer/gpt_block_{i}/ln2/beta/value/data"
        if key_name in existing_state:
            gpt2_key = f"{prefix}.ln_2.bias"
            new_val = state_dict[gpt2_key].cpu().numpy()
            update_key(existing_state, key_name, new_val)
        # mlp.fc1.weight (from mlp.c_fc.weight, transposed)
        key_name = f"transformer/transformer/gpt_block_{i}/mlp/fc1/weight/value/data"
        if key_name in existing_state:
            gpt2_key = f"{prefix}.mlp.c_fc.weight"
            new_val = state_dict[gpt2_key].cpu().numpy()
            if any(x in gpt2_key for x in TRANSPOSED_KEYS):
                print("transposing:", gpt2_key)
                new_val = new_val.T
            update_key(existing_state, key_name, new_val)
        # mlp.fc1.bias
        key_name = f"transformer/transformer/gpt_block_{i}/mlp/fc1/bias/value/data"
        if key_name in existing_state:
            gpt2_key = f"{prefix}.mlp.c_fc.bias"
            new_val = state_dict[gpt2_key].cpu().numpy()
            update_key(existing_state, key_name, new_val)
        # mlp.fc2.weight (from mlp.c_proj.weight, transposed)
        key_name = f"transformer/transformer/gpt_block_{i}/mlp/fc2/weight/value/data"
        if key_name in existing_state:
            gpt2_key = f"{prefix}.mlp.c_proj.weight"
            new_val = state_dict[gpt2_key].cpu().numpy()
            if any(x in gpt2_key for x in TRANSPOSED_KEYS):
                print("transposing:", gpt2_key)
                new_val = new_val.T
            update_key(existing_state, key_name, new_val)
        # mlp.fc2.bias
        key_name = f"transformer/transformer/gpt_block_{i}/mlp/fc2/bias/value/data"
        if key_name in existing_state:
            gpt2_key = f"{prefix}.mlp.c_proj.bias"
            new_val = state_dict[gpt2_key].cpu().numpy()
            update_key(existing_state, key_name, new_val)

    # Top-level keys.
    top_level_mapping = {
        "transformer/transformer/tok_emb/weight/value/data": "transformer.wte.weight",
        "transformer/transformer/pos_emb/weight/value/data": "transformer.wpe.weight",
        "transformer/transformer/ln_fc/gamma/value/data": "transformer.ln_f.weight",
        "transformer/transformer/ln_fc/beta/value/data": "transformer.ln_f.bias",
        "transformer/transformer/fc/weight/value/data": "transformer.wte.weight",
    }
    for target_key, hf_key in top_level_mapping.items():
        expected_keys.add(target_key)
        if target_key in existing_state:
            new_val = state_dict[hf_key].cpu().numpy()
            # For top-level fc weight, do NOT transpose unless needed.
            update_key(existing_state, target_key, new_val)

    # --- Check for missing or extra keys in the file ---
    expected_keys.update(top_level_mapping.keys())
    file_keys = {
        k for k in existing_state.keys() if isinstance(k, str) and k.startswith("transformer") and k.endswith("data")
    }
    missing_keys = expected_keys - file_keys
    if missing_keys:
        print("Warning: The following expected GPT2 file keys are missing:")
        for k in missing_keys:
            print(f"  {k}")

    extra_keys = file_keys - expected_keys
    if extra_keys:
        print("Warning: The following keys are present in the file but were not updated from GPT2 state:")
        for k in extra_keys:
            print(f"  {k}")

    # --- Print unused GPT2 keys with their shapes ---
    used_gpt2_keys = set()
    # Record used keys from each block.
    for i in range(12):
        prefix = f"transformer.h.{i}"
        used_gpt2_keys.update(
            {
                f"{prefix}.ln_1.weight",
                f"{prefix}.ln_1.bias",
                f"{prefix}.attn.c_attn.weight",
                f"{prefix}.attn.c_attn.bias",
                f"{prefix}.attn.c_proj.weight",
                f"{prefix}.attn.c_proj.bias",
                f"{prefix}.ln_2.weight",
                f"{prefix}.ln_2.bias",
                f"{prefix}.mlp.c_fc.weight",
                f"{prefix}.mlp.c_fc.bias",
                f"{prefix}.mlp.c_proj.weight",
                f"{prefix}.mlp.c_proj.bias",
            }
        )
    for hf_key in sorted(state_dict.keys()):
        if hf_key not in used_gpt2_keys:
            print(f"Unused GPT2 key: {hf_key}: {state_dict[hf_key].shape}")

    # Save the updated state back to the output file.
    with open(output_file, "wb") as f:
        msgpack.pack(existing_state, f)

    print(f"Updated weights in '{existing_file}' have been saved to '{output_file}'.")
    print("Listing transformer keys after update:")
    print_transformer(existing_state)


def dump_tokenizer(tokenizer_path):
    cache_dir = "/tmp/gpt2_export"
    os.makedirs(cache_dir, exist_ok=True)
    GPT2Tokenizer.from_pretrained("gpt2", cache_dir=cache_dir)

    output_dir = os.path.dirname(tokenizer_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Find tokenizer.json in the cache directory
    tokenizer_json_path = None
    for root, _, files in os.walk(cache_dir):
        if "tokenizer.json" in files:
            tokenizer_json_path = os.path.join(root, "tokenizer.json")
            break

    if tokenizer_json_path is None:
        raise FileNotFoundError("Could not find tokenizer.json in the cache directory")

    shutil.copy(tokenizer_json_path, tokenizer_path)
    print(f"GPT2 tokenizer saved to {tokenizer_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update GPT-2 weights stored in a msgpack file with new GPT-2 state.")
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        help="Path to the input msgpack file containing the GPT-2 weights.",
        default=None,
    )
    parser.add_argument(
        "-o", "--output_file", type=str, help="Path where the updated msgpack file will be saved.", default=None
    )

    parser.add_argument("-t", "--dump_tokenizer_path", type=str, default=None, help="Path to the output tokenizer file")
    args = parser.parse_args()

    if not any([args.dump_tokenizer_path, args.input_file, args.output_file]):
        print("Nothing to do. Please either specify --dump_tokenizer_path or both of --input_file and --output_file.")
        exit(1)
    if args.dump_tokenizer_path:
        dump_tokenizer(args.dump_tokenizer_path)
    if any([args.input_file, args.output_file]) and not all([args.input_file, args.output_file]):
        print("Note: both of input_file and output_file are required to export the weights.")
        exit(1)
    if all([args.input_file, args.output_file]):
        load_and_update(args.input_file, args.output_file)
