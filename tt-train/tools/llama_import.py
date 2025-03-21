#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
This script updates the weights in a msgpack file containing model weights
To use it you need to export tt-train model weights to msgpack file
and then run this script to update the weights with the new weights
from the Hugging Face model.

It supports GPT2 and Llama model architectures.
"""

import argparse
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import msgpack
import msgpack_numpy  # required for serializing numpy arrays
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional, Union

# Patch msgpack to support numpy arrays.
msgpack_numpy.patch()


class ModelConverter(ABC):
    """Base class for model weight conversion."""

    def __init__(self, model_name: str):
        """Initialize the converter with model name."""
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.state_dict = self.model.state_dict()
        print(f"Loaded model: {model_name}")
        print(f"State dict keys: {list(self.state_dict.keys())[:5]}...")

    @abstractmethod
    def get_mapping(self) -> Dict[str, str]:
        """Get mapping from msgpack keys to huggingface keys."""
        pass

    @abstractmethod
    def should_transpose(self, hf_key: str) -> bool:
        """Determine if a weight matrix should be transposed."""
        pass

    def update_weights(self, existing_state: Dict[str, Any]) -> Dict[str, Any]:
        """Update weights in existing_state with weights from the model."""
        mapping = self.get_mapping()

        # Find all transformer keys in existing state
        transformer_keys = [
            k
            for k in existing_state.keys()
            if isinstance(k, str) and k.startswith("transformer") and k.endswith("data")
        ]

        # Create a set of expected keys based on the mapping
        expected_keys = set()
        for target_pattern, _ in mapping.values():
            # For patterns with {layer}, expand for all layers
            if "{layer}" in target_pattern:
                for layer in range(self.get_num_layers()):
                    expected_keys.add(target_pattern.format(layer=layer))
            else:
                expected_keys.add(target_pattern)

        # Track which keys were updated
        updated_keys = set()

        # Update each key if it exists
        for msgpack_key in transformer_keys:
            for target_pattern, (source_pattern, needs_layer) in mapping.items():
                if needs_layer:
                    # Extract layer number and try to match
                    layer_match = self.extract_layer_num(msgpack_key, target_pattern)
                    if layer_match is not None:
                        layer_num = layer_match
                        hf_key = source_pattern.format(layer=layer_num)
                        if hf_key in self.state_dict:
                            self.update_key(existing_state, msgpack_key, self.state_dict[hf_key])
                            updated_keys.add(msgpack_key)
                            break
                else:
                    # Direct match without layer extraction
                    if msgpack_key == target_pattern:
                        hf_key = source_pattern
                        if hf_key in self.state_dict:
                            self.update_key(existing_state, msgpack_key, self.state_dict[hf_key])
                            updated_keys.add(msgpack_key)
                            break

        # Report stats
        print(f"Updated {len(updated_keys)} keys out of {len(transformer_keys)} transformer keys")
        print(f"Expected {len(expected_keys)} keys based on mapping")

        # Find missing expected keys
        missing_keys = expected_keys - set(transformer_keys)
        if missing_keys:
            print(f"Warning: {len(missing_keys)} expected keys were missing from the msgpack file")
            if len(missing_keys) < 10:
                for k in missing_keys:
                    print(f"  {k}")

        # Find keys that weren't updated
        not_updated = set(transformer_keys) - updated_keys
        if not_updated:
            print(f"Warning: {len(not_updated)} keys in msgpack file were not updated")
            if len(not_updated) < 10:
                for k in not_updated:
                    print(f"  {k}")

        return existing_state

    def update_key(self, existing_state: Dict[str, Any], key: str, new_value: np.ndarray) -> None:
        """
        Updates a key in existing_state with new_value.

        All stored values are linear (1D) arrays. For special keys (token embedding,
        fc head weight, etc.) we allow padding:

        - For embedding keys: new_value is a 2D array of shape (n_tokens, emb_dim).
          The stored value is a flat array representing a matrix with shape
          (expected_n_tokens, emb_dim). If new_value has fewer tokens, pad with zeros.

        - For all other keys, the flattened new_value must match the stored volume.
        """
        print(f"Updating key: {key}")
        # In the file, the weight is stored as a linear array in the second element.
        old_value = existing_state[key][1]
        old_array = np.array(old_value) if isinstance(old_value, list) else old_value
        new_value = new_value.cpu().numpy().astype(np.float32)

        # Check if we need to transpose the weight matrix
        hf_key = self.get_hf_key_for_msgpack_key(key)
        if hf_key and self.should_transpose(hf_key):
            print(f"Transposing weight: {key} (from {hf_key})")
            new_value = new_value.T

        # Define keys that require special treatment (may need padding)
        embedding_keys = {
            "transformer/llama/tok_emb/weight/value/data",
            "transformer/transformer/tok_emb/weight/value/data",
            "transformer/llama/fc/weight/value/data",
            "transformer/transformer/fc/weight/value/data",
        }

        if key in embedding_keys:
            # new_value is expected to be 2D: (n_tokens, emb_dim)
            if new_value.ndim != 2:
                raise ValueError(f"Expected new_value for key '{key}' to be 2D, got shape {new_value.shape}")
            emb_dim = new_value.shape[1]
            # Determine expected number of tokens from the stored flat array.
            if old_array.size % emb_dim != 0:
                raise ValueError(
                    f"Stored size for key '{key}' ({old_array.size}) is not divisible by emb_dim ({emb_dim})."
                )
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

    def get_hf_key_for_msgpack_key(self, msgpack_key: str) -> Optional[str]:
        """Get the corresponding HuggingFace key for a msgpack key."""
        mapping = self.get_mapping()
        for target_pattern, (source_pattern, needs_layer) in mapping.items():
            if needs_layer:
                layer_match = self.extract_layer_num(msgpack_key, target_pattern)
                if layer_match is not None:
                    return source_pattern.format(layer=layer_match)
            else:
                if msgpack_key == target_pattern:
                    return source_pattern
        return None

    def extract_layer_num(self, key: str, pattern: str) -> Optional[int]:
        """Extract layer number from key based on pattern."""
        # Replace {layer} with a regex pattern to capture the layer number
        import re

        if "{layer}" not in pattern:
            return None

        # Create a regex pattern by replacing {layer} with a capture group
        regex_pattern = pattern.replace("{layer}", "(\\d+)")
        regex_pattern = regex_pattern.replace(".", "\\.")  # Escape dots
        match = re.match(regex_pattern, key)

        if match:
            return int(match.group(1))
        return None

    @abstractmethod
    def get_num_layers(self) -> int:
        """Get the number of layers in the model."""
        pass


class GPT2Converter(ModelConverter):
    """Converter for GPT2 model weights."""

    def get_num_layers(self) -> int:
        """Get the number of layers in GPT2 model."""
        # Count how many transformer blocks the model has
        return len([k for k in self.state_dict.keys() if k.startswith("transformer.h.")])

    def should_transpose(self, hf_key: str) -> bool:
        """Determine if a GPT2 weight matrix should be transposed."""
        TRANSPOSED_KEYS = {"attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"}
        return any(substr in hf_key for substr in TRANSPOSED_KEYS)

    def get_mapping(self) -> Dict[str, Tuple[str, bool]]:
        """Get mapping from msgpack keys to huggingface keys for GPT2."""
        # Format is {msgpack_key: (huggingface_key, needs_layer_extraction)}
        mapping = {
            # Token and position embeddings
            "transformer/transformer/tok_emb/weight/value/data": ("transformer.wte.weight", False),
            "transformer/transformer/pos_emb/weight/value/data": ("transformer.wpe.weight", False),
            # Final layernorm and language model head
            "transformer/transformer/ln_fc/gamma/value/data": ("transformer.ln_f.weight", False),
            "transformer/transformer/ln_fc/beta/value/data": ("transformer.ln_f.bias", False),
            "transformer/transformer/fc/weight/value/data": ("transformer.wte.weight", False),
            # Layer-specific weights
            "transformer/transformer/gpt_block_{layer}/ln1/gamma/value/data": (
                "transformer.h.{layer}.ln_1.weight",
                True,
            ),
            "transformer/transformer/gpt_block_{layer}/ln1/beta/value/data": ("transformer.h.{layer}.ln_1.bias", True),
            "transformer/transformer/gpt_block_{layer}/attention/qkv_linear/weight/value/data": (
                "transformer.h.{layer}.attn.c_attn.weight",
                True,
            ),
            "transformer/transformer/gpt_block_{layer}/attention/qkv_linear/bias/value/data": (
                "transformer.h.{layer}.attn.c_attn.bias",
                True,
            ),
            "transformer/transformer/gpt_block_{layer}/attention/out_linear/weight/value/data": (
                "transformer.h.{layer}.attn.c_proj.weight",
                True,
            ),
            "transformer/transformer/gpt_block_{layer}/attention/out_linear/bias/value/data": (
                "transformer.h.{layer}.attn.c_proj.bias",
                True,
            ),
            "transformer/transformer/gpt_block_{layer}/ln2/gamma/value/data": (
                "transformer.h.{layer}.ln_2.weight",
                True,
            ),
            "transformer/transformer/gpt_block_{layer}/ln2/beta/value/data": ("transformer.h.{layer}.ln_2.bias", True),
            "transformer/transformer/gpt_block_{layer}/mlp/fc1/weight/value/data": (
                "transformer.h.{layer}.mlp.c_fc.weight",
                True,
            ),
            "transformer/transformer/gpt_block_{layer}/mlp/fc1/bias/value/data": (
                "transformer.h.{layer}.mlp.c_fc.bias",
                True,
            ),
            "transformer/transformer/gpt_block_{layer}/mlp/fc2/weight/value/data": (
                "transformer.h.{layer}.mlp.c_proj.weight",
                True,
            ),
            "transformer/transformer/gpt_block_{layer}/mlp/fc2/bias/value/data": (
                "transformer.h.{layer}.mlp.c_proj.bias",
                True,
            ),
        }
        return mapping


class LlamaConverter(ModelConverter):
    """Converter for Llama model weights."""

    def get_num_layers(self) -> int:
        """Get the number of layers in Llama model."""
        # Count how many transformer layers the model has
        layer_keys = [k for k in self.state_dict.keys() if k.startswith("model.layers.")]
        if not layer_keys:
            return 0
        layer_indices = set()
        for key in layer_keys:
            # Extract layer index from keys like model.layers.0.xxx
            parts = key.split(".")
            if len(parts) > 2 and parts[0] == "model" and parts[1] == "layers":
                try:
                    layer_indices.add(int(parts[2]))
                except ValueError:
                    continue
        return max(layer_indices) + 1 if layer_indices else 0

    def should_transpose(self, hf_key: str) -> bool:
        """Determine if a Llama weight matrix should be transposed."""
        # In Llama weights are typically not transposed compared to the Hugging Face format
        return False

    def get_mapping(self) -> Dict[str, Tuple[str, bool]]:
        """Get mapping from msgpack keys to huggingface keys for Llama."""
        # Format is {msgpack_key: (huggingface_key, needs_layer_extraction)}
        mapping = {
            # Token embeddings
            "transformer/llama/tok_emb/weight/value/data": ("model.embed_tokens.weight", False),
            # Final layernorm and language model head
            "transformer/llama/ln_fc/gamma/value/data": ("model.norm.weight", False),
            "transformer/llama/fc/weight/value/data": ("lm_head.weight", False),
            # Layer-specific weights
            "transformer/llama/llama_block_{layer}/attention_norm/gamma/value/data": (
                "model.layers.{layer}.input_layernorm.weight",
                True,
            ),
            "transformer/llama/llama_block_{layer}/mlp_norm/gamma/value/data": (
                "model.layers.{layer}.post_attention_layernorm.weight",
                True,
            ),
            # Attention weights
            "transformer/llama/llama_block_{layer}/attention/q_linear/weight/value/data": (
                "model.layers.{layer}.self_attn.q_proj.weight",
                True,
            ),
            "transformer/llama/llama_block_{layer}/attention/q_linear/bias/value/data": ("", True),  # No bias in Llama
            "transformer/llama/llama_block_{layer}/attention/kv_linear/weight/value/data": (
                "",
                True,
            ),  # Special handling needed
            "transformer/llama/llama_block_{layer}/attention/kv_linear/bias/value/data": ("", True),  # No bias in Llama
            "transformer/llama/llama_block_{layer}/attention/out_linear/weight/value/data": (
                "model.layers.{layer}.self_attn.o_proj.weight",
                True,
            ),
            "transformer/llama/llama_block_{layer}/attention/out_linear/bias/value/data": (
                "",
                True,
            ),  # No bias in Llama
            # MLP weights
            "transformer/llama/llama_block_{layer}/mlp/w1/weight/value/data": (
                "model.layers.{layer}.mlp.gate_proj.weight",
                True,
            ),
            "transformer/llama/llama_block_{layer}/mlp/w1/bias/value/data": ("", True),  # No bias in Llama
            "transformer/llama/llama_block_{layer}/mlp/w2/weight/value/data": (
                "model.layers.{layer}.mlp.down_proj.weight",
                True,
            ),
            "transformer/llama/llama_block_{layer}/mlp/w2/bias/value/data": ("", True),  # No bias in Llama
            "transformer/llama/llama_block_{layer}/mlp/w3/weight/value/data": (
                "model.layers.{layer}.mlp.up_proj.weight",
                True,
            ),
            "transformer/llama/llama_block_{layer}/mlp/w3/bias/value/data": ("", True),  # No bias in Llama
        }
        return mapping

    def update_weights(self, existing_state: Dict[str, Any]) -> Dict[str, Any]:
        """Update weights in existing_state with weights from the Llama model."""
        # Call parent method first for most weights
        existing_state = super().update_weights(existing_state)

        # Handle special case: kv_linear combines k_proj and v_proj in Llama
        transformer_keys = [
            k
            for k in existing_state.keys()
            if isinstance(k, str) and k.startswith("transformer/llama/llama_block_") and "kv_linear/weight" in k
        ]

        for key in transformer_keys:
            layer_match = self.extract_layer_num(
                key, "transformer/llama/llama_block_{layer}/attention/kv_linear/weight/value/data"
            )
            if layer_match is not None:
                k_key = f"model.layers.{layer_match}.self_attn.k_proj.weight"
                v_key = f"model.layers.{layer_match}.self_attn.v_proj.weight"

                if k_key in self.state_dict and v_key in self.state_dict:
                    # Stack k and v weights for the KV linear layer
                    k_weight = self.state_dict[k_key].cpu().numpy().astype(np.float32)
                    v_weight = self.state_dict[v_key].cpu().numpy().astype(np.float32)

                    print(f"Processing KV weights for layer {layer_match}")
                    print(f"K shape: {k_weight.shape}, V shape: {v_weight.shape}")

                    combined = np.vstack([k_weight, v_weight])
                    self.update_key(existing_state, key, torch.tensor(combined))

        return existing_state


def print_transformer_keys(data: Dict[str, Any], prefix: str = ""):
    """
    Recursively prints keys from a nested dictionary (or list)
    that start with 'transformer' and end with 'data'.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(key, str) and key.startswith("transformer") and key.endswith("data"):
                print(f"{prefix}{key}")
            print_transformer_keys(value, prefix=prefix + "  ")
    elif isinstance(data, list):
        for item in data:
            print_transformer_keys(item, prefix=prefix)


def load_and_update(existing_file: str, output_file: str, model_name: str, model_type: str = "auto"):
    """
    Load the existing state dictionary from the msgpack file,
    update it with weights from the specified model,
    and save the result to the output file.
    """
    # Load the existing state dictionary
    with open(existing_file, "rb") as f:
        existing_state = msgpack.unpack(f, raw=False)

    # Print existing keys for debugging
    print("Existing transformer keys:")
    transformer_keys = [
        k for k in existing_state.keys() if isinstance(k, str) and k.startswith("transformer") and k.endswith("data")
    ]
    print(f"Found {len(transformer_keys)} transformer keys")
    if len(transformer_keys) < 10:
        for k in transformer_keys:
            print(f"  {k}")

    # Determine model type if auto
    if model_type == "auto":
        # Check keys to determine model type
        if any("llama" in k for k in transformer_keys):
            model_type = "llama"
        else:
            model_type = "gpt2"
        print(f"Auto-detected model type: {model_type}")

    # Create appropriate converter
    if model_type.lower() == "gpt2":
        converter = GPT2Converter(model_name)
    elif model_type.lower() == "llama":
        converter = LlamaConverter(model_name)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported types: gpt2, llama")

    # Import torch only when needed (after model type is determined)
    import torch

    # Update weights
    updated_state = converter.update_weights(existing_state)

    # Save the updated state back to the output file
    with open(output_file, "wb") as f:
        msgpack.pack(updated_state, f)

    print(f"Updated weights in '{existing_file}' have been saved to '{output_file}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update model weights stored in a msgpack file with new weights from Hugging Face."
    )
    parser.add_argument("input_file", type=str, help="Path to the input msgpack file containing the model weights.")
    parser.add_argument("output_file", type=str, help="Path where the updated msgpack file will be saved.")
    parser.add_argument(
        "model_name",
        type=str,
        help="Name of the Hugging Face model to use (e.g., 'gpt2', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0').",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["auto", "gpt2", "llama"],
        default="auto",
        help="Type of model architecture. If 'auto', it will be detected from the msgpack file.",
    )

    args = parser.parse_args()
    load_and_update(args.input_file, args.output_file, args.model_name, args.model_type)
