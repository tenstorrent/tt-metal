# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test script for model_configs.py to verify the Pydantic models work correctly.

This script tests the parsing and standardization of different model configuration formats.
"""

import json

from models.tt_transformers.tt.model_configs import parse_model_config_from_dict


def test_meta_llama_config():
    """Test Meta format Llama configuration."""
    meta_config = json.load(open("models/tt_transformers/model_params/Llama-3.1-8B-Instruct/params.json"))

    standard_config = parse_model_config_from_dict(meta_config)

    print("Meta Llama Config Test:")
    print(f"  Architecture: {standard_config.architecture}")
    print(f"  Dimensions: {standard_config.dim}")
    print(f"  Layers: {standard_config.n_layers}")
    print(f"  Heads: {standard_config.n_heads}")
    print(f"  KV Heads: {standard_config.n_kv_heads}")
    print(f"  Vocab Size: {standard_config.vocab_size}")
    print(f"  Max Context Length: {standard_config.max_context_len}")
    print(f"  RoPE Scaling: {standard_config.rope_scaling}")
    print()


def test_hf_llama_config():
    """Test HuggingFace format Llama configuration."""
    hf_config = json.load(open("models/tt_transformers/model_params/Llama-3.1-8B-Instruct/config.json"))

    standard_config = parse_model_config_from_dict(hf_config)

    print("HuggingFace Llama Config Test:")
    print(f"  Architecture: {standard_config.architecture}")
    print(f"  Dimensions: {standard_config.dim}")
    print(f"  Layers: {standard_config.n_layers}")
    print(f"  Heads: {standard_config.n_heads}")
    print(f"  KV Heads: {standard_config.n_kv_heads}")
    print(f"  Vocab Size: {standard_config.vocab_size}")
    print(f"  Max Context Length: {standard_config.max_context_len}")
    print(f"  RoPE Scaling: {standard_config.rope_scaling}")
    print()


def test_qwen_config():
    """Test Qwen2.5 configuration."""
    qwen_config = json.load(open("models/tt_transformers/model_params/Qwen2.5-7B-Instruct/config.json"))

    standard_config = parse_model_config_from_dict(qwen_config)

    print("Qwen2.5 Config Test:")
    print(f"  Architecture: {standard_config.architecture}")
    print(f"  Dimensions: {standard_config.dim}")
    print(f"  Layers: {standard_config.n_layers}")
    print(f"  Heads: {standard_config.n_heads}")
    print(f"  KV Heads: {standard_config.n_kv_heads}")
    print(f"  Vocab Size: {standard_config.vocab_size}")
    print(f"  Max Position Embeddings: {standard_config.max_context_len}")
    print()


if __name__ == "__main__":
    print("Testing Model Configuration Parsers\n")
    print("=" * 50)

    test_meta_llama_config()
    test_hf_llama_config()
    test_qwen_config()

    print("All tests completed successfully!")
