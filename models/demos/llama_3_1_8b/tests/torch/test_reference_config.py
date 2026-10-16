# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""D1: Validate config class against vendored HF config.json."""

import json
from pathlib import Path

import pytest

from models.demos.llama_3_1_8b.config import Llama31Config, Llama31ModelConfig

HF_CONFIG_PATH = Path("/mnt/models/meta-llama/Llama-3.1-8B-Instruct/config.json")


@pytest.fixture
def hf_config_dict():
    if not HF_CONFIG_PATH.exists():
        pytest.skip(f"HF config not found at {HF_CONFIG_PATH}")
    with open(HF_CONFIG_PATH) as f:
        return json.load(f)


def test_config_constants_match_hf(hf_config_dict):
    assert Llama31Config.hidden_size == hf_config_dict["hidden_size"]
    assert Llama31Config.intermediate_size == hf_config_dict["intermediate_size"]
    assert Llama31Config.num_attention_heads == hf_config_dict["num_attention_heads"]
    assert Llama31Config.num_key_value_heads == hf_config_dict["num_key_value_heads"]
    assert Llama31Config.num_hidden_layers == hf_config_dict["num_hidden_layers"]
    assert Llama31Config.vocab_size == hf_config_dict["vocab_size"]
    assert Llama31Config.rms_norm_eps == hf_config_dict["rms_norm_eps"]
    assert Llama31Config.rope_theta == hf_config_dict["rope_theta"]
    assert Llama31Config.head_dim == hf_config_dict["hidden_size"] // hf_config_dict["num_attention_heads"]


def test_model_config_constants_match(hf_config_dict):
    assert Llama31ModelConfig.NUM_ATTENTION_HEADS == hf_config_dict["num_attention_heads"]
    assert Llama31ModelConfig.NUM_KEY_VALUE_HEADS == hf_config_dict["num_key_value_heads"]
    assert Llama31ModelConfig.HEAD_DIM == hf_config_dict["hidden_size"] // hf_config_dict["num_attention_heads"]
    assert Llama31ModelConfig.NUM_HIDDEN_LAYERS == hf_config_dict["num_hidden_layers"]
    assert Llama31ModelConfig.HIDDEN_SIZE == hf_config_dict["hidden_size"]
    assert Llama31ModelConfig.INTERMEDIATE_SIZE == hf_config_dict["intermediate_size"]


def test_from_hf_config(hf_config_dict):
    cfg = Llama31Config.from_hf_config(hf_config_dict)
    assert cfg.hidden_size == 4096
    assert cfg.num_attention_heads == 32
    assert cfg.num_key_value_heads == 8
    assert cfg.head_dim == 128
    assert cfg.num_hidden_layers == 32
    assert cfg.vocab_size == 128256
    assert cfg.rope_theta == 500000.0
    assert cfg.rope_scaling.factor == 8.0
    assert cfg.rope_scaling.rope_type == "llama3"
    assert cfg.attention_bias is False
    assert cfg.mlp_bias is False
    assert cfg.tie_word_embeddings is False


def test_gqa_ratio():
    cfg = Llama31Config()
    assert cfg.num_attention_heads // cfg.num_key_value_heads == 4
