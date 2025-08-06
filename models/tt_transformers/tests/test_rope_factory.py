# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import json
import tempfile
from pathlib import Path

import pytest
import torch

from models.tt_transformers.tt.common import RopeScalingLlama3, RopeScalingYarn, rope_scaling_model_factory
from models.tt_transformers.tt.rope import (
    LlamaRotaryEmbedding,
    RotaryEmbedding,
    YarnRotaryEmbedding,
    rotary_embedding_factory,
)


@pytest.fixture
def rope_configs():
    """Fixture providing test configurations for different rope scaling types."""
    return {
        "no_scaling": {"description": "No rope scaling - should return basic RotaryEmbedding"},
        "llama3_scaling": {
            "rope_type": "llama3",
            "factor": 8.0,
            "original_max_position_embeddings": 8192,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
        },
        "yarn_scaling": {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 0.0,
        },
    }


@pytest.fixture
def temp_config_files(rope_configs):
    """Create temporary JSON config files for testing."""
    temp_files = {}
    temp_dir = tempfile.mkdtemp()

    try:
        for config_name, config_data in rope_configs.items():
            if config_name != "no_scaling":  # Skip the no_scaling case as it doesn't need a file
                file_path = Path(temp_dir) / f"{config_name}.json"
                with open(file_path, "w") as f:
                    json.dump(config_data, f, indent=2)
                temp_files[config_name] = str(file_path)

        yield temp_files
    finally:
        # Cleanup
        import shutil

        shutil.rmtree(temp_dir)


def test_rope_scaling_pydantic_validation(rope_configs):
    """Test that rope scaling configs can be validated with pydantic models."""

    # Test Llama3 config validation
    llama3_config = rope_configs["llama3_scaling"]
    llama3_model = rope_scaling_model_factory(llama3_config)
    assert isinstance(llama3_model, RopeScalingLlama3)
    assert llama3_model.factor == 8.0
    assert llama3_model.original_max_position_embeddings == 8192
    assert llama3_model.low_freq_factor == 1.0
    assert llama3_model.high_freq_factor == 4.0

    # Test Yarn config validation
    yarn_config = rope_configs["yarn_scaling"]
    yarn_model = rope_scaling_model_factory(yarn_config)
    assert isinstance(yarn_model, RopeScalingYarn)
    assert yarn_model.factor == 4.0
    assert yarn_model.original_max_position_embeddings == 4096
    assert yarn_model.beta_fast == 32
    assert yarn_model.beta_slow == 1
    assert yarn_model.mscale == 1.0
    assert yarn_model.mscale_all_dim == 0.0


def test_rope_factory_no_scaling():
    """Test rotary embedding factory with no scaling (basic RotaryEmbedding)."""
    dim = 128
    max_position_embeddings = 2048
    base = 10000.0

    embedding = rotary_embedding_factory(
        dim=dim, max_position_embeddings=max_position_embeddings, base=base, rope_scaling=None
    )

    assert isinstance(embedding, RotaryEmbedding)
    assert not isinstance(embedding, (YarnRotaryEmbedding, LlamaRotaryEmbedding))
    assert embedding.dim == dim
    assert embedding.max_position_embeddings == max_position_embeddings
    assert embedding.base == base


def test_rope_factory_llama3_scaling(rope_configs):
    """Test rotary embedding factory with Llama3 scaling."""
    dim = 128
    max_position_embeddings = 16384
    base = 10000.0

    # Create validated rope scaling model
    llama3_config = rope_configs["llama3_scaling"]
    rope_scaling = rope_scaling_model_factory(llama3_config)

    embedding = rotary_embedding_factory(
        dim=dim, max_position_embeddings=max_position_embeddings, base=base, rope_scaling=rope_scaling
    )

    assert isinstance(embedding, LlamaRotaryEmbedding)
    assert embedding.dim == dim
    assert embedding.max_position_embeddings == max_position_embeddings
    assert embedding.base == base
    assert embedding.scaling_factor == 8.0
    assert embedding.orig_context_len == 8192
    assert embedding.low_freq_factor == 1.0
    assert embedding.high_freq_factor == 4.0


def test_rope_factory_yarn_scaling(rope_configs):
    """Test rotary embedding factory with Yarn scaling."""
    dim = 64
    max_position_embeddings = 8192
    base = 10000.0

    # Create validated rope scaling model
    yarn_config = rope_configs["yarn_scaling"]
    rope_scaling = rope_scaling_model_factory(yarn_config)

    embedding = rotary_embedding_factory(
        dim=dim, max_position_embeddings=max_position_embeddings, base=base, rope_scaling=rope_scaling
    )

    assert isinstance(embedding, YarnRotaryEmbedding)
    assert embedding.dim == dim
    assert embedding.max_position_embeddings == max_position_embeddings
    assert embedding.base == base
    assert embedding.scaling_factor == 4.0
    assert embedding.original_max_position_embeddings == 4096
    assert embedding.beta_fast == 32
    assert embedding.beta_slow == 1
    assert embedding.mscale == 1.0
    assert embedding.mscale_all_dim == 0.0


def test_rope_factory_from_json_configs(temp_config_files):
    """Test loading rope configs from JSON files and using them with the factory."""
    dim = 128
    max_position_embeddings = 4096
    base = 10000.0

    # Test Llama3 config from JSON
    with open(temp_config_files["llama3_scaling"], "r") as f:
        llama3_data = json.load(f)

    llama3_rope_scaling = rope_scaling_model_factory(llama3_data)
    llama3_embedding = rotary_embedding_factory(
        dim=dim, max_position_embeddings=max_position_embeddings, base=base, rope_scaling=llama3_rope_scaling
    )
    assert isinstance(llama3_embedding, LlamaRotaryEmbedding)

    # Test Yarn config from JSON
    with open(temp_config_files["yarn_scaling"], "r") as f:
        yarn_data = json.load(f)

    yarn_rope_scaling = rope_scaling_model_factory(yarn_data)
    yarn_embedding = rotary_embedding_factory(
        dim=dim, max_position_embeddings=max_position_embeddings, base=base, rope_scaling=yarn_rope_scaling
    )
    assert isinstance(yarn_embedding, YarnRotaryEmbedding)


def test_rope_embeddings_forward_pass():
    """Test that created embeddings can perform forward passes."""
    dim = 64
    max_position_embeddings = 2048
    base = 10000.0
    batch_size = 2
    seq_len = 512
    num_heads = 8

    # Create test input tensor
    x = torch.randn(batch_size, num_heads, seq_len, dim)

    # Test basic RotaryEmbedding
    basic_embedding = rotary_embedding_factory(
        dim=dim, max_position_embeddings=max_position_embeddings, base=base, rope_scaling=None
    )

    cos, sin = basic_embedding.forward(x, seq_len)
    assert cos.shape == (1, 1, seq_len, dim)
    assert sin.shape == (1, 1, seq_len, dim)
    assert cos.dtype == x.dtype
    assert sin.dtype == x.dtype


def test_invalid_rope_scaling_type():
    """Test that invalid rope scaling types raise appropriate errors."""
    invalid_config = {"rope_type": "invalid_type", "factor": 2.0, "original_max_position_embeddings": 4096}

    with pytest.raises(ValueError, match="Invalid RoPE scaling type"):
        rope_scaling_model_factory(invalid_config)


@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("max_pos_emb", [2048, 4096])
@pytest.mark.parametrize("base", [10000.0, 500000.0])
def test_rope_factory_parameter_variations(dim, max_pos_emb, base):
    """Test rope factory with different parameter combinations."""
    embedding = rotary_embedding_factory(dim=dim, max_position_embeddings=max_pos_emb, base=base, rope_scaling=None)

    assert isinstance(embedding, RotaryEmbedding)
    assert embedding.dim == dim
    assert embedding.max_position_embeddings == max_pos_emb
    assert embedding.base == base


def test_rope_scaling_model_dump():
    """Test that rope scaling models can be dumped and used with factory."""
    config_data = {
        "rope_type": "llama3",
        "factor": 2.0,
        "original_max_position_embeddings": 2048,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
    }

    # Create validated model
    rope_scaling = rope_scaling_model_factory(config_data)

    # Test that model_dump works and can be used with factory
    dumped_config = rope_scaling.model_dump()

    embedding = rotary_embedding_factory(dim=128, max_position_embeddings=4096, base=10000.0, rope_scaling=rope_scaling)

    assert isinstance(embedding, LlamaRotaryEmbedding)
    # Verify that the dumped config was used correctly
    assert embedding.scaling_factor == 2.0
    assert embedding.orig_context_len == 2048


def test_rope_factory_with_dedicated_config_files():
    """Test rotary embedding factory with dedicated config files."""

    # Get the test directory path
    test_dir = Path(__file__).parent
    rope_configs_dir = test_dir / "rope_configs"

    dim = 128
    max_position_embeddings = 8192
    base = 10000.0

    # Test Llama3 standard config
    llama3_config_path = rope_configs_dir / "llama3_rope_config.json"
    if llama3_config_path.exists():
        with open(llama3_config_path, "r") as f:
            llama3_data = json.load(f)

        rope_scaling = rope_scaling_model_factory(llama3_data)
        embedding = rotary_embedding_factory(
            dim=dim, max_position_embeddings=max_position_embeddings, base=base, rope_scaling=rope_scaling
        )

        assert isinstance(embedding, LlamaRotaryEmbedding)
        assert embedding.scaling_factor == 8.0
        assert embedding.orig_context_len == 8192

    # Test Yarn standard config
    yarn_config_path = rope_configs_dir / "yarn_rope_config.json"
    if yarn_config_path.exists():
        with open(yarn_config_path, "r") as f:
            yarn_data = json.load(f)

        rope_scaling = rope_scaling_model_factory(yarn_data)
        embedding = rotary_embedding_factory(
            dim=dim, max_position_embeddings=max_position_embeddings, base=base, rope_scaling=rope_scaling
        )

        assert isinstance(embedding, YarnRotaryEmbedding)
        assert embedding.scaling_factor == 4.0
        assert embedding.original_max_position_embeddings == 4096

    # Test Llama3 large context config
    llama3_large_path = rope_configs_dir / "llama3_large_context.json"
    if llama3_large_path.exists():
        with open(llama3_large_path, "r") as f:
            llama3_large_data = json.load(f)

        rope_scaling = rope_scaling_model_factory(llama3_large_data)
        embedding = rotary_embedding_factory(
            dim=dim, max_position_embeddings=65536, base=base, rope_scaling=rope_scaling  # Use larger max for this test
        )

        assert isinstance(embedding, LlamaRotaryEmbedding)
        assert embedding.scaling_factor == 16.0
        assert embedding.orig_context_len == 32768
        assert embedding.low_freq_factor == 2.0
        assert embedding.high_freq_factor == 8.0


def test_rope_embeddings_consistency():
    """Test that embeddings created from the same config produce consistent results."""
    config_data = {
        "rope_type": "llama3",
        "factor": 4.0,
        "original_max_position_embeddings": 4096,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
    }

    rope_scaling = rope_scaling_model_factory(config_data)

    # Create two embeddings with the same parameters
    embedding1 = rotary_embedding_factory(dim=64, max_position_embeddings=8192, base=10000.0, rope_scaling=rope_scaling)

    embedding2 = rotary_embedding_factory(dim=64, max_position_embeddings=8192, base=10000.0, rope_scaling=rope_scaling)

    # Both should be the same type
    assert type(embedding1) == type(embedding2)
    assert isinstance(embedding1, LlamaRotaryEmbedding)

    # Should have the same parameters
    assert embedding1.scaling_factor == embedding2.scaling_factor
    assert embedding1.orig_context_len == embedding2.orig_context_len
    assert embedding1.low_freq_factor == embedding2.low_freq_factor
    assert embedding1.high_freq_factor == embedding2.high_freq_factor
