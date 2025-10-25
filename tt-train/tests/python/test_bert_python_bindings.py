# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive tests for BERT Python bindings.

Tests BERT model creation, configuration, and basic functionality through
the Python API, following TTML testing conventions.
"""

import numpy as np
import pytest
import os
import sys

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml  # noqa: E402


@pytest.fixture(autouse=True)
def reset_graph():
    """Reset the autograd graph before each test."""
    yield
    # Graph is automatically reset between tests


class TestBertConfig:
    """Test BertConfig Python bindings."""

    def test_config_creation_and_defaults(self):
        """Test BertConfig creation with default values."""
        config = ttml.models.bert.BertConfig()

        # Verify config has expected attributes
        assert hasattr(config, "vocab_size")
        assert hasattr(config, "embedding_dim")
        assert hasattr(config, "num_heads")
        assert hasattr(config, "num_blocks")

    def test_config_parameter_assignment(self):
        """Test setting all BertConfig parameters."""
        config = ttml.models.bert.BertConfig()

        # Set BERT-base-like configuration
        config.vocab_size = 30522
        config.max_sequence_length = 512
        config.embedding_dim = 768
        config.intermediate_size = 3072
        config.num_heads = 12
        config.num_blocks = 12
        config.dropout_prob = 0.1
        config.layer_norm_eps = 1e-12
        config.use_token_type_embeddings = True
        config.use_pooler = False

        # Verify values were set correctly
        assert config.vocab_size == 30522
        assert config.max_sequence_length == 512
        assert config.embedding_dim == 768
        assert config.intermediate_size == 3072
        assert config.num_heads == 12
        assert config.num_blocks == 12
        assert abs(config.dropout_prob - 0.1) < 1e-6
        assert abs(config.layer_norm_eps - 1e-12) < 1e-15
        assert config.use_token_type_embeddings is True
        assert config.use_pooler is False


class TestBertModelCreation:
    """Test BERT model creation through Python API."""

    def test_create_small_bert_model(self):
        """Test creating a small BERT model via create() function."""
        config = ttml.models.bert.BertConfig()
        config.vocab_size = 1000
        config.max_sequence_length = 128
        config.embedding_dim = 256
        config.intermediate_size = 512
        config.num_heads = 8
        config.num_blocks = 2
        config.dropout_prob = 0.0

        bert = ttml.models.bert.create(config)
        assert bert is not None

    def test_create_bert_via_constructor(self):
        """Test creating BERT model via Bert() constructor."""
        config = ttml.models.bert.BertConfig()
        config.vocab_size = 1000
        config.max_sequence_length = 128
        config.embedding_dim = 256
        config.intermediate_size = 512
        config.num_heads = 8
        config.num_blocks = 1
        config.dropout_prob = 0.0

        bert = ttml.models.bert.Bert(config)
        assert bert is not None

    def test_bert_parameters_accessible(self):
        """Test that BERT model parameters are accessible."""
        config = ttml.models.bert.BertConfig()
        config.vocab_size = 1000
        config.max_sequence_length = 128
        config.embedding_dim = 256
        config.intermediate_size = 512
        config.num_heads = 8
        config.num_blocks = 1
        config.dropout_prob = 0.0

        bert = ttml.models.bert.create(config)
        params = bert.parameters()

        # Verify parameters is a NamedParameters map
        assert isinstance(params, ttml.NamedParameters)

        # Check expected parameter names exist
        assert "bert/token_embeddings/weight" in params
        assert "bert/position_embeddings/weight" in params
        assert "bert/bert_block_0/attention/self_attention/qkv_linear/weight" in params


class TestBertWeightLoading:
    """Test BERT weight loading functionality."""

    def test_load_model_from_safetensors_nonexistent_file(self):
        """Test that loading from non-existent file raises error."""
        config = ttml.models.bert.BertConfig()
        config.vocab_size = 1000
        config.max_sequence_length = 128
        config.embedding_dim = 256
        config.intermediate_size = 512
        config.num_heads = 8
        config.num_blocks = 1
        config.dropout_prob = 0.0

        bert = ttml.models.bert.create(config)

        # Should raise exception for non-existent file
        with pytest.raises(Exception):
            bert.load_model_from_safetensors("/nonexistent/path/model.safetensors")


if __name__ == "__main__":
    # Run with: python test_bert_python_bindings.py
    pytest.main([__file__, "-v", "-s"])
