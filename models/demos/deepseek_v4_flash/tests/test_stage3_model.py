# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from models.demos.deepseek_v4_flash.converter import convert_hf_checkpoint
from models.demos.deepseek_v4_flash.synthetic import generate_tiny_hf_checkpoint
from models.demos.deepseek_v4_flash.ttnn_model import (
    ModelEmbeddingHeadWeights,
    embed_input_ids_host,
    load_model_embedding_head_weights,
    normalize_tiny_model_layer_ids,
    validate_model_embedding_head_weights,
    validate_model_input_ids,
)


def test_tiny_model_embedding_head_loading_and_host_embedding_lookup(tmp_path) -> None:
    source = generate_tiny_hf_checkpoint(tmp_path / "source", num_hidden_layers=3)
    output = convert_hf_checkpoint(source, tmp_path / "tt_preprocessed")

    weights = load_model_embedding_head_weights(output)

    assert weights.embed_weight.shape == (64, 32)
    assert weights.head_weight.shape == (64, 32)
    validate_model_embedding_head_weights(weights, hidden_size=32, vocab_size=64)

    input_ids = torch.tensor([[0, 1, 63, 2]], dtype=torch.int64)
    hidden_states = embed_input_ids_host(input_ids, weights.embed_weight)

    assert hidden_states.shape == (1, 4, 32)
    assert hidden_states.dtype == weights.embed_weight.dtype
    torch.testing.assert_close(hidden_states[0, 2], weights.embed_weight[63])


def test_tiny_model_api_validation_errors() -> None:
    weights = ModelEmbeddingHeadWeights(embed_weight=torch.zeros(64, 32), head_weight=torch.zeros(64, 32))
    validate_model_embedding_head_weights(weights, hidden_size=32, vocab_size=64)
    validate_model_input_ids(torch.zeros(1, 4, dtype=torch.int64), vocab_size=64)

    with pytest.raises(ValueError, match="Expected embed_weight shape"):
        validate_model_embedding_head_weights(
            ModelEmbeddingHeadWeights(embed_weight=torch.zeros(63, 32), head_weight=weights.head_weight),
            hidden_size=32,
            vocab_size=64,
        )
    with pytest.raises(ValueError, match="Expected head_weight shape"):
        validate_model_embedding_head_weights(
            ModelEmbeddingHeadWeights(embed_weight=weights.embed_weight, head_weight=torch.zeros(64, 16)),
            hidden_size=32,
            vocab_size=64,
        )
    with pytest.raises(ValueError, match="input_ids must have shape"):
        validate_model_input_ids(torch.zeros(2, 4, dtype=torch.int64), vocab_size=64)
    with pytest.raises(ValueError, match="at least one token"):
        validate_model_input_ids(torch.zeros(1, 0, dtype=torch.int64), vocab_size=64)
    with pytest.raises(ValueError, match="dtype must be int32 or int64"):
        validate_model_input_ids(torch.zeros(1, 4, dtype=torch.float32), vocab_size=64)
    with pytest.raises(ValueError, match=r"input_ids values must be in \[0, 64\)"):
        validate_model_input_ids(torch.tensor([[0, 64]], dtype=torch.int64), vocab_size=64)
    with pytest.raises(ValueError, match=r"input_ids values must be in \[0, 64\)"):
        embed_input_ids_host(torch.tensor([[-1]], dtype=torch.int64), weights.embed_weight)


def test_tiny_model_layer_id_normalization() -> None:
    assert normalize_tiny_model_layer_ids(layer=2, num_hidden_layers=3) == (2,)
    assert normalize_tiny_model_layer_ids(layer_ids=(0, 1), num_hidden_layers=3) == (0, 1)
    assert normalize_tiny_model_layer_ids(layer_ids=[0, 1, 2], num_hidden_layers=3) == (0, 1, 2)

    with pytest.raises(ValueError, match="layer_ids must be non-empty"):
        normalize_tiny_model_layer_ids(layer_ids=())
    with pytest.raises(ValueError, match="duplicates"):
        normalize_tiny_model_layer_ids(layer_ids=(0, 0))
    with pytest.raises(ValueError, match=r"layer id must be in \[0, 2\)"):
        normalize_tiny_model_layer_ids(layer_ids=(0, 2), num_hidden_layers=2)
    with pytest.raises(ValueError, match="Pass either layer or layer_ids"):
        normalize_tiny_model_layer_ids(layer=1, layer_ids=(0, 1), num_hidden_layers=3)
    with pytest.raises(TypeError, match="layer ids must be integers"):
        normalize_tiny_model_layer_ids(layer_ids=(True,))
