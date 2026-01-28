# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from typing import Union

import pytest
import torch

import ttnn
from models.demos.deepseek_v3.utils.cache import InMemoryCacheStorage, TensorCache
from models.demos.deepseek_v3.utils.weight_spec import WeightSpec


@pytest.fixture
def sample_state_dict():
    """Create a sample state dict for testing."""
    return {
        "model.embedding.weight": torch.zeros((128, 128), dtype=torch.bfloat16),
        "model.layers.0.weight0": torch.randn((32, 32), dtype=torch.bfloat16),
        "model.layers.0.weight1": torch.randn((32, 32), dtype=torch.bfloat16),
        "model.layers.1.weight0": torch.randn((32, 32), dtype=torch.bfloat16),
        "model.layers.1.weight1": torch.randn((32, 32), dtype=torch.bfloat16),
        "model.lmhead": torch.randn((64, 64), dtype=torch.bfloat16),
    }


@pytest.fixture
def sample_hf_config():
    return {"factor": 2, "hidden_size": 128}


ModuleWeightSpec = dict[str, Union["ModuleWeightSpec", WeightSpec]]


class SimpleModule:
    @classmethod
    def convert_weights(cls, hf_config, mesh_shape) -> ModuleWeightSpec:
        return {
            "weight0": WeightSpec(
                name="weight0",
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                preprocessor=lambda t: t.reshape(32, 32),
            ),
            "weight1": WeightSpec(
                name="weight1",
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                preprocessor=lambda t: t.reshape(32, 32),
            ),
        }


class SimpleEmbeddingModule:
    @classmethod
    def convert_weights(cls, hf_config, mesh_shape) -> ModuleWeightSpec:
        return {
            "weight": WeightSpec(
                name="weight",
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                preprocessor=lambda t: t.reshape(128, 128),
            ),
        }


class SimpleModel:
    @classmethod
    def convert_weights(cls, hf_config, mesh_shape) -> ModuleWeightSpec:
        return {
            "embedding": SimpleEmbeddingModule.convert_weights(hf_config, mesh_shape),
            "layers.0": SimpleModule.convert_weights(hf_config, mesh_shape),
            "layers.1": SimpleModule.convert_weights(hf_config, mesh_shape),
            "lmhead": WeightSpec(
                name="lmhead",
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                preprocessor=lambda t: t.reshape(64, 64),
            ),
        }


def create_weight_config_from_weight_spec(
    module_weight_spec: ModuleWeightSpec, path: str, cache: TensorCache, delimiter: str = "."
):
    """
    Materialize a weight config from a weight spec.

    This will recursively materialize the weight config from the weight spec, but querying the cache for each weight spec with the fully qualified path (the key in the original state dict).
    """
    weight_config = {}
    for key, value in module_weight_spec.items():
        if isinstance(value, WeightSpec):
            # If its a weight spec we should load it from the cache using the fully qualified path (the key in the original state dict)
            name = path + delimiter + key
            tensor = cache.get_tensor(name, value.dtype, value.layout)
            weight_config[key] = tensor
        else:
            weight_config[key] = create_weight_config_from_weight_spec(value, path + delimiter + key, cache)
    return weight_config


def test_cache_integration(sample_hf_config, sample_state_dict):
    mesh_shape = (8, 8)

    cache_storage = InMemoryCacheStorage()
    cache = TensorCache(sample_state_dict, sample_hf_config, cache_storage)

    single_layer_weight_config = SimpleModule.convert_weights(sample_hf_config, mesh_shape)
    embedding_layer_weight_config = SimpleEmbeddingModule.convert_weights(sample_hf_config, mesh_shape)
    whole_model_weight_config = SimpleModel.convert_weights(sample_hf_config, mesh_shape)

    single_layer_weight_config = create_weight_config_from_weight_spec(
        single_layer_weight_config, "model.layers.0", cache
    )
    embedding_layer_weight_config = create_weight_config_from_weight_spec(
        embedding_layer_weight_config, "model.embedding", cache
    )
    whole_model_weight_config = create_weight_config_from_weight_spec(whole_model_weight_config, "model", cache)

    breakpoint()
