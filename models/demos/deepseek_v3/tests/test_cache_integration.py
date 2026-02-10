# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3.tt.rms_norm.rms_norm import RMSNorm
from models.demos.deepseek_v3.utils.cache import InMemoryCacheStorage, TensorCache
from models.demos.deepseek_v3.utils.weight_spec import (
    ModuleWeightSpec,
    WeightSpec,
    WeightSpecContext,
    create_weight_config_from_weight_spec,
)


@pytest.fixture
def sample_state_dict():
    """Create a sample state dict for testing."""
    return {
        "model.embedding.weight": torch.randn((128, 128), dtype=torch.bfloat16),
        "model.layers.0.weight0": torch.randn((32, 32), dtype=torch.bfloat16),
        "model.layers.0.weight1": torch.randn((32, 32), dtype=torch.bfloat16),
        "model.layers.1.weight0": torch.randn((32, 32), dtype=torch.bfloat16),
        "model.layers.1.weight1": torch.randn((32, 32), dtype=torch.bfloat16),
        "model.lmhead": torch.randn((64, 64), dtype=torch.bfloat16),
    }


@pytest.fixture
def sample_hf_config():
    return {"factor": 2, "hidden_size": 128}


class SimpleModule:
    @classmethod
    def create_weight_spec(cls, hf_config, mesh_shape: (int, int), context: WeightSpecContext) -> ModuleWeightSpec:
        return {
            "weight0": WeightSpec(
                name="weight0",
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                preprocessor=lambda t: t.reshape(context.get_reference_tensor("weight0").shape),
            ),
            "weight1": WeightSpec(
                name="weight1",
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                preprocessor=lambda t: t.reshape(context.get_reference_tensor("weight1").shape),
            ),
        }


class SimpleEmbeddingModule:
    @classmethod
    def create_weight_spec(cls, hf_config, mesh_shape: (int, int), context: WeightSpecContext) -> ModuleWeightSpec:
        return {
            "weight": WeightSpec(
                name="weight",
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                preprocessor=lambda t: t.reshape(context.get_reference_tensor("weight").shape),
            ),
        }


class SimpleModel:
    @classmethod
    def create_weight_spec(cls, hf_config, mesh_shape: (int, int), context: WeightSpecContext) -> ModuleWeightSpec:
        return {
            "embedding": SimpleEmbeddingModule.create_weight_spec(
                hf_config, mesh_shape, context.with_prefix("embedding")
            ),
            "layers.0": SimpleModule.create_weight_spec(hf_config, mesh_shape, context.with_prefix("layers.0")),
            "layers.1": SimpleModule.create_weight_spec(hf_config, mesh_shape, context.with_prefix("layers.1")),
            "lmhead": WeightSpec(
                name="lmhead",
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                preprocessor=lambda t: t.reshape(context.get_reference_tensor("lmhead").shape),
            ),
        }


def test_weight_spec_context_resolves_prefixed_names(sample_state_dict):
    context = WeightSpecContext(resolver=lambda key: sample_state_dict[key])
    layer_ctx = context.with_prefix("model").with_prefix("layers.0")
    embedding_ctx = context.with_prefix("model.embedding")

    assert layer_ctx.get_reference_tensor("weight0") is sample_state_dict["model.layers.0.weight0"]
    assert embedding_ctx.get_reference_tensor("weight") is sample_state_dict["model.embedding.weight"]


def test_cache_integration(sample_hf_config, sample_state_dict):
    mesh_shape = (8, 8)

    cache_storage = InMemoryCacheStorage()
    cache = TensorCache(sample_state_dict, sample_hf_config, cache_storage)

    context = WeightSpecContext(resolver=lambda key: sample_state_dict[key])
    single_layer_weight_spec = SimpleModule.create_weight_spec(
        sample_hf_config, mesh_shape, context.with_prefix("model.layers.0")
    )
    embedding_layer_weight_spec = SimpleEmbeddingModule.create_weight_spec(
        sample_hf_config, mesh_shape, context.with_prefix("model.embedding")
    )
    whole_model_weight_spec = SimpleModel.create_weight_spec(sample_hf_config, mesh_shape, context.with_prefix("model"))

    single_layer_weight_config = create_weight_config_from_weight_spec(
        single_layer_weight_spec, "model.layers.0", cache
    )
    embedding_layer_weight_config = create_weight_config_from_weight_spec(
        embedding_layer_weight_spec, "model.embedding", cache
    )
    whole_model_weight_config = create_weight_config_from_weight_spec(whole_model_weight_spec, "model", cache)

    # Sanity check that the weights are tensors
    assert all(isinstance(v, ttnn.Tensor) for v in single_layer_weight_config.values())
    assert all(isinstance(v, ttnn.Tensor) for v in embedding_layer_weight_config.values())
    assert isinstance(whole_model_weight_config["lmhead"], ttnn.Tensor)

    # The weights should match since they are resident in the same cache
    assert single_layer_weight_config["weight0"] is whole_model_weight_config["layers.0"]["weight0"]
    assert single_layer_weight_config["weight1"] is whole_model_weight_config["layers.0"]["weight1"]
    assert embedding_layer_weight_config["weight"] is whole_model_weight_config["embedding"]["weight"]

    # Check that the weights are the same as the original state dict
    assert torch.allclose(
        ttnn.to_torch(single_layer_weight_config["weight0"]), sample_state_dict["model.layers.0.weight0"]
    )
    assert torch.allclose(
        ttnn.to_torch(embedding_layer_weight_config["weight"]), sample_state_dict["model.embedding.weight"]
    )
    assert torch.allclose(ttnn.to_torch(whole_model_weight_config["lmhead"]), sample_state_dict["model.lmhead"])

    passing, pcc_message = comp_pcc(
        ttnn.to_torch(single_layer_weight_config["weight1"]), sample_state_dict["model.layers.0.weight1"], pcc=0.9999
    )
    assert passing, f"Weight1 does not match: {pcc_message}"


def test_create_weight_spec_for_real_modules(state_dict, hf_config):
    prefix = "model.layers.0.self_attn.kv_a_layernorm"
    weight_key = f"{prefix}.weight"
    mesh_shape = (4, 8)

    # Ensure the real state dict contains the expected RMSNorm weight
    assert (
        weight_key in state_dict
    ), f"State dict missing '{weight_key}'. Available keys (first 20): {list(state_dict.keys())[:20]}"
    reference_weight = state_dict[weight_key]
    assert reference_weight.dim() == 1, f"RMSNorm weight expected 1D, got shape {reference_weight.shape}"

    cache_storage = InMemoryCacheStorage()
    cache = TensorCache(state_dict, hf_config.to_dict(), cache_storage)

    # Get the weight spec for the RMSNorm module
    context = WeightSpecContext(resolver=lambda key: state_dict[key])
    weight_spec = RMSNorm.create_weight_spec(hf_config, mesh_shape, context.with_prefix(prefix))
    weight_config = create_weight_config_from_weight_spec(weight_spec, prefix, cache)

    assert set(weight_spec.keys()) == {"weight"}, f"Expected weight_spec keys {{'weight'}}, got {weight_spec.keys()}"
    assert isinstance(weight_spec["weight"], WeightSpec)
    assert set(weight_config.keys()) == {
        "weight"
    }, f"Expected weight_config keys {{'weight'}}, got {weight_config.keys()}"
    assert isinstance(weight_config["weight"], ttnn.Tensor)
    assert weight_config["weight"].storage_type() == ttnn.StorageType.HOST, "Weight should be on the host"

    # Cache returns the same tensor on second request (cache hit)
    weight_config_2 = create_weight_config_from_weight_spec(weight_spec, prefix, cache)
    assert weight_config["weight"] is weight_config_2["weight"], "Cache should return same tensor for same spec"
    assert weight_config_2["weight"].storage_type() == ttnn.StorageType.HOST, "Weight should be on the host"


def test_create_weight_spec_for_real_modules_with_device(state_dict, hf_config, mesh_device):
    prefix = "model.layers.0.self_attn.kv_a_layernorm"
    weight_key = f"{prefix}.weight"
    mesh_shape = (4, 8)

    # Ensure the real state dict contains the expected RMSNorm weight
    assert (
        weight_key in state_dict
    ), f"State dict missing '{weight_key}'. Available keys (first 20): {list(state_dict.keys())[:20]}"
    reference_weight = state_dict[weight_key]
    assert reference_weight.dim() == 1, f"RMSNorm weight expected 1D, got shape {reference_weight.shape}"

    cache_storage = InMemoryCacheStorage()
    cache = TensorCache(state_dict, hf_config.to_dict(), cache_storage)

    # Get the weight spec for the RMSNorm module (spec includes shard_dims; mesh mapper is derived per-tensor via get_mesh_mapper)
    context = WeightSpecContext(resolver=lambda key: state_dict[key])
    weight_spec = RMSNorm.create_weight_spec(hf_config, mesh_shape, context.with_prefix(prefix))
    weight_config = create_weight_config_from_weight_spec(weight_spec, prefix, cache, device=mesh_device)

    assert set(weight_spec.keys()) == {"weight"}, f"Expected weight_spec keys {{'weight'}}, got {weight_spec.keys()}"
    assert isinstance(weight_spec["weight"], WeightSpec)
    assert set(weight_config.keys()) == {
        "weight"
    }, f"Expected weight_config keys {{'weight'}}, got {weight_config.keys()}"
    assert isinstance(weight_config["weight"], ttnn.Tensor)
    assert weight_config["weight"].storage_type() == ttnn.StorageType.DEVICE, "Weight should be on the mesh device"

    # Cache returns the same tensor on second request (same spec => same derived mesh mapper config => cache hit)
    weight_config_2 = create_weight_config_from_weight_spec(weight_spec, prefix, cache, device=mesh_device)
    assert (
        weight_config["weight"] is weight_config_2["weight"]
    ), "Cache should return same tensor for same spec and mapper config"
    assert weight_config_2["weight"].storage_type() == ttnn.StorageType.DEVICE, "Weight should be on the mesh device"
