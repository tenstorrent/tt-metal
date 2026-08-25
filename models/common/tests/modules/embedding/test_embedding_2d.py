# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import ttnn
from models.common.modules.embedding.embedding_2d import Embedding2D, Embedding2DConfig
from models.common.modules.lazy_weight import LazyWeight
from models.common.tensor_utils import parse_shard_dims_from_mesh_mapper_config


class _ShapeOnly:
    def __init__(self, *shape):
        self.shape = shape


def _galaxy(arch=ttnn.device.Arch.WORMHOLE_B0, shape=(8, 4), devices=32):
    mesh = MagicMock()
    mesh.shape = shape
    mesh.get_num_devices.return_value = devices
    mesh.arch.return_value = arch
    return mesh


@pytest.mark.parametrize(
    "vocab_size,dim,prefill_len,scale",
    [(128256, 8192, 128, 1.0), (128256, 8192, 2048, 1.0), (151936, 5120, 512, 5120**0.5)],
    ids=["llama-prefill-128", "llama-prefill-2048", "qwen-prefill-512"],
)
def test_embedding_2d_resolves_representative_geometry(vocab_size, dim, prefill_len, scale):
    mesh = _galaxy()
    weight = LazyWeight(source=_ShapeOnly(vocab_size, dim), device=mesh)

    module = Embedding2D.from_config(Embedding2DConfig(weight, embed_scale=scale))

    assert module.config.is_resolved()
    assert module.config.vocab_size == vocab_size
    assert module.config.dim == dim
    assert module.config.max_batch_size == 32
    assert prefill_len <= 2048
    assert module.config.weights is not weight
    assert module.config.weights._value is None
    assert parse_shard_dims_from_mesh_mapper_config(module.config.weights.mesh_mapper_config) == [-1]
    assert "PlacementReplicate" in repr(module.config.weights.mesh_mapper_config)
    assert module.config.decode_output_dtype == ttnn.bfloat16
    assert module.config.prefill_output_dtype == ttnn.bfloat8_b
    assert not hasattr(Embedding2D, "from_model_args")


def test_embedding_2d_config_is_immutable():
    config = Embedding2DConfig(LazyWeight(source=_ShapeOnly(128256, 8192)))
    with pytest.raises(FrozenInstanceError):
        config.dim = 8192


def test_embedding_2d_normalizes_physical_weight_to_rank_four():
    mesh = _galaxy()
    module = Embedding2D(LazyWeight(source=torch.arange(64, dtype=torch.bfloat16).reshape(8, 8), device=mesh))

    assert tuple(module.config.weights.source.shape) == (1, 1, 8, 8)


@pytest.mark.parametrize(
    "shape,devices,arch,error",
    [
        ((4, 8), 32, ttnn.device.Arch.WORMHOLE_B0, r"shape \(8, 4\)"),
        ((8, 4), 31, ttnn.device.Arch.WORMHOLE_B0, "exactly 32"),
        ((8, 4), 32, ttnn.device.Arch.BLACKHOLE, "Wormhole only"),
    ],
)
def test_embedding_2d_fails_closed_on_platform(shape, devices, arch, error):
    mesh = _galaxy(arch=arch, shape=shape, devices=devices)
    weight = LazyWeight(source=_ShapeOnly(128256, 8192), device=mesh)
    with pytest.raises(ValueError, match=error):
        Embedding2D(weight)


def test_embedding_2d_rejects_invalid_shape_and_cross_mesh_weight():
    mesh = _galaxy()
    with pytest.raises(ValueError, match=r"\[vocab_size, dim\]"):
        Embedding2D(LazyWeight(source=_ShapeOnly(1, 1, 128256, 8192), device=mesh))

    other_mesh = _galaxy()
    weight = LazyWeight(source=_ShapeOnly(128256, 8192), device=other_mesh)
    with pytest.raises(ValueError, match="different mesh"):
        Embedding2D.from_config(Embedding2DConfig(weight, mesh_device=mesh))


def test_embedding_2d_deallocates_scaled_and_lazy_input_intermediates(monkeypatch):
    mesh = _galaxy()
    module = Embedding2D(LazyWeight(source=_ShapeOnly(128256, 8192), device=mesh), embed_scale=2.0)
    token_ids = LazyWeight(source=_ShapeOnly(32), device=mesh)
    tt_token_ids = object()
    embedding = SimpleNamespace(shape=(1, 32, 2048))
    scaled = object()
    deallocated = []

    monkeypatch.setattr(module, "load_device_weights", lambda: None)
    module.weights = object()
    loaded_ids = []

    def load_ids(value):
        loaded_ids.append(value)
        return tt_token_ids

    monkeypatch.setattr(LazyWeight, "get_device_weight", load_ids)
    monkeypatch.setattr(ttnn, "embedding", lambda *args, **kwargs: embedding)
    monkeypatch.setattr(ttnn, "reshape", lambda value, _shape: value)
    monkeypatch.setattr(ttnn, "multiply", lambda *args, **kwargs: scaled)
    monkeypatch.setattr(ttnn, "deallocate", deallocated.append)

    assert module.decode_forward(token_ids) is scaled
    assert deallocated == [tt_token_ids, embedding]
    assert loaded_ids[0].dtype == ttnn.uint32
    assert loaded_ids[0].mesh_mapper_config is None


def test_embedding_2d_release_is_repeatable(monkeypatch):
    mesh = _galaxy()
    module = Embedding2D(LazyWeight(source=_ShapeOnly(128256, 8192), device=mesh))
    value = object()
    module.config.weights._value = value
    module.weights = value
    module._device_weights_loaded = True
    deallocated = []
    monkeypatch.setattr(ttnn, "deallocate", deallocated.append)

    module.release()
    module.release()

    assert deallocated == [value]
    assert module.config.weights._value is None
    assert not module._device_weights_loaded
    assert not hasattr(module, "weights")
