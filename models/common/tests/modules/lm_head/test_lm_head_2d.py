# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest
import torch

import ttnn
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.lm_head.lm_head_2d import LMHead2D, LMHead2DConfig, _load_input
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


class _Collective:
    cluster_axis = 1
    consumes_input = False
    returns_owned_output = True

    def __init__(self, mesh):
        self.mesh_device = mesh

    def __call__(self, tensor):
        return MagicMock(name="reduced")


@pytest.mark.parametrize(
    "dim,vocab_size,padded_vocab_size",
    [(8192, 128256, 128256), (5120, 151936, 152064)],
    ids=["llama", "qwen"],
)
def test_lm_head_2d_resolves_representative_geometry(dim, vocab_size, padded_vocab_size):
    mesh = _galaxy()
    weight = LazyWeight(source=_ShapeOnly(dim, padded_vocab_size), device=mesh, dtype=ttnn.bfloat8_b)
    module = LMHead2D([weight], vocab_size, _Collective(mesh))

    assert module.config.is_resolved()
    assert module.config.dim == dim
    assert module.config.vocab_size == vocab_size
    assert module.config.padded_vocab_size == padded_vocab_size
    assert module.config.max_batch_size == 32
    assert module.config.output_weights[0] is not weight
    assert module.config.output_weights[0]._value is None
    assert parse_shard_dims_from_mesh_mapper_config(module.config.output_weights[0].mesh_mapper_config) == [-1, -2]
    assert parse_shard_dims_from_mesh_mapper_config(module.config._invalid_logits_mask.mesh_mapper_config) == [-1]
    mask = module.config._invalid_logits_mask.source
    assert torch.all(mask[..., :vocab_size] == 0)
    if vocab_size < padded_vocab_size:
        assert torch.all(torch.isneginf(mask[..., vocab_size:]))
    assert not hasattr(LMHead2D, "from_model_args")


def test_lm_head_2d_supports_explicit_lazy_mode_weights_and_splits():
    mesh = _galaxy()
    decode = [
        LazyWeight(source=_ShapeOnly(8192, 64000), device=mesh),
        LazyWeight(source=_ShapeOnly(8192, 64256), device=mesh),
    ]
    prefill = [
        LazyWeight(source=_ShapeOnly(8192, 64000), device=mesh),
        LazyWeight(source=_ShapeOnly(8192, 64256), device=mesh),
    ]
    collective = _Collective(mesh)
    module = LMHead2D.from_config(
        LMHead2DConfig(
            decode,
            vocab_size=128256,
            decode_collective=collective,
            prefill_output_weights=prefill,
            prefill_collective=collective,
        )
    )
    assert len(module.config.output_weights) == 2
    assert len(module.config.prefill_output_weights) == 2
    assert [weight.padded_shape[1] // 8 for weight in module.config.output_weights] == [8000, 8032]
    assert all(weight._value is None for weight in module.config.output_weights)
    assert all(weight._value is None for weight in module.config.prefill_output_weights)


def test_lm_head_2d_config_is_immutable():
    config = LMHead2DConfig(
        [LazyWeight(source=_ShapeOnly(8192, 128256))],
        vocab_size=128256,
        decode_collective=None,
    )
    with pytest.raises(FrozenInstanceError):
        config.dim = 8192


@pytest.mark.parametrize(
    "shape,devices,arch,error",
    [
        ((4, 8), 32, ttnn.device.Arch.WORMHOLE_B0, r"shape \(8, 4\)"),
        ((8, 4), 16, ttnn.device.Arch.WORMHOLE_B0, "exactly 32"),
        ((8, 4), 32, ttnn.device.Arch.BLACKHOLE, "Wormhole only"),
    ],
)
def test_lm_head_2d_fails_closed_on_platform(shape, devices, arch, error):
    mesh = _galaxy(arch=arch, shape=shape, devices=devices)
    weight = LazyWeight(source=_ShapeOnly(8192, 128256), device=mesh)
    with pytest.raises(ValueError, match=error):
        LMHead2D([weight], 128256, _Collective(mesh))


def test_lm_head_2d_rejects_unpadded_or_mismatched_weights():
    mesh = _galaxy()
    with pytest.raises(ValueError, match="padded_vocab_size does not match"):
        LMHead2D.from_config(
            LMHead2DConfig(
                [LazyWeight(source=_ShapeOnly(5120, 151936), device=mesh)],
                151936,
                _Collective(mesh),
                padded_vocab_size=151936,
            )
        )

    decode = [LazyWeight(source=_ShapeOnly(8192, 128256), device=mesh)]
    prefill = [LazyWeight(source=_ShapeOnly(8192, 128512), device=mesh)]
    with pytest.raises(ValueError, match="same padded vocabulary"):
        LMHead2D.from_config(LMHead2DConfig(decode, 128256, _Collective(mesh), prefill_output_weights=prefill))


def test_lm_head_2d_rejects_missing_collective_and_cross_mesh_weight():
    mesh = _galaxy()
    weight = LazyWeight(source=_ShapeOnly(8192, 128256), device=mesh)
    with pytest.raises(ValueError, match="collective callable"):
        LMHead2D.from_config(LMHead2DConfig([weight], 128256, None))

    other_mesh = _galaxy()
    cross_mesh = LazyWeight(source=_ShapeOnly(8192, 128256), device=other_mesh)
    with pytest.raises(ValueError, match="different mesh"):
        LMHead2D.from_config(LMHead2DConfig([cross_mesh], 128256, _Collective(mesh), mesh_device=mesh))


def test_lm_head_2d_rejects_invalid_vocab_and_input_geometry():
    mesh = _galaxy()
    weight = LazyWeight(source=_ShapeOnly(8192, 128256), device=mesh)
    with pytest.raises(ValueError, match="vocab_size must be positive"):
        LMHead2D([weight], 0, _Collective(mesh))

    module = LMHead2D([weight], 128256, _Collective(mesh))
    with pytest.raises(ValueError, match="physical batch 32"):
        _load_input(
            LazyWeight(source=_ShapeOnly(1, 1, 16, 8192), device=mesh),
            module.config,
            module.config.decode_input_memcfg,
            mode="decode",
        )
    with pytest.raises(ValueError, match=r"\[N, C, S, 8192\]"):
        _load_input(
            LazyWeight(source=_ShapeOnly(1, 1, 128, 4096), device=mesh),
            module.config,
            module.config.prefill_input_memcfg,
            mode="prefill",
        )


def test_lm_head_2d_accepts_column_local_device_activation_width():
    mesh = _galaxy()
    module = LMHead2D([LazyWeight(source=_ShapeOnly(8192, 128256), device=mesh)], 128256, _Collective(mesh))
    # A device activation produced by the column-sharded residual stream carries
    # its own column shard (dim / 4), not the complete hidden dimension.
    activation = MagicMock(spec=ttnn.Tensor)
    activation.shape = (1, 1, 32, 2048)

    assert _load_input(activation, module.config, module.config.decode_input_memcfg, mode="decode") == (
        activation,
        False,
    )

    wrong = MagicMock(spec=ttnn.Tensor)
    wrong.shape = (1, 1, 32, 1024)
    with pytest.raises(ValueError, match=r"\[N, C, S, 2048\]"):
        _load_input(wrong, module.config, module.config.decode_input_memcfg, mode="decode")


def test_lm_head_2d_rejects_foreign_or_materialized_lazy_input():
    mesh = _galaxy()
    module = LMHead2D([LazyWeight(source=_ShapeOnly(8192, 128256), device=mesh)], 128256, _Collective(mesh))
    foreign = LazyWeight(source=_ShapeOnly(1, 1, 32, 8192), device=_galaxy())
    with pytest.raises(ValueError, match="different mesh"):
        _load_input(foreign, module.config, module.config.decode_input_memcfg, mode="decode")

    materialized = LazyWeight(source=_ShapeOnly(1, 1, 32, 8192), device=mesh)
    materialized._value = object()
    with pytest.raises(ValueError, match="materialized LazyWeight"):
        _load_input(materialized, module.config, module.config.decode_input_memcfg, mode="decode")


def test_lm_head_2d_uses_per_axis_padding_and_exact_mask_metadata():
    mesh = _galaxy()
    weight = LazyWeight(source=_ShapeOnly(5120, 151936), device=mesh)
    module = LMHead2D([weight], 151936, _Collective(mesh))

    assert module.config.dim == 5120
    assert module.config.padded_vocab_size == 152064
    assert module.config.output_weights[0].padded_shape == (5120, 152064)
    assert module.config._invalid_logits_mask.source.shape[-1] == 152064
    assert module.config._invalid_logits_mask.padded_shape == (1, 1, 1, 152064)
    assert torch.all(torch.isneginf(module.config._invalid_logits_mask.source[..., 151936:]))


def test_lm_head_2d_default_mode_weights_alias_and_materialize_lazily(monkeypatch):
    mesh = _galaxy()
    module = LMHead2D([LazyWeight(source=_ShapeOnly(8192, 128256), device=mesh)], 128256, _Collective(mesh))
    decode_weight = module.config.output_weights[0]

    assert module.config.prefill_output_weights is module.config.output_weights
    assert module.config.prefill_output_weights[0] is decode_weight
    assert decode_weight._value is None

    monkeypatch.setattr(decode_weight, "get_device_weight", MagicMock(return_value=object()))
    monkeypatch.setattr(module.config._invalid_logits_mask, "get_device_weight", MagicMock(return_value=object()))
    module.load_device_weights("decode")

    assert decode_weight.get_device_weight.call_count == 1
    assert not module._prefill_weights_loaded


def test_lm_head_2d_rejects_padding_between_vocab_splits():
    mesh = _galaxy()
    weights = [
        LazyWeight(source=_ShapeOnly(5120, 64128), device=mesh),
        LazyWeight(source=_ShapeOnly(5120, 87808), device=mesh),
    ]

    with pytest.raises(ValueError, match="pad only the final"):
        LMHead2D(weights, 151936, _Collective(mesh))


def test_lm_head_2d_collective_contract_fails_closed():
    mesh = _galaxy()
    weight = LazyWeight(source=_ShapeOnly(8192, 128256), device=mesh)

    with pytest.raises(ValueError, match="resolved mesh_device"):
        LMHead2D([weight], 128256, lambda tensor: tensor)

    incomplete = MagicMock()
    incomplete.mesh_device = mesh
    incomplete.cluster_axis = 1
    incomplete.consumes_input = False
    with pytest.raises(ValueError, match="returns_owned_output=True"):
        LMHead2D([weight], 128256, incomplete)


def test_lm_head_2d_deallocates_projection_transients(monkeypatch):
    mesh = _galaxy()
    collective = _Collective(mesh)
    module = LMHead2D([LazyWeight(source=_ShapeOnly(8192, 128256), device=mesh)], 128256, collective)
    partial = object()
    reduced = object()
    result = object()
    deallocated = []

    monkeypatch.setattr(ttnn, "linear", lambda *args, **kwargs: partial)
    monkeypatch.setattr(ttnn, "add", lambda *args, **kwargs: result)
    monkeypatch.setattr(ttnn, "deallocate", deallocated.append)
    module.invalid_logits_mask = object()
    collective.__call__ = MagicMock(return_value=reduced)
    # Special-method lookup is class-based, so use a local callable with the same contract.
    resource = MagicMock(return_value=reduced)
    resource.mesh_device = mesh
    resource.cluster_axis = 1
    resource.consumes_input = False
    resource.returns_owned_output = True

    assert module._project(object(), [object()], [None], ttnn.bfloat16, object(), resource) is result
    assert deallocated == [partial, reduced]


def test_lm_head_2d_repeat_projection_and_collective_failure_cleanup(monkeypatch):
    mesh = _galaxy()
    module = LMHead2D([LazyWeight(source=_ShapeOnly(8192, 128256), device=mesh)], 128256, _Collective(mesh))
    partials = [object(), object(), object()]
    reduced = [object(), object()]
    results = [object(), object()]
    deallocated = []
    resource = MagicMock(side_effect=[*reduced, RuntimeError("collective failed")])
    resource.mesh_device = mesh
    resource.cluster_axis = 1
    resource.consumes_input = False
    resource.returns_owned_output = True
    monkeypatch.setattr(ttnn, "linear", MagicMock(side_effect=partials))
    monkeypatch.setattr(ttnn, "add", MagicMock(side_effect=results))
    monkeypatch.setattr(ttnn, "deallocate", deallocated.append)
    module.invalid_logits_mask = object()

    for result in results:
        assert module._project(object(), [object()], [None], ttnn.bfloat16, object(), resource) is result
    with pytest.raises(RuntimeError, match="collective failed"):
        module._project(object(), [object()], [None], ttnn.bfloat16, object(), resource)

    assert deallocated == [partials[0], reduced[0], partials[1], reduced[1], partials[2]]


def test_lm_head_2d_release_deallocates_aliased_mode_weights_once(monkeypatch):
    mesh = _galaxy()
    module = LMHead2D([LazyWeight(source=_ShapeOnly(8192, 128256), device=mesh)], 128256, _Collective(mesh))
    weight_value, mask_value = object(), object()
    module.config.output_weights[0]._value = weight_value
    module.config._invalid_logits_mask._value = mask_value
    module.output_weights = [weight_value]
    module.prefill_output_weights = [weight_value]
    module.invalid_logits_mask = mask_value
    module._decode_weights_loaded = module._prefill_weights_loaded = module._mask_loaded = True
    deallocated = []
    monkeypatch.setattr(ttnn, "deallocate", deallocated.append)

    module.release()
    module.release()

    assert deallocated == [weight_value, mask_value]
    assert not module._decode_weights_loaded
    assert not module._prefill_weights_loaded
    assert not module._mask_loaded


def test_lm_head_2d_release_deduplicates_shared_values_across_distinct_lazy_weights(monkeypatch):
    mesh = _galaxy()
    decode = [LazyWeight(source=_ShapeOnly(8192, 128256), device=mesh)]
    prefill = [LazyWeight(source=_ShapeOnly(8192, 128256), device=mesh)]
    module = LMHead2D.from_config(LMHead2DConfig(decode, 128256, _Collective(mesh), prefill_output_weights=prefill))
    shared_value, mask_value = object(), object()
    module.config.output_weights[0]._value = shared_value
    module.config.prefill_output_weights[0]._value = shared_value
    module.config._invalid_logits_mask._value = mask_value
    deallocated = []
    monkeypatch.setattr(ttnn, "deallocate", deallocated.append)

    module.release()

    assert deallocated == [shared_value, mask_value]
    assert module.config.output_weights[0]._value is None
    assert module.config.prefill_output_weights[0]._value is None
