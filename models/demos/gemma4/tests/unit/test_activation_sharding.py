# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from models.demos.gemma4.tt import activation_sharding


def test_activation_sharding_is_default_off(monkeypatch):
    monkeypatch.delenv("GEMMA4_SHARD_ACTIVATIONS", raising=False)
    policy = activation_sharding.ActivationSharding.from_env()
    assert not policy.enabled
    assert policy.cores_for(256) is None


def test_activation_sharding_core_selection_uses_minimum_tiles():
    policy = activation_sharding.ActivationSharding(enabled=True)
    assert policy.cores_for(256) == 2
    assert policy.shard_tiles(256) == activation_sharding.MIN_SHARD_TILES
    assert policy.cores_for(1024) == 8
    assert policy.cores_for(96) is None
    assert policy.cores_for(255) is None


def test_activation_sharding_environment_override(monkeypatch):
    monkeypatch.setenv("GEMMA4_SHARD_ACTIVATIONS", "1")
    monkeypatch.setenv("GEMMA4_SHARD_MIN_TILES", "2")
    policy = activation_sharding.ActivationSharding.from_env()
    assert policy.enabled
    assert policy.min_shard_tiles == 2
    assert policy.cores_for(256) == 4


def test_activation_sharding_spec_is_cached(monkeypatch):
    calls = []

    monkeypatch.setattr(activation_sharding.ttnn, "CoreGrid", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        activation_sharding.ttnn,
        "create_sharded_memory_config",
        lambda **kwargs: calls.append(kwargs) or SimpleNamespace(shard_spec=SimpleNamespace(shape=kwargs["shape"])),
    )
    policy = activation_sharding.ActivationSharding(enabled=True)
    assert policy.spec(256) is policy.spec(256)
    assert len(calls) == 1
    assert calls[0]["shape"] == (activation_sharding.ttnn.TILE_SIZE, 128)


def test_activation_sharding_only_applies_to_decode_rows(monkeypatch):
    policy = activation_sharding.ActivationSharding(enabled=True)
    marker = object()
    monkeypatch.setattr(policy, "spec", lambda dim: marker if dim == 256 else None)
    assert policy.applies(SimpleNamespace(shape=(1, 1, 1, 256)))
    assert policy.applies(SimpleNamespace(shape=(1, 1, 32, 256)))
    assert not policy.applies(SimpleNamespace(shape=(1, 1, 33, 256)))
    assert not policy.applies(SimpleNamespace(shape=(1, 1, 1, 255)))


def test_to_stream_like_aligns_binary_operands(monkeypatch):
    wanted = object()
    converted = object()
    calls = []
    other = SimpleNamespace(is_sharded=lambda: True, memory_config=lambda: wanted)
    tensor = SimpleNamespace(is_sharded=lambda: False)
    monkeypatch.setattr(
        activation_sharding.ttnn,
        "to_memory_config",
        lambda value, memory_config: calls.append((value, memory_config)) or converted,
    )
    policy = activation_sharding.ActivationSharding(enabled=True)
    assert policy.to_stream_like(tensor, other) is converted
    assert calls == [(tensor, wanted)]
