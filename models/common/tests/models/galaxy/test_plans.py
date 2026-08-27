# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only contracts for the Galaxy collective-resource plans and prefetch policy."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import ttnn
from models.common.models.galaxy.plans import (
    build_galaxy_resources_config,
    galaxy_decode_mode_plan,
    galaxy_prefill_mode_plan,
    select_galaxy_resource,
)
from models.common.models.galaxy.prefetch import (
    galaxy_address_memory_config,
    galaxy_dram_prefetch_start,
    galaxy_sender_receiver_mapping,
)
from models.common.models.galaxy.recipes import (
    GALAXY_MESH_SHAPE,
    GalaxyDenseGeometry,
    resolve_galaxy_decode_placements,
    worker_cores,
)

LLAMA = dict(dim=8192, hidden_dim=28672, n_heads=64, n_kv_heads=8, head_dim=128, vocab_size=128256)
QWEN = dict(dim=5120, hidden_dim=25600, n_heads=64, n_kv_heads=8, head_dim=128, vocab_size=151936)


@pytest.fixture(autouse=True)
def _host_mesh_mappers(monkeypatch):
    """Mesh mappers require a live device; plans only need placement identity."""

    monkeypatch.setattr(ttnn, "ShardTensor2dMesh", lambda *args, **kwargs: "shard-2d-mapper")
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda *args, **kwargs: "replicate-mapper")


def _mesh(shape=GALAXY_MESH_SHAPE, *, devices=32, arch=ttnn.device.Arch.WORMHOLE_B0):
    mesh = MagicMock(spec=ttnn.MeshDevice)
    mesh.shape = shape
    mesh.get_num_devices.return_value = devices
    mesh.arch.return_value = arch
    mesh.dram_grid_size.return_value = SimpleNamespace(x=12, y=1)
    mesh.compute_with_storage_grid_size.return_value = SimpleNamespace(x=7, y=10)
    return mesh


def _geometry(model: dict, lengths=(128, 2048)) -> GalaxyDenseGeometry:
    return GalaxyDenseGeometry(**model, max_seq_len=2048, prefill_sequence_lengths=lengths)


def _resources(model: dict, lengths=(128, 2048)):
    mesh = _mesh()
    geometry = _geometry(model, lengths)
    decode = resolve_galaxy_decode_placements(geometry, mesh)
    return geometry, build_galaxy_resources_config(mesh, geometry, decode)


def _keys(plan):
    return tuple((key.operation, key.cluster_axis, key.geometry, key.sequence_key) for key in (p.key for p in plan))


@pytest.mark.parametrize("model", [LLAMA, QWEN], ids=["llama-3.3-70b", "qwen3-32b"])
def test_decode_plan_covers_exactly_the_qualified_collectives(model):
    geometry, config = _resources(model)
    keys = set(_keys(config.decode.collectives))

    assert keys == {
        ("all_reduce_create_qkv_heads", 1, (1, 1, 32, geometry.local_qkv_size), 32),
        ("all_gather", 1, (1, 8, geometry.local_heads, geometry.head_dim), 8 * geometry.local_heads),
        ("reduce_scatter", 1, (1, 1, 32, geometry.local_hidden_dim), 32),
        ("all_gather", 1, (1, 1, 32, geometry.decode_reduce_scatter_width), 32),
        ("all_reduce", 0, (1, 1, 32, geometry.local_dim), 32),
        ("all_gather", 1, (1, 1, 32, 32), 32),
        # The decode LM head's column all-reduce. Missing from this set until now,
        # which is why the "exactly" in this test's name was not being enforced:
        # attempt 2 added the collective to the decode plan (it needs its own keyed
        # persistent buffer, sized for the logits rather than the residual stream)
        # and this expectation was not extended with it.
        #
        # The key carries `local_padded_vocab_size`, the width TTNN reports for the
        # logits - 16128 for Llama, 19200 for Qwen, both ring-exact since D-B19.
        ("all_reduce", 1, (1, 1, 32, geometry.local_padded_vocab_size), 32),
    }


@pytest.mark.parametrize("model", [LLAMA, QWEN], ids=["llama-3.3-70b", "qwen3-32b"])
def test_prefill_plan_is_sequence_keyed_and_unique(model):
    geometry, config = _resources(model)
    keys = _keys(config.prefill.collectives)

    assert len(keys) == len(set(keys)) == 8 * len(geometry.prefill_sequence_lengths)
    for length in geometry.prefill_sequence_lengths:
        leading = geometry.prefill_leading_shape(length)
        assert ("all_reduce", 1, (1, 1, length, geometry.local_qkv_size), length) in keys
        assert ("all_reduce", 0, (1, 1, length, geometry.local_dim), length) in keys
        assert ("reduce_scatter", 1, (*leading, geometry.local_hidden_dim), (length, "w1")) in keys
        assert ("reduce_scatter", 1, (*leading, geometry.local_hidden_dim), (length, "w3")) in keys
        assert ("all_gather", 1, (*leading, geometry.local_hidden_dim // 4), (length, "gated")) in keys
        assert ("reduce_scatter", 0, (1, 1, length, geometry.local_dim), (length, "final")) in keys
        assert ("all_gather", 0, (1, 1, length, geometry.local_dim // 8), (length, "final")) in keys
        assert ("all_gather", 1, (1, 1, length, 32), length) in keys


@pytest.mark.parametrize("model", [LLAMA, QWEN], ids=["llama-3.3-70b", "qwen3-32b"])
def test_batched_prefill_reuses_the_single_row_plan_at_its_token_count(model):
    """Concat-32 prefill is a single-row prefill of ``32 * length`` tokens.

    Its collectives are keyed by the token count TTNN observes, and the attention
    reshape into 32 rows happens between two collectives without changing the
    leading product, so no extra plan family is needed.
    """

    mesh = _mesh()
    geometry = GalaxyDenseGeometry(
        **model, max_seq_len=2048, prefill_sequence_lengths=(128,), batched_prefill_sequence_lengths=(128,)
    )
    decode = resolve_galaxy_decode_placements(geometry, mesh)
    config = build_galaxy_resources_config(mesh, geometry, decode)
    keys = _keys(config.prefill.collectives)
    tokens = 32 * 128

    assert len(keys) == len(set(keys)) == 8 * 2
    assert ("all_reduce", 1, (1, 1, tokens, geometry.local_qkv_size), tokens) in keys
    assert ("all_reduce", 0, (1, 1, tokens, geometry.local_dim), tokens) in keys
    assert ("all_gather", 1, (1, 1, tokens, 32), tokens) in keys


def test_a_batched_length_that_duplicates_a_single_row_length_is_not_registered_twice():
    """A 4096-token single-row recipe and 32x128 batched prefill share one key."""

    mesh = _mesh()
    geometry = GalaxyDenseGeometry(
        **LLAMA,
        max_seq_len=4096,
        prefill_sequence_lengths=(128, 4096),
        batched_prefill_sequence_lengths=(128,),
    )
    decode = resolve_galaxy_decode_placements(geometry, mesh)
    config = build_galaxy_resources_config(mesh, geometry, decode)
    keys = _keys(config.prefill.collectives)

    assert geometry.collective_prefill_token_counts == (128, 4096)
    assert len(keys) == len(set(keys)) == 8 * 2


def test_reduce_scatter_plans_carry_intermediate_buffers():
    _, config = _resources(LLAMA)
    for plan in (*config.decode.collectives, *config.prefill.collectives):
        if plan.key.operation == "reduce_scatter":
            assert plan.intermediate_output_specs
            assert plan.semaphores_per_slot == 3
        assert plan.persistent_output_specs


def test_selector_finds_the_plan_that_matches_the_observed_tensor():
    geometry, config = _resources(LLAMA)
    resources = {
        (plan.key.operation, plan.key.cluster_axis, plan.key.geometry, plan.key.sequence_key): plan
        for plan in config.decode.collectives
    }
    context = SimpleNamespace(resources=lambda *key: resources[key])

    tensor = SimpleNamespace(shape=(1, 1, 32, geometry.local_hidden_dim))
    assert select_galaxy_resource(context, "reduce_scatter", 1, tensor).key.operation == "reduce_scatter"
    # RMSNorm2D calls the selector with four positional arguments; MLP2D adds a
    # sequence key. Both arities must resolve.
    stats = (1, 1, 32, 32)
    assert select_galaxy_resource(context, "all_gather", 1, stats).key.geometry == stats
    assert select_galaxy_resource(context, "all_gather", 1, stats, None).key.geometry == stats


def test_selector_keys_prefill_stages_apart():
    geometry, config = _resources(LLAMA, lengths=(128,))
    resources = {
        (plan.key.operation, plan.key.cluster_axis, plan.key.geometry, plan.key.sequence_key): plan
        for plan in config.prefill.collectives
    }
    context = SimpleNamespace(resources=lambda *key: resources[key])
    hidden = SimpleNamespace(shape=(1, 1, 128, geometry.local_hidden_dim))

    assert select_galaxy_resource(context, "reduce_scatter", 1, hidden, "w1").key.sequence_key == (128, "w1")
    assert select_galaxy_resource(context, "reduce_scatter", 1, hidden, "w3").key.sequence_key == (128, "w3")
    with pytest.raises(KeyError):
        select_galaxy_resource(context, "reduce_scatter", 1, hidden, "w2")


def test_mode_plans_partition_prefill_and_decode_subdevices():
    mesh = _mesh()
    geometry = _geometry(LLAMA)
    decode_placements = resolve_galaxy_decode_placements(geometry, mesh)
    config = build_galaxy_resources_config(mesh, geometry, decode_placements)

    assert len(config.prefill.sub_devices) == 1
    assert config.prefill.worker_sub_device_id == ttnn.SubDeviceId(0)
    assert config.prefill.stall_group == (ttnn.SubDeviceId(0),)

    # Decode reserves a sender subdevice for the prefetch producer.
    assert len(config.decode.sub_devices) == 2
    assert config.decode.worker_sub_device_id == ttnn.SubDeviceId(1)
    assert config.decode.stall_group == (ttnn.SubDeviceId(1),)
    assert config.decode.semaphore_cores.num_cores() == worker_cores().num_cores()


def test_mode_plans_fail_closed_on_a_semaphore_set_narrower_than_the_workers():
    """Milestone A defect D3, promoted from a test-helper docstring into the plan.

    The generic async CCLs pick sender worker cores from the worker subdevice, so
    a global semaphore allocated on a narrower set leaves a sender polling L1
    that its own core never had reserved. That hangs indefinitely instead of
    failing, which is why this is checked rather than commented. Production
    already complied; nothing made it.
    """

    _, config = _resources(LLAMA, lengths=(128,))

    for plan in (config.prefill, config.decode):
        assert plan.worker_cores is not None, "a production plan must declare its worker subdevice"
        assert plan.worker_cores.subtract(plan.semaphore_cores).num_cores() == 0
        assert plan.allow_narrow_semaphore_cores is False

    narrow = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 3))})
    with pytest.raises(ValueError, match="semaphore_cores must cover the worker subdevice"):
        replace(config.decode, semaphore_cores=narrow)

    # The fused RMS all-gather binds its semaphore to a grid it owns, so it may
    # narrow - but it has to say so, because the plan cannot tell the two apart.
    deliberate = replace(config.decode, semaphore_cores=narrow, allow_narrow_semaphore_cores=True)
    assert deliberate.semaphore_cores is narrow


def test_mode_plan_builders_reject_duplicate_keys():
    _, config = _resources(LLAMA, lengths=(128,))
    duplicated = config.decode.collectives + (config.decode.collectives[0],)
    with pytest.raises(ValueError, match="duplicate Galaxy resource key"):
        galaxy_decode_mode_plan(duplicated)
    with pytest.raises(ValueError, match="duplicate Galaxy resource key"):
        galaxy_prefill_mode_plan(_mesh(), config.prefill.collectives + (config.prefill.collectives[0],))


def test_prefetch_mapping_pairs_every_sender_with_receivers():
    mapping = galaxy_sender_receiver_mapping()

    assert len(mapping) == 20
    assert len({(core.x, core.y) for core, _ in mapping}) == 20
    for _, receivers in mapping:
        assert receivers.num_cores() >= 1


def test_prefetch_address_placement_tracks_the_registered_weight_count():
    memory_config = galaxy_address_memory_config(400)

    assert tuple(memory_config.shard_spec.shape) == (1, 400)
    assert memory_config.shard_spec.grid.num_cores() == 12
    with pytest.raises(ValueError, match="must be positive"):
        galaxy_address_memory_config(0)


def test_dram_prefetch_producer_streams_one_layer_plus_addresses(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ttnn, "dram_prefetcher", lambda tensors, num_layers, global_cb: calls.append((tuple(tensors), num_layers))
    )
    start = galaxy_dram_prefetch_start(tensors_per_layer=5, num_layers=3)
    context = SimpleNamespace(
        weights=tuple(f"w{index}" for index in range(15)),
        weight_address_metadata="addresses",
        global_cb="global-cb",
    )

    start(context)

    assert calls == [(("w0", "w1", "w2", "w3", "w4", "addresses"), 3)]
    with pytest.raises(ValueError, match="must be positive"):
        galaxy_dram_prefetch_start(tensors_per_layer=0, num_layers=1)


def test_dram_prefetch_producer_fails_when_registration_is_short():
    start = galaxy_dram_prefetch_start(tensors_per_layer=5, num_layers=1)
    context = SimpleNamespace(weights=("w0",), weight_address_metadata="addresses", global_cb="global-cb")

    with pytest.raises(RuntimeError, match="registered weights per layer"):
        start(context)
