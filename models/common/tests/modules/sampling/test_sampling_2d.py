# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-safe config and behavior tests for Sampling2D."""

from unittest.mock import MagicMock

import pytest
import torch

import ttnn
from models.common.modules.lazy_buffer import LazyBuffer
from models.common.modules.sampling.sampling_2d import Sampling2D, Sampling2DConfig, _resolve_sampling2d_config


def _galaxy_mesh():
    mesh = MagicMock()
    mesh.shape = (8, 4)
    mesh.get_num_devices.return_value = 32
    mesh.arch.return_value = ttnn.device.Arch.WORMHOLE_B0
    return mesh


def _sampler(vocab_size=96, padded_vocab_size=256):
    return Sampling2D(
        vocab_size,
        padded_vocab_size,
        _galaxy_mesh(),
        sub_core_grids=object(),
        sub_core_grid_topk=object(),
    )


def _config(**kwargs):
    kwargs.setdefault("sub_core_grids", object())
    kwargs.setdefault("sub_core_grid_topk", object())
    return Sampling2DConfig(**kwargs)


def test_config_resolves_galaxy_vocab_and_user_placement():
    config = _resolve_sampling2d_config(
        _config(vocab_size=128256, padded_vocab_size=128256, mesh_device=_galaxy_mesh())
    )

    assert config.is_resolved()
    assert config.vocab_shards == 8
    assert config.user_shards == 4
    assert config.users_per_shard == 8
    assert config.local_indices.source.shape == (1, 1, 32, 16032)
    assert config.index_offsets.source[0, 0, 0, 32].item() == 16032


def test_qwen_padded_vocab_is_tile_aligned_per_vocab_shard():
    config = _resolve_sampling2d_config(
        _config(vocab_size=151936, padded_vocab_size=152064, mesh_device=_galaxy_mesh())
    )

    assert config.local_indices.source.shape[-1] == 19008
    assert config.invalid_vocab_mask.source.shape == (1, 1, 32, 152064)
    assert torch.all(config.invalid_vocab_mask.source[..., :151936] == 0)
    assert torch.all(config.invalid_vocab_mask.source[..., 151936:] < 0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cluster_shape", (4, 8), "cluster_shape"),
        ("architecture", "blackhole", "Wormhole"),
        ("max_batch_size", 16, "physical batch 32"),
        ("sampling_all_gather_axis", 1, "vocabulary axis 0"),
    ],
)
def test_config_fails_closed_outside_canonical_galaxy(field, value, message, expect_error):
    kwargs = {field: value}
    with expect_error(ValueError, message):
        _resolve_sampling2d_config(_config(vocab_size=256, padded_vocab_size=256, mesh_device=_galaxy_mesh(), **kwargs))


def test_config_rejects_wrong_device_count(expect_error):
    mesh = _galaxy_mesh()
    mesh.get_num_devices.return_value = 31
    with expect_error(ValueError, "32 devices"):
        _resolve_sampling2d_config(_config(vocab_size=256, mesh_device=mesh))


def test_config_rejects_non_wormhole_mesh(expect_error):
    mesh = _galaxy_mesh()
    mesh.arch.return_value = ttnn.device.Arch.BLACKHOLE
    with expect_error(ValueError, "Wormhole"):
        _resolve_sampling2d_config(_config(vocab_size=256, mesh_device=mesh))


def test_config_rejects_unaligned_local_vocabulary(expect_error):
    with expect_error(ValueError, "tile aligned"):
        _resolve_sampling2d_config(_config(vocab_size=151936, padded_vocab_size=151936, mesh_device=_galaxy_mesh()))


def test_config_accepts_ring_exact_padding_and_rejects_more_than_one_shard(expect_error):
    """Padding beyond the minimum is legal; padding beyond a shard per row is not.

    This test used to require *exactly* the minimal Galaxy-aligned width, and that
    rejected the only width the Galaxy decode chain can run. `all_reduce_async`'s
    reduction kernel waits for a full shard on every output core, so the LM head's
    reduced logits must be an exact multiple of `cores * shard_width`; Qwen3-32B's
    minimal 152064 is not, and its ring-exact width is 153600. See D-B19.

    So the bound moved rather than disappeared: at least the minimum, and less than
    one extra vocabulary shard per mesh row.
    """

    # Qwen3-32B's ring-exact width, which the old rule rejected.
    _resolve_sampling2d_config(_config(vocab_size=151936, padded_vocab_size=153600, mesh_device=_galaxy_mesh()))
    # One tile past the minimum is legal too.
    _resolve_sampling2d_config(_config(vocab_size=151936, padded_vocab_size=152320, mesh_device=_galaxy_mesh()))
    # A whole extra shard per mesh row is a geometry mistake, not a padding choice.
    with expect_error(ValueError, "more than one"):
        _resolve_sampling2d_config(_config(vocab_size=151936, padded_vocab_size=154112, mesh_device=_galaxy_mesh()))
    # No case is written for "below the minimum": any width that is a multiple of
    # the vocabulary shards and at least `vocab_size` is at least the minimum by
    # construction, and the smaller ones are already rejected by the divisibility
    # and tile-alignment checks above.


def test_mutable_device_state_uses_lazy_buffers_with_2d_mappers():
    config = _sampler().config
    names = (
        "top_k_buffer",
        "top_p_buffer",
        "temperature_buffer",
        "seed_buffer",
        "user_ids",
        "index_offsets",
        "local_indices",
        "invalid_vocab_mask",
    )

    assert all(isinstance(getattr(config, name), LazyBuffer) for name in names)
    assert config.top_k_buffer.mesh_mapper.dims == (None, 0)
    assert config.local_indices.mesh_mapper.dims == (None, 2)
    assert config.invalid_vocab_mask.mesh_mapper.dims == (3, 2)


def test_slot_placement_is_contiguous_across_galaxy_columns():
    sampler = _sampler()

    assert sampler.slot_placement(0) == (0, 0)
    assert sampler.slot_placement(7) == (0, 7)
    assert sampler.slot_placement(8) == (1, 0)
    assert sampler.slot_placement(31) == (3, 7)


def test_prepare_call_preserves_each_per_call_value_without_module_state():
    sampler = _sampler()
    call = sampler.prepare_call(
        slot_ids=[1, 9, 17],
        top_k=[2, 5, 7],
        top_p=[0.25, 0.5, 0.9],
        temperature=[0.75, 1.0, 1.25],
        seed=[11, None, 33],
        forced_argmax=[False, True, False],
        update_buffers=False,
    )

    assert call.slot_ids == (1, 9, 17)
    assert call.top_k == (2, 5, 7)
    assert call.top_p == (0.25, 0.5, 0.9)
    assert call.temperature == (0.75, 1.0, 1.25)
    assert call.seed == (11, None, 33)
    assert call.forced_argmax == (False, True, False)
    assert not hasattr(sampler, "top_k")
    assert not hasattr(sampler, "seed")


def test_prepare_call_refreshes_lazy_sources_by_global_slot():
    sampler = _sampler()
    sampler.prepare_call(
        slot_ids=[2, 18],
        top_k=[4, 6],
        top_p=[0.4, 0.8],
        temperature=[0.5, 1.5],
        seed=[123, 456],
        forced_argmax=[False, True],
    )

    config = sampler.config
    assert config.top_k_buffer.source[2].item() == 4
    assert config.top_p_buffer.source[2].float().item() == pytest.approx(0.4, abs=0.01)
    # The device buffer carries the reciprocal temperature; ttnn.sampling multiplies by it.
    assert config.temperature_buffer.source[2].float().item() == pytest.approx(2.0, abs=0.01)
    assert config.top_k_buffer.source[18].item() == 1
    assert config.top_p_buffer.source[18].item() == 0
    assert config.temperature_buffer.source[18].item() == 1


def test_greedy_sampling_rejects_high_padded_logits():
    sampler = _sampler(vocab_size=96, padded_vocab_size=256)
    logits = torch.full((2, 256), -10.0)
    logits[0, 31] = 5.0
    logits[1, 95] = 4.0
    logits[:, 96:] = 1000.0

    tokens = sampler.sample_host(
        logits,
        slot_ids=[0, 31],
        top_k=[32, 32],
        top_p=[1.0, 1.0],
        temperature=[1.0, 0.0],
        forced_argmax=[True, False],
    )

    assert tokens.tolist() == [31, 95]
    assert torch.all(tokens < 96)


def test_seeded_sampling_is_repeatable_and_slot_stable():
    sampler = _sampler()
    generator = torch.Generator().manual_seed(4)
    logits = torch.randn(3, 256, generator=generator)
    kwargs = dict(top_k=16, top_p=0.9, temperature=0.8, seed=[101, 202, 303])

    original = sampler.sample_host(logits, slot_ids=[2, 10, 30], **kwargs)
    repeated = sampler.sample_host(logits, slot_ids=[2, 10, 30], **kwargs)
    permutation = [2, 0, 1]
    remapped = sampler.sample_host(
        logits[permutation],
        slot_ids=[30, 2, 10],
        top_k=16,
        top_p=0.9,
        temperature=0.8,
        seed=[303, 101, 202],
    )

    assert torch.equal(original, repeated)
    assert remapped.tolist() == original[permutation].tolist()


def test_device_temperature_buffer_holds_reciprocal_temperature():
    """The device temperature buffer carries 1/T, because ttnn.sampling multiplies by it.

    ``ttnn.sampling``'s ``temp`` argument is documented as ``1/T`` and its compute kernel
    applies ``values *= temp`` before the softmax. Writing the raw temperature here
    inverts the effect of every non-unit temperature on device while leaving ``T == 1.0``
    and the greedy path - where the buffer is forced to ``1.0`` - looking correct, so no
    greedy hardware test can catch it.
    """
    sampler = _sampler()
    sampler.prepare_call(
        slot_ids=[0, 1, 2, 3],
        top_k=8,
        top_p=0.9,
        temperature=[0.8, 2.0, 0.0, 1.0],
        forced_argmax=[False, False, False, True],
    )
    temperatures = sampler.config.temperature_buffer.source

    # 1/0.8 and 1/2.0 are both exact in bfloat16, so this is an equality, not a tolerance.
    assert temperatures[0].item() == 1.25
    assert temperatures[1].item() == 0.5
    # temperature=0.0 and forced_argmax both collapse to greedy, which uses 1.0.
    assert temperatures[2].item() == 1.0
    assert temperatures[3].item() == 1.0


def test_unseeded_sampling_uses_fresh_randomness(monkeypatch):
    sampler = _sampler()
    logits = torch.zeros(1, 256)
    random_values = iter((1, 2))
    monkeypatch.setattr("models.common.modules.sampling.sampling_2d.secrets.randbits", lambda _: next(random_values))

    first = sampler.sample_host(logits, slot_ids=[0], top_k=32, top_p=1.0, temperature=1.0)
    second = sampler.sample_host(logits, slot_ids=[0], top_k=32, top_p=1.0, temperature=1.0)

    assert first.item() < sampler.config.vocab_size
    assert second.item() < sampler.config.vocab_size
    assert first.item() != second.item()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"top_k": 0}, "top_k"),
        ({"top_p": 1.1}, "top_p"),
        ({"temperature": -0.1}, "temperature"),
        ({"seed": -1}, "seed"),
        ({"slot_ids": [0, 0]}, "unique"),
    ],
)
def test_invalid_per_call_values_are_rejected(kwargs, message, expect_error):
    sampler = _sampler()
    values = dict(slot_ids=[0], top_k=1, top_p=1.0, temperature=1.0, seed=None)
    values.update(kwargs)
    if values["slot_ids"] == [0, 0]:
        values.update(top_k=[1, 1], top_p=[1.0, 1.0], temperature=[1.0, 1.0], seed=[None, None])
    with expect_error(ValueError, message):
        sampler.prepare_call(**values, update_buffers=False)


def test_source_has_no_legacy_or_penalties_dependency():
    source = __import__("inspect").getsource(__import__("models.common.modules.sampling.sampling_2d", fromlist=["*"]))

    assert "sampling_1d" not in source
    assert "tt_sampling" not in source
    assert "Penalt" not in source


def test_config_requires_explicit_device_execution_resources(expect_error):
    with expect_error(ValueError, "explicit sub_core_grids"):
        _resolve_sampling2d_config(Sampling2DConfig(vocab_size=256, padded_vocab_size=256, mesh_device=_galaxy_mesh()))


def test_decode_deallocates_all_owned_transients_and_preserves_borrowed_tensors(monkeypatch):
    sampler = _sampler()
    sampler._device_buffers_loaded = True
    for name in ("_top_k", "_top_p", "_temperature", "_seeds", "_user_ids", "_index_offsets", "_local_indices"):
        setattr(sampler, name, object())
    sampler._invalid_vocab_mask = object()
    logits = MagicMock()
    logits.dtype = ttnn.bfloat8_b
    transient = iter(object() for _ in range(9))
    cast_logits = MagicMock()
    masked = MagicMock()
    local_values, local_indices = next(transient), next(transient)
    gathered_values, gathered_indices = next(transient), next(transient)
    cast_indices, offset_indices, global_indices = next(transient), next(transient), next(transient)
    result = object()
    deallocated = []

    monkeypatch.setattr(ttnn, "typecast", MagicMock(side_effect=[cast_logits, cast_indices]))
    monkeypatch.setattr(ttnn, "add", MagicMock(side_effect=[masked, offset_indices]))
    monkeypatch.setattr(ttnn, "topk", MagicMock(return_value=(local_values, local_indices)))
    monkeypatch.setattr(ttnn, "all_gather", MagicMock(side_effect=[gathered_values, gathered_indices]))
    monkeypatch.setattr(ttnn, "untilize", MagicMock(return_value=global_indices))
    monkeypatch.setattr(ttnn, "manual_seed", MagicMock())
    monkeypatch.setattr(ttnn, "sampling", MagicMock(return_value=result))
    monkeypatch.setattr(ttnn, "deallocate", deallocated.append)

    output = sampler.decode_forward(logits, top_k=1, top_p=1.0, temperature=1.0, slot_ids=[0])

    assert output is result
    assert set(deallocated) == {
        cast_logits,
        masked,
        local_values,
        local_indices,
        gathered_values,
        gathered_indices,
        cast_indices,
        offset_indices,
        global_indices,
    }
    assert logits not in deallocated
    assert result not in deallocated


def test_release_is_repeatable_and_clears_materialized_handles(monkeypatch):
    sampler = _sampler()
    buffers = [
        getattr(sampler.config, name)
        for name in (
            "top_k_buffer",
            "top_p_buffer",
            "temperature_buffer",
            "seed_buffer",
            "user_ids",
            "index_offsets",
            "local_indices",
            "invalid_vocab_mask",
        )
    ]
    releases = []
    for buffer in buffers:
        monkeypatch.setattr(buffer, "release", lambda buffer=buffer: releases.append(buffer))
    sampler._device_buffers_loaded = True
    for name in ("_top_k", "_top_p", "_temperature", "_seeds", "_user_ids", "_index_offsets", "_local_indices"):
        setattr(sampler, name, object())
    sampler._invalid_vocab_mask = object()

    sampler.release()
    sampler.release()

    assert len(releases) == 2 * len(buffers)
    assert not sampler._device_buffers_loaded
    assert not hasattr(sampler, "_top_k")
    assert not hasattr(sampler, "_invalid_vocab_mask")
