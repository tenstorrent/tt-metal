# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only contracts for the Galaxy Attention2D collective adapters.

Two behaviours are pinned here because they decide tensor *shape*, which no
hardware test can recover from once it is wrong:

- concatenated physical-batch-32 prefill splits the reduced QKV projection into
  one row per user and merges the attention output back into one token stream;
- the column user selector turns column-replicated decode logits into the
  per-column user slice ``Sampling2D`` consumes.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import ttnn
from models.common.models.galaxy import collectives as galaxy_collectives
from models.common.models.galaxy.collectives import GalaxyAttentionCollectives, GalaxyColumnUserSelector
from models.common.models.galaxy.recipes import GALAXY_MESH_SHAPE, GalaxyDenseGeometry
from models.common.modules.attention.attention_2d import (
    PrefillAttentionMode,
    PrefillCollectiveMode,
    PrefillRecipeIdentity,
    PrefillRowMode,
)

LLAMA = dict(dim=8192, hidden_dim=28672, n_heads=64, n_kv_heads=8, head_dim=128, vocab_size=128256)


class _Tensor:
    """A shape-only stand-in for a TTNN tensor.

    ``sharded`` matters for the decode output reduction: ``all_reduce_async``
    rejects an interleaved input, so the collective width-shards it first, and
    the default here models the interleaved ``wo`` output that really arrives.
    """

    def __init__(self, name, shape, sharded=False, dtype=None):
        self.name = name
        self.shape = tuple(shape)
        self.sharded = sharded
        self.dtype = ttnn.bfloat16 if dtype is None else dtype

    def memory_config(self):
        return SimpleNamespace(is_sharded=lambda: self.sharded)

    def is_allocated(self):
        return True

    def deallocate(self, force=False):
        self.allocated = False


def _mesh():
    mesh = MagicMock(spec=ttnn.MeshDevice)
    mesh.shape = GALAXY_MESH_SHAPE
    mesh.get_num_devices.return_value = 32
    mesh.arch.return_value = ttnn.device.Arch.WORMHOLE_B0
    return mesh


def _identity(row_mode, length=128):
    return PrefillRecipeIdentity(length, row_mode, PrefillCollectiveMode.REGULAR, PrefillAttentionMode.REGULAR)


@pytest.fixture
def collectives(monkeypatch):
    """A collectives instance whose all-reduce is a shape-preserving stub."""

    events = []

    # The output reduction is mode-split, mirroring MLP2D: prefill keeps the
    # plain reduce_scatter + all_gather pair, decode uses all_reduce_async
    # against the keyed resource's persistent buffer. Decode cannot use the pair
    # because the Galaxy decode worker sub-device is non-contiguous and
    # ttnn.reduce_scatter places its workers from that sub-device's bounding box,
    # which crosses the x=4 prefetch sender column.
    def reduce_scatter(tensor, dim, **kwargs):
        events.append(("reduce_scatter", tensor.shape))
        return _Tensor("scattered", tensor.shape)

    def all_gather(tensor, dim, **kwargs):
        events.append(("all_gather", tensor.shape))
        return _Tensor("reduced", tensor.shape)

    def all_reduce_async(tensor, buffer_tensor, **kwargs):
        events.append(("all_reduce_async", tensor.shape))
        # The shared persistent buffer is allocated at the collective dtype, so
        # the reduction returns bfloat8_b and the collective has to cast back to
        # the activation dtype Attention2D contracts for.
        return _Tensor("reduced", tensor.shape, sharded=True, dtype=kwargs.get("dtype", ttnn.bfloat8_b))

    def interleaved_to_sharded(tensor, memory_config, **kwargs):
        events.append(("interleaved_to_sharded", tensor.shape))
        return _Tensor(
            f"sharded-{tensor.name}", tensor.shape, sharded=True, dtype=kwargs.get("output_dtype") or tensor.dtype
        )

    def sharded_to_interleaved(tensor, memory_config, **kwargs):
        events.append(("sharded_to_interleaved", tensor.shape))
        return _Tensor(f"interleaved-{tensor.name}", tensor.shape, dtype=kwargs.get("output_dtype") or tensor.dtype)

    def reshape(tensor, shape):
        events.append(("reshape", tuple(shape)))
        return _Tensor(f"view-{tensor.name}", tuple(shape))

    monkeypatch.setattr(galaxy_collectives.ttnn, "reduce_scatter", reduce_scatter)
    monkeypatch.setattr(galaxy_collectives.ttnn, "all_gather", all_gather)
    monkeypatch.setattr(galaxy_collectives.ttnn.experimental, "all_reduce_async", all_reduce_async)
    monkeypatch.setattr(galaxy_collectives.ttnn, "interleaved_to_sharded", interleaved_to_sharded)
    monkeypatch.setattr(galaxy_collectives.ttnn, "sharded_to_interleaved", sharded_to_interleaved)
    monkeypatch.setattr(galaxy_collectives.ttnn, "reshape", reshape)
    monkeypatch.setattr(galaxy_collectives.ttnn, "Shape", lambda value: tuple(value))
    monkeypatch.setattr(
        galaxy_collectives,
        "select_galaxy_resource",
        lambda *args, **kwargs: SimpleNamespace(
            num_links=1,
            topology="ring",
            persistent_output_buffers=(_Tensor("buffer", (1,)),),
            key=SimpleNamespace(operation="all_reduce", cluster_axis=0, geometry=(), sequence_key=None),
        ),
    )
    resources = SimpleNamespace(
        context=lambda mode: SimpleNamespace(
            worker_sub_device_id="worker",
            next_semaphore_handles=lambda *args, **kwargs: ["sem0", "sem1", "sem2"],
            next_barrier_semaphore_handle=lambda *args, **kwargs: "barrier",
        ),
        synchronize=lambda mode: events.append(("synchronize", mode)),
    )
    instance = GalaxyAttentionCollectives(
        resources=resources,
        mesh_device=_mesh(),
        geometry=GalaxyDenseGeometry(**LLAMA, max_seq_len=2048, prefill_sequence_lengths=(128,)),
        decode_placements=SimpleNamespace(residual_memcfg="residual", attention_qkv_reduced_memcfg="qkv-reduced"),
    )
    instance.events = events
    return instance


# ---------------------------------------------------------------------------
# Concatenated prefill row handling
# ---------------------------------------------------------------------------


def test_concat32_qkv_reduction_returns_one_row_per_user(collectives):
    reduced = collectives.reduce_qkv(
        _Tensor("qkv", (1, 1, 32 * 128, 1280)), mode="prefill", recipe=_identity(PrefillRowMode.CONCAT_32)
    )

    assert reduced.shape == (32, 1, 128, 1280)


def test_concat32_output_reduction_merges_the_rows_before_reducing(collectives):
    """The residual stream is one token stream, whatever the prefill row mode."""

    output = collectives.reduce_output(
        _Tensor("wo", (32, 1, 128, 2048)), mode="prefill", recipe=_identity(PrefillRowMode.CONCAT_32)
    )

    assert output.shape == (1, 1, 32 * 128, 2048)
    # The merge happens first, so the collective sees the token-stream geometry
    # its resource key was registered for.
    assert ("reduce_scatter", (1, 1, 32 * 128, 2048)) in collectives.events


@pytest.mark.parametrize("recipe", [None, _identity(PrefillRowMode.SINGLE_ROW)])
def test_single_row_prefill_and_decode_are_never_reshaped(collectives, recipe):
    reduced = collectives.reduce_qkv(_Tensor("qkv", (1, 1, 128, 1280)), mode="prefill", recipe=recipe)
    output = collectives.reduce_output(_Tensor("wo", (1, 1, 128, 2048)), mode="prefill", recipe=recipe)

    assert reduced.shape == (1, 1, 128, 1280)
    assert output.shape == (1, 1, 128, 2048)
    assert not [event for event in collectives.events if event[0] == "reshape"]


def test_a_token_stream_that_does_not_split_into_32_rows_fails_closed(collectives):
    with pytest.raises(ValueError, match="concat-32 prefill needs 32 equal rows"):
        collectives.reduce_qkv(
            _Tensor("qkv", (1, 1, 130, 1280)), mode="prefill", recipe=_identity(PrefillRowMode.CONCAT_32)
        )


# ---------------------------------------------------------------------------
# Column user selection
# ---------------------------------------------------------------------------


@pytest.fixture
def selector(monkeypatch):
    sources = []

    def from_torch(tensor, **kwargs):
        """Model the shard: a mapped dimension is divided by its mesh extent."""

        sources.append((tensor, kwargs))
        shape = list(tensor.shape)
        for extent, dim in zip(GALAXY_MESH_SHAPE, kwargs["mesh_mapper"]["dims"]):
            if dim is not None:
                shape[dim] //= extent
        return _Tensor("selector", tuple(shape))

    def matmul(left, right, **kwargs):
        return _Tensor("selected", (*right.shape[:-2], left.shape[-2], right.shape[-1]))

    monkeypatch.setattr(galaxy_collectives.ttnn, "from_torch", from_torch)
    monkeypatch.setattr(galaxy_collectives.ttnn, "matmul", matmul)
    monkeypatch.setattr(galaxy_collectives.ttnn, "ShardTensor2dMesh", lambda *args, **kwargs: kwargs)
    instance = GalaxyColumnUserSelector(_mesh())
    instance.sources = sources
    return instance


def test_the_selector_is_the_identity_sharded_over_the_user_axis(selector):
    selector.selector()
    source, kwargs = selector.sources[0]

    assert tuple(source.shape) == (1, 1, 32, 32)
    assert torch.equal(source.reshape(32, 32), torch.eye(32))
    # dims=(None, 2): rows replicate, columns shard the user axis, so column c
    # owns identity rows 8c..8c+7.
    assert kwargs["mesh_mapper"]["dims"] == (None, 2)


def test_the_selector_materializes_once(selector):
    first = selector.selector()

    assert selector.selector() is first
    assert len(selector.sources) == 1


def test_selection_returns_one_columns_users(selector):
    selected = selector(_Tensor("logits", (1, 1, 32, 16032)))

    assert selected.shape == (1, 1, 8, 16032)


@pytest.mark.parametrize("shape", [(1, 1, 8, 16032), (1, 32, 16032), (1, 1, 16, 16032)])
def test_selection_rejects_anything_but_the_full_physical_batch(selector, shape):
    with pytest.raises(ValueError, match=r"expects \[1, 1, 32, W\]"):
        selector(_Tensor("logits", shape))


def test_release_is_idempotent(selector):
    selector.selector()
    selector.release()
    selector.release()

    assert selector._selector is None


# ---------------------------------------------------------------------------
# Deallocation guard
# ---------------------------------------------------------------------------


def test_deallocating_an_already_released_tensor_is_a_no_op():
    """Ownership handoffs release the same tensor twice; that must be safe."""

    calls = []
    tensor = SimpleNamespace(is_allocated=lambda: False, deallocate=lambda force: calls.append(force))

    galaxy_collectives.deallocate_if_allocated(tensor)
    galaxy_collectives.deallocate_if_allocated(None)

    assert calls == []


def test_decode_output_reduction_uses_all_reduce_async_not_the_scatter_pair(collectives):
    """Decode may not use ttnn.reduce_scatter: the worker sub-device is split.

    ``ttnn.reduce_scatter`` derives its worker layout from
    ``worker_cores(TENSIX, sub_device_id).bounding_box()``. On WH Galaxy that
    bounding box spans the ``x=4`` prefetch sender column, so the program lands
    on cores the decode sub-device manager does not own and aborts with "Kernel
    group cores do not match sub device cores". ``all_reduce_async`` takes the
    keyed resource's persistent buffer and stays inside the partition.
    """

    output = collectives.reduce_output(_Tensor("wo", (1, 1, 32, 2048)), mode="decode", recipe=None)

    assert output.shape == (1, 1, 32, 2048)
    labels = [event[0] for event in collectives.events]
    assert "all_reduce_async" in labels, labels
    assert "reduce_scatter" not in labels, labels
    assert "all_gather" not in labels, labels
    # An interleaved wo output has to be width-sharded first; all_reduce_async
    # rejects TensorMemoryLayout::INTERLEAVED outright.
    assert "interleaved_to_sharded" in labels, labels
    # And the bfloat8_b result must come back as the activation dtype, or
    # Attention2D's own decode-output contract rejects it.
    assert output.dtype == ttnn.bfloat16, output.dtype
