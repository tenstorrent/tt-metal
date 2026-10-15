# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Revisiting a bucket must not disturb the recordings the other buckets hold, on device.

Allocating a device buffer beside a live trace is what forces every recording to be released, and
a release costs the recapture of all of them -- measured at 16.9 s with four resident. A bucket
allocates once and keeps its buffers, so only its first sight pays that; a later visit re-runs it
eagerly and records it in place while the rest stay recorded.

The correctness claim underneath is that a recording replays against the same state an eager run
would have written, so the walk below checks state as well as counters.
"""

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_t3k.tt.decoder_tp import native_mesh_shape
from models.demos.qwen38_27b_t3k.tt.generator import MAX_RESIDENT_PREFILL, build_generator, configure_fabric

CAPACITY = 2048
LAYERS = [0, 3]  # layer_types alternates every fourth index: 0 is linear_attention, 3 full
# One more bucket than the set can record, so the walk forces an eviction and a return to it.
BUCKETS = [32 * (index + 1) for index in range(MAX_RESIDENT_PREFILL + 1)]


@pytest.fixture(scope="module")
def harness():
    from pathlib import Path

    root = Path(__file__).parents[1]
    configure_fabric()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*native_mesh_shape()), trace_region_size=200000000)
    try:
        gen = build_generator(root, mesh, precision_config=root / "config/precision.json", layer_indices=LAYERS)
        cache = gen.model.allocate_cache(batch_size=1, capacity=CAPACITY)
        table = torch.arange(cache.num_pages, dtype=torch.int32).reshape(1, -1)
        gen.bind_cache(cache, table)
        yield gen, mesh, cache, table
        gen.close()
    finally:
        ttnn.close_mesh_device(mesh)


def _state(gen, mesh, cache):
    out = []
    for layer, state in zip(gen.model.layers, cache.layers):
        if layer.kind != "linear_attention":
            continue
        for name in ("recurrent", "conv"):
            tensor = getattr(state, name)
            out.append(ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float().clone())
    assert out, "the harness must load a linear_attention layer or this test measures nothing"
    return out


def _prompt(gen, table, length, *, seed):
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(1, 1000, (1, length), generator=generator, dtype=torch.int64)
    gen.reset_recurrent_slots([0])
    gen.serving_prefill_tokens(
        tokens, page_table=table, kv_cache=gen.cache, prompt_lens=[length], start_pos=[0], slots=[0]
    )
    return tokens


def test_a_bucket_allocates_once_however_often_the_walk_returns_to_it(harness):
    gen, _, _, table = harness
    for length in BUCKETS:
        _prompt(gen, table, length, seed=length)
    allocated = gen.counters["prefill_bucket_allocations"]
    # The first bucket's recording is gone by now, but its buffers are not.
    for length in BUCKETS:
        _prompt(gen, table, length, seed=length)
    assert gen.counters["prefill_bucket_allocations"] == allocated == len(BUCKETS)


def test_an_evicted_bucket_comes_back_to_the_very_same_buffers(harness):
    """Identity is the point: a recording holds addresses, so reusing a bucket must reuse them.

    Reallocating is what would force every recording to be released, and releasing is what costs
    the recapture of all of them.
    """
    gen, _, _, table = harness
    for length in BUCKETS:
        _prompt(gen, table, length, seed=length)
    held = {key: (prepared["tokens"], prepared["positions"]) for key, prepared in gen.prefill_buckets.items()}
    assert len(held) == len(BUCKETS)
    evicted = [key for key in held if key not in gen.prefill_traced]
    assert evicted, "the walk must evict something or this test measures nothing"
    for length in BUCKETS:
        _prompt(gen, table, length, seed=length)
        prepared = gen.prefill_prepared
        assert (prepared["tokens"], prepared["positions"]) == held[prepared["key"]]


def test_a_bucket_reached_through_kept_buffers_leaves_what_a_fresh_one_leaves(harness):
    """Buffers that outlive their recording must still hold only what the current prompt put there.

    They are refreshed in place rather than reallocated, so a stale tail would be read as prompt.
    """
    gen, mesh, cache, table = harness
    length = BUCKETS[0]
    tokens = _prompt(gen, table, length, seed=7)
    fresh = _state(gen, mesh, cache)
    # Walk far enough to evict this bucket, so the return goes through buffers it already owns.
    for other in BUCKETS[1:]:
        _prompt(gen, table, other, seed=other)
    gen.reset_recurrent_slots([0])
    gen.serving_prefill_tokens(
        tokens, page_table=table, kv_cache=gen.cache, prompt_lens=[length], start_pos=[0], slots=[0]
    )
    for before, after in zip(fresh, _state(gen, mesh, cache)):
        scale = before.abs().max().clamp_min(1e-6)
        assert ((after - before).abs().max() / scale).item() < 0.02
