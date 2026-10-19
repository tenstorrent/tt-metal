# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""An exact prompt and a padded one of the same bucket must not share a recording, on device.

A prompt whose length is already a whole number of pages runs a program that masks nothing and
slices its last token at an offset the kernel bakes in. A padded prompt of the same bucket runs a
different program: the tail is masked and the last token comes through a selector. Keyed on the
bucket alone the two share one entry, and the second one to arrive reaches capture with no
selectors allocated -- a host-to-device write inside `begin_trace_capture`, which is fatal.

Capture is what makes this reachable, so the walk below has to reach a first decode.
"""

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_t3k.tt.decoder_tp import native_mesh_shape
from models.demos.qwen38_27b_t3k.tt.generator import build_generator, configure_fabric

CAPACITY = 2048
LAYERS = [0, 3]  # layer_types alternates every fourth index: 0 is linear_attention, 3 full
BUCKET = 128
PADDED = 100  # Same bucket as BUCKET, but short of it, so the masked program is the one required.


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


def _prompt(gen, table, length, *, seed):
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(1, 1000, (1, length), generator=generator, dtype=torch.int64)
    gen.reset_recurrent_slots([0])
    gen.serving_prefill_tokens(
        tokens, page_table=table, kv_cache=gen.cache, prompt_lens=[length], start_pos=[0], slots=[0]
    )
    return tokens


def test_the_two_prompt_shapes_take_separate_buckets(harness):
    gen, _, _, table = harness
    _prompt(gen, table, BUCKET, seed=1)
    exact = gen.prefill_prepared
    assert exact["selectors"] is None, "an exact prompt must reach the program that bakes its slice in"
    _prompt(gen, table, PADDED, seed=2)
    padded = gen.prefill_prepared
    assert padded["key"] != exact["key"]
    assert padded["selectors"] is not None
    assert (padded["length"], padded["valid"]) == (BUCKET, PADDED)


def test_capture_after_both_shapes_allocates_nothing(harness):
    """The crash this pins is a write during capture, so the decode that captures is the test."""
    gen, _, cache, table = harness
    _prompt(gen, table, BUCKET, seed=3)
    _prompt(gen, table, PADDED, seed=4)
    assert gen.trace is None, "capture must still be ahead or this test measures nothing"
    gen.decode_forward(torch.tensor([5]), [PADDED], page_table=table, kv_cache=cache)
    assert gen.trace is not None
    # Both shapes must come back as replays rather than eager runs, which is what says each one
    # found a recording built against its own program.
    replays = gen.counters["prefill_replays"]
    _prompt(gen, table, BUCKET, seed=5)
    _prompt(gen, table, PADDED, seed=6)
    assert gen.counters["prefill_replays"] == replays + 2
