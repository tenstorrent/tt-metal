# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A prefill group served row by row must leave what the grouped kernel leaves.

Grouping is refused for a shape nothing compiled, because building one mid-request costs 8 s to
15 s of TTFT. That refusal silently swaps which kernel serves the request, so the two have to
agree: a concurrent wave takes the row path, a warmed one takes prefill_batch, and the 48
recurrent layers summarise whatever they are handed without complaining.

Two layers, one of each kind, so a divergence on either path shows.
"""

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_t3k.tt.decoder_tp import native_mesh_shape
from models.demos.qwen38_27b_t3k.tt.generator import batched_prefill_shape, build_generator, configure_fabric

BATCH = 4
CAPACITY = 1024
BUCKET = 128
LAYERS = [0, 3]  # layer_types alternates every fourth index: 0 is linear_attention, 3 full


@pytest.fixture(scope="module")
def harness():
    from pathlib import Path

    root = Path(__file__).parents[1]
    configure_fabric()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*native_mesh_shape()), trace_region_size=200000000)
    try:
        gen = build_generator(root, mesh, precision_config=root / "config/precision.json", layer_indices=LAYERS)
        gen.batched_prefill = True
        cache = gen.model.allocate_cache(batch_size=BATCH, capacity=CAPACITY)
        pages = cache.num_pages // BATCH
        table = torch.arange(BATCH * pages, dtype=torch.int32).reshape(BATCH, pages)
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


def _wave(gen, cache, table, tokens, rows):
    slots = list(range(rows))
    gen.reset_recurrent_slots(slots)
    before = gen.counters["batched_prefill_calls"]
    gen.prefill_forward(
        tokens[:rows],
        page_table=table,
        kv_cache=cache,
        prompt_lens=[BUCKET] * rows,
        slots=slots,
    )
    return gen.counters["batched_prefill_calls"] - before


def test_an_uncompiled_group_is_served_row_by_row(harness):
    gen, mesh, cache, table = harness
    gen.batched_prefill_shapes.clear()
    gen._batched_prefill_compiling = False
    prompt = torch.randint(1000, 5000, (BATCH, BUCKET), dtype=torch.int64)
    assert _wave(gen, cache, table, prompt, 3) == 0, "an unwarmed group reached prefill_batch"


def test_the_row_path_and_the_grouped_path_leave_the_same_state(harness):
    gen, mesh, cache, table = harness
    prompt = torch.randint(1000, 5000, (BATCH, BUCKET), dtype=torch.int64)

    gen.batched_prefill_shapes.clear()
    gen._batched_prefill_compiling = False
    assert _wave(gen, cache, table, prompt, 3) == 0
    rows = _state(gen, mesh, cache)

    # Warmup is the only thing allowed to build a group, so stand in for it here.
    gen._batched_prefill_compiling = True
    try:
        assert _wave(gen, cache, table, prompt, 3) == 1
    finally:
        gen._batched_prefill_compiling = False
    grouped = _state(gen, mesh, cache)

    scale = max(float(a.abs().max()) for a in grouped)
    drift = max(float((a - b).abs().max()) for a, b in zip(rows, grouped))
    assert drift / scale < 0.02, f"the row path drifted {drift} from the grouped one, scale {scale}"


def test_a_compiled_group_keeps_grouping_without_the_warmup_flag(harness):
    gen, mesh, cache, table = harness
    prompt = torch.randint(1000, 5000, (BATCH, BUCKET), dtype=torch.int64)
    shape = batched_prefill_shape([0, 1, 2], [0, 0, 0], [BUCKET] * 3, False)

    gen._batched_prefill_compiling = True
    try:
        _wave(gen, cache, table, prompt, 3)
    finally:
        gen._batched_prefill_compiling = False
    assert shape in gen.batched_prefill_shapes
    assert _wave(gen, cache, table, prompt, 3) == 1, "a compiled group stopped grouping"
    # A different occupancy is a different stack, so it has to fall back rather than compile.
    assert _wave(gen, cache, table, prompt, 2) == 0
