# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Milestone B step 7, area 5: long-context capacity accounting.

The 4K / 32K / 128K smokes are functional, not numerical, and the brief expects
memory and paging - not numerics - to be the limiting factor. It also asks for a
record of *where each one actually spends its capacity*. That record is
arithmetic over the resolved geometry, so it can be produced and checked without
a mesh, and it is what tells the next run whether a failure was a bug or a
capacity wall.

Everything here mirrors the configuration
``test_llama33_70b_galaxy_long_context_smoke`` builds: batch 1, one served slot,
2048-token chunks, one chunk of headroom past the prompt, one sink block per
idle slot.

The smoke itself has never run. Nothing here claims it passes; these tests say
what it will cost if it does.
"""

from __future__ import annotations

import pytest

from models.common.models.galaxy.direct_runner import GalaxyDirectRunner
from models.common.models.galaxy.kv_contract import GalaxyAttentionKVSpec, GalaxyPagedAttentionConfig
from models.common.models.galaxy.recipes import GALAXY_ROWS, GalaxyDenseGeometry
from models.common.tests.models.galaxy.step7_harness import (
    BLOCK_SIZE,
    GALAXY_PHYSICAL_BATCH,
    RecordingModel,
    patch_direct_runner,
)

_CONTEXTS = (4096, 32768, 131072)
_CHUNK = 2048

#: A bfloat8_b 32x32 tile is 1024 data bytes plus a 64-byte exponent section.
_BFP8_BYTES_PER_ELEMENT = 1088 / 1024
_BF16_BYTES_PER_ELEMENT = 2

#: Llama-3.3-70B on Galaxy: 8 KV heads over 8 mesh rows is one head per device.
_LLAMA = dict(n_kv_heads=8, head_dim=128, n_layers=80, name="llama33-70b")
#: Qwen3-32B: 8 KV heads, 128-wide, 64 layers.
_QWEN = dict(n_kv_heads=8, head_dim=128, n_layers=64, name="qwen3-32b")


def _served(context: int) -> int:
    """One chunk of headroom so the decode after a full prefill has a block."""

    return context + _CHUNK


def _pool(context: int, *, active_slots: int = 1) -> GalaxyPagedAttentionConfig:
    blocks_per_user = -(-_served(context) // BLOCK_SIZE)
    sinks = GALAXY_PHYSICAL_BATCH - active_slots
    return GalaxyPagedAttentionConfig(block_size=BLOCK_SIZE, max_num_blocks=blocks_per_user * active_slots + sinks)


def _kv_bytes_per_device(context: int, model: dict) -> float:
    """Bytes of paged KV one device holds, across every layer.

    The direct runner *replicates* the block pool: every device owns the whole
    pool and writes only the users its column serves, which is what makes a
    single page table valid on every device. So this is a per-device number and
    it does not shrink with the mesh.
    """

    pool = _pool(context).max_num_blocks
    local_kv_heads = model["n_kv_heads"] // GALAXY_ROWS
    elements = pool * local_kv_heads * BLOCK_SIZE * model["head_dim"]
    return elements * _BFP8_BYTES_PER_ELEMENT * 2 * model["n_layers"]


def _rope_table_len(context: int) -> int:
    return ((max(_served(context) * 2, 8192) + 127) // 128) * 128


# ---------------------------------------------------------------------------
# The geometry each smoke needs is legal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("context", _CONTEXTS, ids=["4k", "32k", "128k"])
def test_each_long_context_geometry_resolves(context):
    geometry = GalaxyDenseGeometry(
        dim=8192,
        hidden_dim=28672,
        n_heads=64,
        n_kv_heads=8,
        head_dim=128,
        vocab_size=128256,
        max_seq_len=_served(context),
        prefill_sequence_lengths=(_CHUNK,),
    )
    assert geometry.max_seq_len % geometry.chunk_alignment == 0
    assert geometry.prefill_sequence_lengths == (_CHUNK,)
    # One recipe only: a long context is reached by chunking, not by adding a
    # resolved recipe per length.
    assert geometry.collective_prefill_token_counts == (_CHUNK,)


@pytest.mark.parametrize("context", _CONTEXTS, ids=["4k", "32k", "128k"])
def test_the_block_pool_covers_the_served_context_and_every_idle_sink(context):
    pool = _pool(context)
    spec = GalaxyAttentionKVSpec(n_local_kv_heads=1, head_dim=128, kv_cache_dtype="bfp8", paged_attention_config=pool)
    blocks_per_user = -(-_served(context) // BLOCK_SIZE)
    assert pool.max_num_blocks == blocks_per_user + (GALAXY_PHYSICAL_BATCH - 1)
    assert blocks_per_user * BLOCK_SIZE >= _served(context)
    assert spec.local_cache_shape() == (pool.max_num_blocks, 1, BLOCK_SIZE, 128)


@pytest.mark.parametrize("context", _CONTEXTS, ids=["4k", "32k", "128k"])
def test_the_runner_accepts_the_batch_one_long_context_allocation(monkeypatch, context):
    patch_direct_runner(monkeypatch)
    pool = _pool(context)
    model = RecordingModel(
        max_seq_len=_served(context),
        prefill_sequence_lengths=(_CHUNK,),
        batched_prefill_sequence_lengths=(),
        max_num_blocks=pool.max_num_blocks,
    )
    runner = GalaxyDirectRunner(model, active_slots=1)
    assert runner.blocks_per_user * BLOCK_SIZE >= _served(context)
    assert runner.sink_blocks == GALAXY_PHYSICAL_BATCH - 1
    runner.open()
    try:
        rows = runner._page_table_rows()
        served = set(int(value) for value in rows[0].tolist())
        idle = set(int(value) for slot in range(1, GALAXY_PHYSICAL_BATCH) for value in rows[slot].tolist())
        assert not (served & idle), "an idle slot's sink landed in the served slot's context"
        assert len(served) == runner.blocks_per_user
    finally:
        runner.close()


@pytest.mark.parametrize("context", _CONTEXTS, ids=["4k", "32k", "128k"])
def test_a_pool_one_block_short_of_the_context_fails_closed(monkeypatch, context):
    patch_direct_runner(monkeypatch)
    pool = _pool(context)
    model = RecordingModel(
        max_seq_len=_served(context),
        prefill_sequence_lengths=(_CHUNK,),
        max_num_blocks=pool.max_num_blocks - 1,
    )
    with pytest.raises(ValueError, match="cannot hold max_seq_len"):
        GalaxyDirectRunner(model, active_slots=1)


# ---------------------------------------------------------------------------
# Where the capacity goes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("context", _CONTEXTS, ids=["4k", "32k", "128k"])
@pytest.mark.parametrize("model", [_LLAMA, _QWEN], ids=["llama", "qwen"])
def test_the_paged_kv_pool_is_the_dominant_per_device_cost(context, model):
    """Recorded numbers, not a gate. See REPORT.md area 5 for the table.

    A Wormhole device has 12 GB of DRAM. Both models' weights are roughly
    ``params / 32`` devices at bfloat8_b; the KV pool is on top of that and is
    *replicated*, so it does not shrink with the mesh.
    """

    kv = _kv_bytes_per_device(context, model)
    assert kv > 0
    # 128K is where this stops being negligible: it must be reported, and it
    # must still be under one device's DRAM on its own.
    assert kv < 12 * 1024**3, f"{model['name']} {context}: KV alone is {kv / 1024 ** 3:.2f} GiB per device"

    if context == 131072 and model is _LLAMA:
        # 4191 blocks * 1 head * 32 * 128 * 1.0625 B * 2 tensors * 80 layers.
        assert _pool(context).max_num_blocks == 4191
        assert kv / 1024**3 == pytest.approx(2.72, abs=0.05)


@pytest.mark.parametrize("context", _CONTEXTS, ids=["4k", "32k", "128k"])
def test_the_page_table_stays_small_even_at_the_longest_context(context):
    """The tables are int32 ``[32, blocks_per_user]``; 128K is still sub-megabyte."""

    blocks_per_user = -(-_served(context) // BLOCK_SIZE)
    aligned = ((blocks_per_user + 7) // 8) * 8
    prefill_bytes = GALAXY_PHYSICAL_BATCH * aligned * 4
    decode_bytes = GALAXY_PHYSICAL_BATCH * blocks_per_user * 4
    assert prefill_bytes < 1024**2, f"prefill page table is {prefill_bytes} bytes"
    assert decode_bytes <= prefill_bytes


@pytest.mark.parametrize("context", _CONTEXTS, ids=["4k", "32k", "128k"])
def test_the_rope_tables_grow_linearly_with_the_served_context(context):
    """Replicated bf16 ``[1, 1, table_len, head_dim]`` cos and sin, per device."""

    table_len = _rope_table_len(context)
    assert table_len == max(_served(context) * 2, 8192)
    per_table = table_len * 128 * _BF16_BYTES_PER_ELEMENT
    total = 2 * per_table
    assert total < 512 * 1024**2, f"rope tables are {total / 1024 ** 2:.0f} MiB per device"
    if context == 131072:
        assert table_len == 266240
        assert total / 1024**2 == pytest.approx(130.0, abs=1.0)


@pytest.mark.parametrize("context", _CONTEXTS, ids=["4k", "32k", "128k"])
def test_the_chunk_count_is_what_decides_how_long_a_smoke_takes(context):
    """A 128K prefill is 64 chunked prefill graphs, not one."""

    chunks = -(-context // _CHUNK)
    assert chunks == {4096: 2, 32768: 16, 131072: 64}[context]


def test_capacity_grows_linearly_in_context_and_is_reported_for_all_three():
    """One assertion that pins the whole recorded table at once."""

    measured = {context: _kv_bytes_per_device(context, _LLAMA) / 1024**3 for context in _CONTEXTS}
    assert measured[4096] == pytest.approx(0.13, abs=0.02)
    assert measured[32768] == pytest.approx(0.71, abs=0.03)
    assert measured[131072] == pytest.approx(2.72, abs=0.05)
    assert measured[4096] < measured[32768] < measured[131072]


# ---------------------------------------------------------------------------
# Chunked prefill at long-context scale, planned on the host
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("context", [4096, 32768], ids=["4k", "32k"])
def test_the_chunked_prefill_plan_walks_the_context_without_revisiting_a_block(monkeypatch, context):
    """Every chunk addresses its own block range; none overlaps another.

    128K is left out on purpose: its plan is 64 chunks of identical shape, and
    running it here would only re-measure the same arithmetic 64 times.
    """

    patch_direct_runner(monkeypatch)
    pool = _pool(context)
    model = RecordingModel(
        max_seq_len=_served(context),
        prefill_sequence_lengths=(_CHUNK,),
        max_num_blocks=pool.max_num_blocks,
    )
    runner = GalaxyDirectRunner(model, active_slots=1)
    runner.open()
    try:
        blocks = _CHUNK // BLOCK_SIZE
        full = runner._page_table_rows()
        seen: set[int] = set()
        for index in range(context // _CHUNK):
            chunk_rows = full[:, index * blocks : (index + 1) * blocks]
            served = set(int(value) for value in chunk_rows[0].tolist())
            assert not (served & seen), f"chunk {index} revisits blocks {sorted(served & seen)}"
            seen |= served
        assert len(seen) == (context // _CHUNK) * blocks
    finally:
        runner.close()


def test_a_decode_after_a_full_prefill_still_has_a_block_to_write_into(monkeypatch):
    """The headroom chunk is why ``served = context + chunk``."""

    patch_direct_runner(monkeypatch)
    context = 4096
    pool = _pool(context)
    model = RecordingModel(
        max_seq_len=_served(context),
        prefill_sequence_lengths=(_CHUNK,),
        max_num_blocks=pool.max_num_blocks,
    )
    runner = GalaxyDirectRunner(model, active_slots=1)
    runner.open()
    try:
        prefill_blocks = context // BLOCK_SIZE
        assert runner.blocks_per_user > prefill_blocks
        assert (runner.blocks_per_user - prefill_blocks) * BLOCK_SIZE >= _CHUNK
    finally:
        runner.close()


def test_the_served_context_must_stay_a_multiple_of_the_chunk_alignment():
    with pytest.raises(ValueError, match="must be a positive multiple of 128"):
        GalaxyDenseGeometry(
            dim=8192,
            hidden_dim=28672,
            n_heads=64,
            n_kv_heads=8,
            head_dim=128,
            vocab_size=128256,
            max_seq_len=131072 + 1,
            prefill_sequence_lengths=(_CHUNK,),
        )
