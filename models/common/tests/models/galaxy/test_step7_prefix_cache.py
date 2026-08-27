# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Milestone B step 7, area 3: prefix-cached and chunked prefill.

The gate itself - *prefix-cached output matches uncached execution* - is a
numerical claim and needs silicon. What this suite proves on the host is the
addressing underneath it, which is where the two known traps are:

* ``paged_fill_cache`` walks the chunk table from entry zero, so a chunk that
  starts at token ``c`` must be handed the slice of the slot's blocks that
  begins at block ``c / block_size`` - not the slot's whole row;
* chunked SDPA requires the page table's leading dimension to equal Q's batch,
  which is **1** for a single-row prefill. ``Attention2D`` slices the addressed
  user's row out for exactly this reason, and the row it picks must be the
  *prefix* user, not ``user_ids[0]``, whenever the two differ.

The interaction cases the brief asks for - a prefix-cached request followed by a
normal one, and a mix of both across slots - are exercised as *plans*: which
metadata each call carries, and whether one call leaves state that changes the
next. State that leaks between requests is host-visible; numerics are not.
"""

from __future__ import annotations

import pytest
import torch

from models.common.models.galaxy.direct_runner import GalaxyDirectRunner
from models.common.modules.attention.attention_2d import Attention2D, PrefillAttentionMode, PrefillMetadata
from models.common.tests.models.galaxy.step7_harness import (
    BLOCK_SIZE,
    GALAXY_PHYSICAL_BATCH,
    RecordingModel,
    ReplicateMapper,
    patch_compose,
    patch_direct_runner,
)
from models.common.tests.modules.attention.test_attention_2d import (  # noqa: F401 - host_ttnn is a fixture
    _config,
    _page_table,
    _paged_binding,
    _Tensor,
    host_ttnn,
)

_CHUNK = 128


def _open_runner(monkeypatch, *, max_seq_len=2048, prefill_lengths=(128, 256, 512)):
    recorder = patch_direct_runner(monkeypatch)
    blocks_per_user = max_seq_len // BLOCK_SIZE
    model = RecordingModel(
        max_seq_len=max_seq_len,
        prefill_sequence_lengths=prefill_lengths,
        batched_prefill_sequence_lengths=(128,),
        max_num_blocks=blocks_per_user * GALAXY_PHYSICAL_BATCH,
    )
    runner = GalaxyDirectRunner(model)
    patch_compose(monkeypatch, lambda tensor: torch.zeros(1, model.vocab_size))
    runner.open()
    return runner, model, recorder


# ---------------------------------------------------------------------------
# The chunk page table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_index", [1, 2, 7])
def test_a_chunk_table_starts_at_that_chunks_first_block_for_every_slot(monkeypatch, chunk_index):
    """``paged_fill_cache`` reads the chunk table from entry zero."""

    runner, _, recorder = _open_runner(monkeypatch)
    try:
        blocks = _CHUNK // BLOCK_SIZE
        before = len(recorder.staged)
        runner.stage_chunk_page_table(chunk_start=chunk_index * _CHUNK, length=_CHUNK)
        staged = recorder.staged[before:]
        assert len(staged) == 1
        table = staged[0]
        assert isinstance(table.mapper, ReplicateMapper)

        full = runner._page_table_rows()
        expected = full[:, chunk_index * blocks : (chunk_index + 1) * blocks]
        assert torch.equal(table.host[:, :blocks], expected)
        # Stick alignment pads to eight int32 entries with zeros only.
        assert table.host.shape[1] == 8
        assert torch.equal(table.host[:, blocks:], torch.zeros((GALAXY_PHYSICAL_BATCH, 8 - blocks), dtype=torch.int32))
    finally:
        runner.close()


def test_a_chunk_table_never_reaches_another_slots_blocks(monkeypatch):
    runner, _, recorder = _open_runner(monkeypatch)
    try:
        blocks = _CHUNK // BLOCK_SIZE
        before = len(recorder.staged)
        runner.stage_chunk_page_table(chunk_start=3 * _CHUNK, length=_CHUNK)
        table = recorder.staged[before].host[:, :blocks]
        owned = [set(int(value) for value in table[slot].tolist()) for slot in range(GALAXY_PHYSICAL_BATCH)]
        for slot, entries in enumerate(owned):
            for other in range(slot + 1, GALAXY_PHYSICAL_BATCH):
                assert not (entries & owned[other]), f"chunk table shares blocks between slots {slot} and {other}"
    finally:
        runner.close()


def test_an_unaligned_chunk_start_is_refused(monkeypatch):
    runner, _, _ = _open_runner(monkeypatch)
    try:
        with pytest.raises(ValueError, match="must be a multiple of block size"):
            runner.stage_chunk_page_table(chunk_start=BLOCK_SIZE // 2, length=_CHUNK)
    finally:
        runner.close()


def test_a_chunk_past_a_slots_allocation_is_refused(monkeypatch):
    runner, _, _ = _open_runner(monkeypatch)
    try:
        beyond = runner.blocks_per_user * BLOCK_SIZE
        with pytest.raises(ValueError, match="exceeds each slot's block allocation"):
            runner.stage_chunk_page_table(chunk_start=beyond, length=_CHUNK)
    finally:
        runner.close()


def test_chunked_prefill_needs_a_paged_cache(monkeypatch):
    recorder = patch_direct_runner(monkeypatch)
    model = RecordingModel(paged=False)
    runner = GalaxyDirectRunner(model)
    runner.open()
    try:
        with pytest.raises(RuntimeError, match="chunked prefill requires a paged KV cache"):
            runner.stage_chunk_page_table(chunk_start=0, length=_CHUNK)
    finally:
        runner.close()
    assert recorder.staged, "the contiguous cache still allocates"


# ---------------------------------------------------------------------------
# The chunked prefill plan
# ---------------------------------------------------------------------------


def test_the_first_chunk_is_an_ordinary_prefill_and_later_chunks_are_prefix_cached(monkeypatch):
    runner, model, _ = _open_runner(monkeypatch)
    try:
        runner.prefill_chunked(list(range(1, 4 * _CHUNK + 1)), slot=5, chunk_length=_CHUNK)

        assert len(model.prefill_calls) == 4
        first = model.prefill_calls[0]
        assert first["chunk_start"] is None
        assert first["chunk_page_table"] is None
        assert first["prefix_user_id"] is None
        assert first["user_ids"] == (5,)

        for index, call in enumerate(model.prefill_calls[1:], start=1):
            assert call["chunk_start"] == index * _CHUNK, f"chunk {index} declared the wrong start"
            assert call["chunk_page_table"] is not None
            assert call["prefix_user_id"] == 5
            assert call["user_ids"] == (5,)
            assert call["page_table"] is runner._prefill_page_table
    finally:
        runner.close()


def test_every_chunk_table_is_released_before_the_next_chunk_is_staged(monkeypatch):
    """A chunk table that outlives its chunk is a leak across a long context."""

    runner, model, recorder = _open_runner(monkeypatch)
    try:
        runner.prefill_chunked(list(range(1, 3 * _CHUNK + 1)), slot=0, chunk_length=_CHUNK)
        chunk_tables = [call["chunk_page_table"] for call in model.prefill_calls if call["chunk_page_table"]]
        assert len(chunk_tables) == 2
        assert all(table.is_allocated() is False for table in chunk_tables)
    finally:
        runner.close()


def test_a_prefix_cached_request_leaves_the_next_plain_request_unchanged(monkeypatch):
    """Interaction: prefix-cached, then normal."""

    runner, model, _ = _open_runner(monkeypatch)
    try:
        runner.prefill_chunked(list(range(1, 2 * _CHUNK + 1)), slot=0, chunk_length=_CHUNK)
        model.prefill_calls.clear()

        runner.prefill_row(list(range(1, _CHUNK + 1)), slot=1)

        assert len(model.prefill_calls) == 1
        call = model.prefill_calls[0]
        assert call["prefix_user_id"] is None
        assert call["chunk_start"] is None
        assert call["chunk_page_table"] is None
        assert call["user_ids"] == (1,)
        assert call["page_table"] is runner._prefill_page_table
    finally:
        runner.close()


def test_prefix_cached_and_plain_requests_mix_across_slots_without_sharing_state(monkeypatch):
    """Interaction: a mix of both in the same batch of requests."""

    runner, model, _ = _open_runner(monkeypatch)
    try:
        runner.prefill_row(list(range(1, _CHUNK + 1)), slot=0)
        runner.prefill_chunked(list(range(1, 3 * _CHUNK + 1)), slot=1, chunk_length=_CHUNK)
        runner.prefill_row(list(range(1, _CHUNK + 1)), slot=2)

        plain = [call for call in model.prefill_calls if call["prefix_user_id"] is None]
        cached = [call for call in model.prefill_calls if call["prefix_user_id"] is not None]
        assert [call["user_ids"] for call in plain] == [(0,), (1,), (2,)]
        assert all(call["prefix_user_id"] == 1 for call in cached)
        assert {call["user_ids"] for call in cached} == {(1,)}
        # Every call addressed exactly one slot: no request widened another's batch.
        assert all(len(call["user_ids"]) == 1 for call in model.prefill_calls)
    finally:
        runner.close()


def test_chunked_prefill_needs_a_positive_chunk_and_at_least_one_token(monkeypatch):
    runner, _, _ = _open_runner(monkeypatch)
    try:
        with pytest.raises(ValueError, match="positive chunk length"):
            runner.prefill_chunked([1, 2, 3], slot=0, chunk_length=0)
        with pytest.raises(ValueError, match="positive chunk length"):
            runner.prefill_chunked([], slot=0, chunk_length=_CHUNK)
    finally:
        runner.close()


# ---------------------------------------------------------------------------
# The single-row page-table slice chunked SDPA needs
# ---------------------------------------------------------------------------


def test_single_row_chunked_sdpa_reads_one_row_and_it_is_the_prefix_users(host_ttnn):
    """Q's batch is 1, so the table handed to chunked SDPA must have one row.

    ``prefix_user_id`` has to be one of ``user_ids`` - ``_validate_prefill``
    enforces it - so for a single-row prefill the addressed user and the prefix
    user are necessarily the same value. The slicing still has to happen.
    """

    model = Attention2D.from_config(_config())
    model.bind_kv_cache(_paged_binding(model))
    table = _page_table(rows=32)
    metadata = PrefillMetadata(
        128,
        (11,),
        page_table=table,
        chunk_page_table=_page_table(rows=32),
        chunk_start=128,
        prefix_user_id=11,
    )

    assert model._sdpa_page_table(metadata) is not table, "the whole 32-row table reached chunked SDPA"

    model.prefill_forward(_Tensor("x", dtype="act", placement="prefill-in"), "rot", metadata)
    sdpa = [kwargs for stage, kwargs in host_ttnn if stage == "sdpa-chunked"]
    assert len(sdpa) == 1
    assert sdpa[0]["page_table_tensor"] is not table


def test_a_prefix_user_outside_the_addressed_users_is_refused(host_ttnn):
    model = Attention2D.from_config(_config())
    model.bind_kv_cache(_paged_binding(model))
    with pytest.raises(ValueError, match="prefix_user_id must identify an active prefill user"):
        model.prefill_forward(
            _Tensor("x", dtype="act", placement="prefill-in"),
            "rot",
            PrefillMetadata(128, (0,), page_table=_page_table(rows=32), chunk_start=128, prefix_user_id=11),
        )
    assert host_ttnn == []


def test_the_sliced_row_follows_prefix_user_id_when_it_differs_from_the_filled_user():
    """``prefix_user_id`` wins over ``user_ids[0]``; they differ when a request resumes."""

    model = Attention2D.from_config(_config())
    model.bind_kv_cache(_paged_binding(model))

    class _Traced(_Tensor):
        def __init__(self):
            super().__init__("table", shape=(32, 64), dtype="uint32")
            self.slices = []

        def __getitem__(self, item):
            self.slices.append(item)
            return _Tensor("table-row", shape=(1, 64), dtype="uint32")

    table = _Traced()
    metadata = PrefillMetadata(128, (4,), page_table=table, chunk_start=128, prefix_user_id=27)
    model._sdpa_page_table(metadata)
    assert table.slices == [(slice(27, 28), slice(None))]

    table_default = _Traced()
    model._sdpa_page_table(PrefillMetadata(128, (4,), page_table=table_default, chunk_start=128))
    assert table_default.slices == [(slice(4, 5), slice(None))]


def test_an_already_single_row_table_is_passed_straight_through():
    model = Attention2D.from_config(_config())
    model.bind_kv_cache(_paged_binding(model))
    table = _page_table(rows=1)
    metadata = PrefillMetadata(128, (0,), page_table=table, chunk_start=128, prefix_user_id=0)
    assert model._sdpa_page_table(metadata) is table


def test_a_concatenated_prefill_keeps_the_full_table_because_q_carries_every_row():
    model = Attention2D.from_config(_config())
    model.bind_kv_cache(_paged_binding(model))
    table = _page_table(rows=32)
    metadata = PrefillMetadata(
        128, tuple(range(GALAXY_PHYSICAL_BATCH)), page_table=table, chunk_start=128, prefix_user_id=0
    )
    assert model._sdpa_page_table(metadata) is table


def test_a_chunk_page_table_alone_selects_the_prefix_chunked_recipe(host_ttnn):
    """Recorded finding G-C3: the guard against it is unreachable.

    ``_recipe_identity`` treats a non-None ``chunk_page_table`` as one of the
    four signals that select ``PREFIX_CHUNKED``. So by the time
    ``_validate_prefill`` reaches its
    ``"chunk_page_table requires a prefix/chunked recipe"`` branch, the recipe is
    already PREFIX_CHUNKED and the branch can never fire. Passing a chunk table
    with no chunk start silently runs the chunked recipe from token 0 instead of
    being refused.
    """

    model = Attention2D.from_config(_config())
    model.bind_kv_cache(_paged_binding(model))
    metadata = PrefillMetadata(128, (0,), page_table=_page_table(), chunk_page_table=_page_table())

    identity = model._recipe_identity(metadata)
    assert identity.attention_mode is PrefillAttentionMode.PREFIX_CHUNKED

    model.prefill_forward(_Tensor("x", dtype="act", placement="prefill-in"), "rot", metadata)
    assert "sdpa-chunked" in [stage for stage, _ in host_ttnn]
