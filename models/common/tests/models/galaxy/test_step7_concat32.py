# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Milestone B step 7, area 2: concat-32 physical prefill.

The plan's risk for this area is precise: *padding inactive rows must not write
KV or return logits for inactive slots*. Whether that holds is decided by three
host-side artefacts, and this suite inspects all three directly:

1. **the planned tokens** - the flat token stream ``prefill_batched`` builds,
   which row occupies which span of it, and what fills the padding;
2. **the page table** - which table is handed to the fill, and which row of it
   each source row is allowed to touch;
3. **the source rows** - the mapping ``Attention2D`` makes from concatenated
   row *r* to user ``user_ids[r]`` when it slices the KV heads.

Sequence lengths are qualified in ascending order, 128 first, then out to 2048,
because each length is a separate resolved recipe and a separate set of
collective resources.

Not claimed here: that the device produces the same logits through the batched
and the sequential path. That is
``test_..._direct_demo_concat32_prefill_matches_sequential`` in each model's
demo, and it has never run.
"""

from __future__ import annotations

import pytest
import torch

from models.common.models.galaxy.direct_runner import GalaxyDirectRunner
from models.common.models.galaxy.recipes import GalaxyDenseGeometry
from models.common.modules.attention.attention_2d import Attention2D, PrefillMetadata
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
    _paged_binding,
    _Tensor,
    host_ttnn,
)

#: Ascending, as the brief requires. Every entry is a multiple of the 128-token
#: chunk alignment and no larger than the served context.
_BATCHED_LENGTHS = (128, 256, 512, 1024, 2048)


def _open_runner(monkeypatch, *, active_slots=32, batched_lengths=_BATCHED_LENGTHS, max_seq_len=2048):
    recorder = patch_direct_runner(monkeypatch)
    blocks_per_user = max_seq_len // BLOCK_SIZE
    model = RecordingModel(
        max_seq_len=max_seq_len,
        prefill_sequence_lengths=batched_lengths,
        batched_prefill_sequence_lengths=batched_lengths,
        max_num_blocks=blocks_per_user * active_slots + (GALAXY_PHYSICAL_BATCH - active_slots),
    )
    runner = GalaxyDirectRunner(model, active_slots=active_slots)
    patch_compose(monkeypatch, lambda tensor: torch.zeros(1, model.vocab_size))
    runner.open()
    return runner, model, recorder


def _rows(count: int, length: int, *, base: int = 1000) -> list[list[int]]:
    """32 rows whose token values identify their row, so a mix-up is visible."""

    rows = []
    for slot in range(GALAXY_PHYSICAL_BATCH):
        real = length if slot < count else 1
        rows.append([base * (slot + 1) + index for index in range(real)])
    return rows


# ---------------------------------------------------------------------------
# The planned tokens
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("length", _BATCHED_LENGTHS, ids=[f"len{value}" for value in _BATCHED_LENGTHS])
def test_the_flat_token_stream_gives_each_row_its_own_span(monkeypatch, length):
    """Row *r* occupies ``[r * length, (r + 1) * length)`` and nothing else."""

    runner, model, recorder = _open_runner(monkeypatch)
    try:
        token_rows = [[10_000 * (slot + 1) + index for index in range(length)] for slot in range(GALAXY_PHYSICAL_BATCH)]
        runner.prefill_batched(token_rows)

        assert model.prefill_token_rows, "the concatenated prefill staged no token row"
        tokens = model.prefill_token_rows[-1].reshape(-1)
        assert tokens.numel() == GALAXY_PHYSICAL_BATCH * length

        for slot in range(GALAXY_PHYSICAL_BATCH):
            span = tokens[slot * length : (slot + 1) * length].tolist()
            assert span == token_rows[slot], f"row {slot} did not land in its own span"
    finally:
        runner.close()


@pytest.mark.parametrize("active", [16, 31, 32], ids=["active16", "active31", "active32"])
def test_padding_a_row_writes_only_zero_tokens_into_that_rows_span(monkeypatch, active):
    """Inactive rows are padding, and padding is token id 0 - never a neighbour's token."""

    length = 128
    runner, model, recorder = _open_runner(monkeypatch, batched_lengths=(length,))
    try:
        token_rows = _rows(active, length)
        runner.prefill_batched(token_rows)
        flat = model.prefill_token_rows[-1].reshape(-1)

        for slot in range(GALAXY_PHYSICAL_BATCH):
            span = flat[slot * length : (slot + 1) * length].tolist()
            real = len(token_rows[slot])
            assert span[:real] == token_rows[slot]
            assert set(span[real:]) <= {0}, f"row {slot} padding carried non-zero tokens"
    finally:
        runner.close()


@pytest.mark.parametrize("active", [16, 31, 32], ids=["active16", "active31", "active32"])
def test_only_each_rows_real_last_token_is_projected(monkeypatch, active):
    """Logit isolation: a row's logits come from its own final real token.

    ``token_indices`` addresses the last *real* token of each row, which is not
    ``sequence_length - 1`` for any row that was padded. Reading the padded tail
    instead would return a logit computed from a zero token.
    """

    length = 128
    runner, model, recorder = _open_runner(monkeypatch, batched_lengths=(length,))
    try:
        token_rows = _rows(active, length)
        runner.prefill_batched(token_rows)

        assert len(model.projection_calls) == 1
        call = model.projection_calls[0]
        assert call["rows"] == GALAXY_PHYSICAL_BATCH
        assert call["sequence_length"] == length
        assert call["token_indices"] == tuple(len(row) - 1 for row in token_rows)
        assert all(0 <= index < length for index in call["token_indices"])
    finally:
        runner.close()


def test_an_empty_row_is_caught_by_the_generation_entry_point_not_by_the_prefill(monkeypatch):
    """Recorded gap - see REPORT.md area 2, G-C2.

    ``generate`` refuses an empty prompt outright. ``prefill_batched`` called
    directly does not: it plans ``token_indices[r] == -1`` for an empty row and
    leaves the rejection to ``project_prefill_logits``, one call further down.
    The rejection does happen, so no padded logit can be returned - but it
    happens after the whole concatenated prefill graph has run, which on a
    128-row-equivalent stream is the expensive way to find out.
    """

    runner, model, _ = _open_runner(monkeypatch, batched_lengths=(128,))
    try:
        with pytest.raises(ValueError, match="every prompt needs at least one token"):
            runner.generate([[]] * GALAXY_PHYSICAL_BATCH, max_new_tokens=1, batched_prefill=True)

        token_rows = _rows(31, 128)
        token_rows[31] = []
        runner.prefill_batched(token_rows)
        indices = model.projection_calls[-1]["token_indices"]
        assert indices[31] == -1, "an empty row plans an out-of-range token index"
    finally:
        runner.close()


# ---------------------------------------------------------------------------
# The page table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("length", _BATCHED_LENGTHS, ids=[f"len{value}" for value in _BATCHED_LENGTHS])
def test_the_concatenated_prefill_uses_the_replicated_table_and_names_every_user(monkeypatch, length):
    runner, model, recorder = _open_runner(monkeypatch)
    try:
        runner.prefill_batched([[7] * length for _ in range(GALAXY_PHYSICAL_BATCH)])
        call = model.prefill_calls[-1]
        assert call["user_ids"] == tuple(range(GALAXY_PHYSICAL_BATCH))
        assert call["sequence_length"] == length
        # ``prefill_batched`` does not pass these at all; absent is the plan.
        assert call.get("chunk_page_table") is None
        assert call.get("chunk_start") is None
        assert call.get("prefix_user_id") is None
        assert isinstance(call["page_table"].mapper, ReplicateMapper)
        assert call["page_table"] is runner._prefill_page_table
    finally:
        runner.close()


# ---------------------------------------------------------------------------
# The source rows
# ---------------------------------------------------------------------------


class _TracedTable(_Tensor):
    """A page table that records which row each slice asked for."""

    def __init__(self, rows=GALAXY_PHYSICAL_BATCH, columns=64):
        super().__init__("table", shape=(rows, columns), dtype="uint32")
        self.slices = []

    def __getitem__(self, item):
        self.slices.append(item)
        return _Tensor("table-row", shape=(1, self.shape[1]), dtype="uint32")


def test_each_concatenated_source_row_touches_exactly_one_page_table_row(host_ttnn):
    """Source row *r* fills user ``user_ids[r]`` and reads only that user's row.

    ``paged_fill_cache`` is called once per row with ``batch_idx=0`` against a
    one-row table slice, so a row cannot address a user it was not assigned. The
    user order is a deliberate non-identity permutation of the whole batch,
    which makes the row->user pairing observable.
    """

    users = tuple(reversed(range(GALAXY_PHYSICAL_BATCH)))
    model = Attention2D.from_config(_config())
    model.bind_kv_cache(_paged_binding(model))
    table = _TracedTable()
    model.prefill_forward(
        _Tensor("x", dtype="act", placement="prefill-in"),
        "rot",
        PrefillMetadata(128, users, page_table=table),
    )

    fills = [kwargs for stage, kwargs in host_ttnn if stage == "paged_fill_cache"]
    assert len(fills) == 2 * len(users), "one K and one V fill per concatenated row"
    assert all(kwargs["batch_idx"] == 0 for kwargs in fills)
    # One table slice per row, in row order, each naming that row's user.
    assert table.slices == [(slice(user, user + 1), slice(None)) for user in users]


def test_a_row_count_between_one_and_thirty_two_has_no_prefill_recipe(host_ttnn):
    """Recorded limitation - REPORT.md area 2.

    ``_recipe_identity`` resolves SINGLE_ROW or CONCAT_32 and nothing between,
    so "active batch 16" and "active batch 31" cannot be a 16- or 31-row prefill
    at the module level either. They are 32 physical rows with padded members.
    """

    model = Attention2D.from_config(_config())
    model.bind_kv_cache(_paged_binding(model))
    for count in (2, 16, 31):
        with pytest.raises(ValueError, match="exactly one row or concat-32 users"):
            model.prefill_forward(
                _Tensor("x", dtype="act", placement="prefill-in"),
                "rot",
                PrefillMetadata(128, tuple(range(count)), page_table=_Tensor("t", shape=(32, 64), dtype="uint32")),
            )
    assert host_ttnn == []


def test_a_single_row_prefill_still_indexes_the_table_by_user(host_ttnn):
    """The one-row path keeps ``batch_idx=user``; only the concat path slices."""

    model = Attention2D.from_config(_config())
    model.bind_kv_cache(_paged_binding(model))
    model.prefill_forward(
        _Tensor("x", dtype="act", placement="prefill-in"),
        "rot",
        PrefillMetadata(128, (19,), page_table=_Tensor("table", shape=(32, 64), dtype="uint32")),
    )

    fills = [kwargs for stage, kwargs in host_ttnn if stage == "paged_fill_cache"]
    assert len(fills) == 2
    assert all(kwargs["batch_idx"] == 19 for kwargs in fills)


# ---------------------------------------------------------------------------
# What the policy does and does not support
# ---------------------------------------------------------------------------


def test_concatenated_prefill_requires_the_whole_physical_batch(monkeypatch):
    """Recorded limitation - see REPORT.md area 2.

    ``prefill_batched`` refuses any runner whose ``active_slots`` is below 32,
    so "active batch 16" and "active batch 31" cannot be expressed as a smaller
    *paged allocation*. They are expressible only as 32 physical rows of which
    16 or 31 carry real prompts, which is what the tests above measure. The two
    isolation mechanisms - sink blocks for idle slots, and concatenated prefill -
    do not compose.
    """

    runner, _, _ = _open_runner(monkeypatch, active_slots=16, batched_lengths=(128,))
    try:
        with pytest.raises(ValueError, match="exactly 32 active rows"):
            runner.prefill_batched([[1] * 128 for _ in range(GALAXY_PHYSICAL_BATCH)])
    finally:
        runner.close()


def test_a_short_row_count_is_refused(monkeypatch):
    runner, _, _ = _open_runner(monkeypatch, batched_lengths=(128,))
    try:
        with pytest.raises(ValueError, match="exactly 32 active rows"):
            runner.prefill_batched([[1] * 128 for _ in range(31)])
    finally:
        runner.close()


@pytest.mark.parametrize("length", _BATCHED_LENGTHS, ids=[f"len{value}" for value in _BATCHED_LENGTHS])
def test_the_batched_recipe_family_is_selected_in_ascending_order(monkeypatch, length):
    """A prompt picks the smallest batched recipe that covers it, not 2048."""

    runner, _, _ = _open_runner(monkeypatch)
    try:
        assert runner.padded_prefill_length(length, batched=True) == length
        assert runner.padded_prefill_length(length - 1, batched=True) == length
    finally:
        runner.close()


def test_a_prompt_past_the_longest_batched_recipe_fails_closed(monkeypatch):
    runner, _, _ = _open_runner(monkeypatch, batched_lengths=(128, 256))
    try:
        with pytest.raises(ValueError, match="no batched prefill recipe covers 257"):
            runner.padded_prefill_length(257, batched=True)
    finally:
        runner.close()


@pytest.mark.parametrize("length", _BATCHED_LENGTHS, ids=[f"len{value}" for value in _BATCHED_LENGTHS])
def test_each_batched_length_costs_one_collective_recipe_at_32x_its_tokens(length):
    """A concatenated prefill's collectives have single-row 32*L geometry."""

    geometry = GalaxyDenseGeometry(
        dim=8192,
        hidden_dim=28672,
        n_heads=64,
        n_kv_heads=8,
        head_dim=128,
        vocab_size=128256,
        max_seq_len=2048,
        prefill_sequence_lengths=(128,),
        batched_prefill_sequence_lengths=(length,),
    )
    assert geometry.batched_prefill_token_counts == (GALAXY_PHYSICAL_BATCH * length,)
    assert GALAXY_PHYSICAL_BATCH * length in geometry.collective_prefill_token_counts


def test_a_batched_length_beyond_the_served_context_is_refused():
    with pytest.raises(ValueError, match="batched prefill length 4096 exceeds max_seq_len"):
        GalaxyDenseGeometry(
            dim=8192,
            hidden_dim=28672,
            n_heads=64,
            n_kv_heads=8,
            head_dim=128,
            vocab_size=128256,
            max_seq_len=2048,
            prefill_sequence_lengths=(128,),
            batched_prefill_sequence_lengths=(4096,),
        )
