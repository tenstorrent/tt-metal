# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only contracts for the Galaxy direct runner.

Everything here runs without a mesh: the runner resolves its paged block
ownership, its page-table alignment and its recipe selection on the host, before
a single TTNN call. Those are exactly the decisions that decide whether one
slot can read or write another slot's KV cache, so they are worth pinning
independently of hardware.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import ttnn
from models.common.models.galaxy.direct_runner import GalaxyDirectRunner, GalaxySamplingPolicy
from models.common.models.galaxy.kv_contract import GalaxyAttentionKVSpec, GalaxyPagedAttentionConfig

_BLOCK_SIZE = 32
_MAX_SEQ_LEN = 2048
_BLOCKS_PER_USER = _MAX_SEQ_LEN // _BLOCK_SIZE


def _mesh():
    mesh = MagicMock(spec=ttnn.MeshDevice)
    mesh.shape = (8, 4)
    mesh.get_num_devices.return_value = 32
    mesh.arch.return_value = ttnn.device.Arch.WORMHOLE_B0
    return mesh


def _geometry(**overrides):
    values = dict(
        max_batch_size=32,
        users_per_column=8,
        max_seq_len=_MAX_SEQ_LEN,
        vocab_size=128256,
        prefill_sequence_lengths=(128, 512),
        batched_prefill_sequence_lengths=(128,),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _model(*, paged: bool = True, max_num_blocks: int | None = None, n_layers: int = 2, **geometry):
    paged_config = (
        GalaxyPagedAttentionConfig(
            block_size=_BLOCK_SIZE,
            max_num_blocks=_BLOCKS_PER_USER * 32 if max_num_blocks is None else max_num_blocks,
        )
        if paged
        else None
    )
    spec = GalaxyAttentionKVSpec(
        n_local_kv_heads=1, head_dim=128, kv_cache_dtype=ttnn.bfloat8_b, paged_attention_config=paged_config
    )
    return SimpleNamespace(
        geometry=_geometry(**geometry),
        mesh_device=_mesh(),
        kv_specs=(spec,) * n_layers,
        n_layers=n_layers,
        vocab_size=_geometry(**geometry).vocab_size,
    )


# ---------------------------------------------------------------------------
# Paged block ownership
# ---------------------------------------------------------------------------


def test_every_active_slot_owns_a_disjoint_contiguous_block_run():
    runner = GalaxyDirectRunner(_model())
    rows = runner._page_table_rows()

    assert rows.shape == (32, _BLOCKS_PER_USER)
    assert runner.blocks_per_user == _BLOCKS_PER_USER
    assert runner.sink_blocks == 0
    for slot in range(32):
        expected = torch.arange(slot * _BLOCKS_PER_USER, (slot + 1) * _BLOCKS_PER_USER, dtype=torch.int32)
        assert torch.equal(rows[slot], expected)
    assert len(set(rows.reshape(-1).tolist())) == rows.numel(), "two slots share a block"


def test_idle_slots_get_their_own_sink_block():
    """An idle slot still decodes, so its writes must land somewhere private."""

    active = 4
    runner = GalaxyDirectRunner(_model(max_num_blocks=_BLOCKS_PER_USER * active + (32 - active)), active_slots=active)
    rows = runner._page_table_rows()

    assert runner.blocks_per_user == _BLOCKS_PER_USER
    assert runner.sink_blocks == 32 - active
    active_blocks = set(rows[:active].reshape(-1).tolist())
    assert len(active_blocks) == active * _BLOCKS_PER_USER
    sinks = [set(rows[slot].tolist()) for slot in range(active, 32)]
    assert all(len(sink) == 1 for sink in sinks), "an idle slot spans more than one block"
    assert len(set().union(*sinks)) == 32 - active, "two idle slots share a sink block"
    assert not active_blocks & set().union(*sinks), "an idle slot writes into an active slot's pages"


def test_a_pool_too_small_for_the_served_context_fails_closed():
    with pytest.raises(ValueError, match="cannot hold max_seq_len"):
        GalaxyDirectRunner(_model(max_num_blocks=_BLOCKS_PER_USER * 32 - 1))


def test_active_slots_apply_to_paged_pools_only():
    with pytest.raises(ValueError, match="contiguous KV cache serves every slot"):
        GalaxyDirectRunner(_model(paged=False), active_slots=1)


# ---------------------------------------------------------------------------
# Page table alignment
# ---------------------------------------------------------------------------


def test_page_table_columns_are_padded_to_the_chunked_sdpa_stick():
    """Chunked SDPA reads 32-byte sticks, i.e. eight int32 entries."""

    runner = GalaxyDirectRunner(_model())
    padded = runner._pad_columns(torch.arange(32 * 4, dtype=torch.int32).reshape(32, 4))

    assert padded.shape == (32, 8)
    assert torch.equal(padded[:, :4], torch.arange(32 * 4, dtype=torch.int32).reshape(32, 4))
    assert torch.all(padded[:, 4:] == 0), "padding must be a read-safe block id"


def test_an_already_aligned_table_is_returned_unchanged():
    runner = GalaxyDirectRunner(_model())
    rows = runner._page_table_rows()

    assert runner._pad_columns(rows) is rows


def test_only_the_prefill_table_is_stick_aligned():
    """The decode SDPA reads its KV length from the table's row width.

    Padding the decode table would claim more cached context than each slot
    owns; padding the prefill table is what chunked SDPA's 32-byte sticks need.
    """

    context = 128
    blocks = context // _BLOCK_SIZE
    runner = GalaxyDirectRunner(_model(max_seq_len=context, max_num_blocks=blocks * 32))

    assert runner.blocks_per_user == blocks == 4
    assert runner.decode_page_table_rows().shape == (32, 4)
    assert runner.prefill_page_table_rows().shape == (32, 8)


# ---------------------------------------------------------------------------
# Recipe selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("tokens", "expected"), [(1, 128), (128, 128), (129, 512), (512, 512)])
def test_prefill_padding_picks_the_smallest_covering_recipe(tokens, expected):
    assert GalaxyDirectRunner(_model()).padded_prefill_length(tokens) == expected


def test_a_prompt_beyond_every_recipe_fails_closed():
    runner = GalaxyDirectRunner(_model())
    with pytest.raises(ValueError, match="no single-row prefill recipe covers 513 tokens"):
        runner.padded_prefill_length(513)


def test_batched_padding_uses_the_batched_recipe_family():
    runner = GalaxyDirectRunner(_model())

    assert runner.padded_prefill_length(128, batched=True) == 128
    with pytest.raises(ValueError, match="no batched prefill recipe covers 129 tokens"):
        runner.padded_prefill_length(129, batched=True)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_greedy_is_temperature_zero_or_top_one():
    assert GalaxySamplingPolicy(top_k=32, temperature=0.0).greedy
    assert GalaxySamplingPolicy(top_k=1, temperature=1.0).greedy
    assert not GalaxySamplingPolicy(top_k=32, temperature=0.8).greedy


def test_greedy_host_sampling_is_argmax():
    runner = GalaxyDirectRunner(_model())
    logits = torch.zeros((3, 16))
    logits[torch.arange(3), torch.tensor([2, 5, 11])] = 1.0

    assert runner.sample_host(logits, GalaxySamplingPolicy()) == [2, 5, 11]


def test_seeded_stochastic_host_sampling_repeats():
    runner = GalaxyDirectRunner(_model())
    torch.manual_seed(0)
    logits = torch.randn((4, 64))
    policy = GalaxySamplingPolicy(top_k=8, top_p=0.9, temperature=0.7, seed=1234)

    assert runner.sample_host(logits, policy) == runner.sample_host(logits, policy)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_graph_calls_before_open_fail_closed():
    runner = GalaxyDirectRunner(_model())
    with pytest.raises(RuntimeError, match="not open"):
        runner.prefill_row([1, 2, 3], slot=0)
    with pytest.raises(RuntimeError, match="not open"):
        runner.decode_logits([0] * 32, [0] * 32)


def test_close_is_idempotent_before_open():
    runner = GalaxyDirectRunner(_model())
    runner.close()
    runner.close()

    assert runner._kv_cache == []
