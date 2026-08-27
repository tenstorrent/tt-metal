# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Milestone B step 7: repeated requests, repeated construction, and teardown.

Two claims sit under "repeat and cleanup", and they have very different amounts
of host-visible surface.

**Repeated requests against one live model.** Determinism is a property of the
plan as much as of the arithmetic: if two identical requests stage different
tokens, different positions or a different page table, the tokens cannot match
whatever the mesh does. That half is checked here.

**Repeated construction and teardown in one process.** This is where Milestone A
limitation **L1** bites: ``Prefetcher2D.cleanup()`` cannot free its global
circular buffer, so a second owner's ``seal()`` fails with an L1 OOM unless the
consumers are torn down before or with the owner. The *OOM* needs L1 and cannot
be reproduced here. What can be reproduced, and is, is the ownership statement
that makes it invisible: after ``cleanup()`` the owner truthfully reports
``owned_resources == ()`` while the global CB it created was never handed to
``deallocate``. That gap is the whole of L1, and it is a host fact.

The prefetcher fakes come from the module suite that owns them.
"""

from __future__ import annotations

import pytest
import torch

from models.common.models.galaxy.direct_runner import GalaxyDirectRunner, GalaxySamplingPolicy
from models.common.tests.models.galaxy.step7_harness import (
    BLOCK_SIZE,
    GALAXY_PHYSICAL_BATCH,
    RecordingModel,
    patch_compose,
    patch_direct_runner,
)
from models.common.tests.modules.prefetcher.test_prefetcher_2d import ResourceHarness, initialized_owner, seal_one


def _open_runner(monkeypatch, *, active_slots=32, vocab_rows=None):
    recorder = patch_direct_runner(monkeypatch)
    blocks_per_user = 2048 // BLOCK_SIZE
    model = RecordingModel(
        max_num_blocks=blocks_per_user * active_slots + (GALAXY_PHYSICAL_BATCH - active_slots),
    )
    runner = GalaxyDirectRunner(model, active_slots=active_slots)
    rows = vocab_rows if vocab_rows is not None else torch.zeros(GALAXY_PHYSICAL_BATCH, model.vocab_size)
    patch_compose(monkeypatch, lambda tensor: rows)
    runner.open()
    return runner, model, recorder


# ---------------------------------------------------------------------------
# Repeated requests against one live model
# ---------------------------------------------------------------------------


def test_two_identical_requests_stage_identical_tokens_positions_and_tables(monkeypatch):
    logits = torch.zeros(GALAXY_PHYSICAL_BATCH, 128256)
    logits[:, 7] = 1.0
    runner, model, recorder = _open_runner(monkeypatch, vocab_rows=logits)
    try:
        prompt = list(range(1, 65))
        policy = GalaxySamplingPolicy(top_k=1, temperature=0.0)

        first = runner.generate([prompt], max_new_tokens=4, policy=policy)
        boundary = (len(model.prefill_calls), len(model.decode_calls), len(model.prefill_token_rows))
        second = runner.generate([prompt], max_new_tokens=4, policy=policy)

        assert first[0].generated_tokens == second[0].generated_tokens
        assert first[0].generated_tokens == [7, 7, 7, 7]

        # The plan repeats exactly: same call counts and same staged values.
        assert len(model.prefill_calls) == 2 * boundary[0]
        assert len(model.decode_calls) == 2 * boundary[1]
        assert torch.equal(model.prefill_token_rows[0], model.prefill_token_rows[boundary[2]])
        half = len(model.decode_token_rows) // 2
        for index in range(half):
            assert torch.equal(model.decode_token_rows[index], model.decode_token_rows[half + index])
    finally:
        runner.close()


def test_a_repeated_request_reuses_the_bound_cache_and_tables_rather_than_restaging(monkeypatch):
    """Nothing is reallocated per request; the KV binding happens once."""

    runner, model, recorder = _open_runner(monkeypatch)
    try:
        staged_after_open = len(recorder.staged)
        binds_after_open = len(model.bind_calls)
        prompt = list(range(1, 33))
        for _ in range(3):
            runner.prefill_row(prompt, slot=0)
        assert len(model.bind_calls) == binds_after_open, "a request rebound the KV cache"
        assert len(recorder.staged) == staged_after_open, "a request restaged a device tensor"
    finally:
        runner.close()


def test_the_same_slot_addresses_the_same_blocks_on_every_repeat(monkeypatch):
    runner, _, _ = _open_runner(monkeypatch)
    try:
        rows = [runner._page_table_rows() for _ in range(3)]
        assert torch.equal(rows[0], rows[1])
        assert torch.equal(rows[1], rows[2])
    finally:
        runner.close()


# ---------------------------------------------------------------------------
# Teardown of the runner
# ---------------------------------------------------------------------------


def test_close_unbinds_the_cache_releases_the_tables_and_is_idempotent(monkeypatch):
    runner, model, recorder = _open_runner(monkeypatch)
    tables = [runner._prefill_page_table, runner._decode_page_table]
    caches = [tensor for pair in runner._kv_cache for tensor in pair]

    runner.close()
    assert model.bind_calls[-1] is None, "close did not unbind the KV cache"
    assert all(table.is_allocated() is False for table in tables)
    assert all(tensor.is_allocated() is False for tensor in caches)

    before = len(model.bind_calls)
    runner.close()
    runner.close()
    assert len(model.bind_calls) == before, "a repeated close unbound again"


def test_a_closed_runner_refuses_further_graph_calls(monkeypatch):
    runner, _, _ = _open_runner(monkeypatch)
    runner.close()
    with pytest.raises(RuntimeError, match="not open"):
        runner.prefill_row([1, 2, 3], slot=0)
    with pytest.raises(RuntimeError, match="not open"):
        runner.decode_logits([1] * GALAXY_PHYSICAL_BATCH, [0] * GALAXY_PHYSICAL_BATCH)


def test_a_failed_open_leaves_nothing_bound(monkeypatch):
    """``open`` records the binding before staging so a staging failure unbinds."""

    recorder = patch_direct_runner(monkeypatch)
    model = RecordingModel()
    runner = GalaxyDirectRunner(model)

    real_stage = runner._stage_page_table
    calls = {"count": 0}

    def failing_stage(rows, *, sharded):
        calls["count"] += 1
        if calls["count"] == 2:  # fail while staging the decode table
            raise RuntimeError("injected staging failure")
        return real_stage(rows, sharded=sharded)

    monkeypatch.setattr(runner, "_stage_page_table", failing_stage)

    with pytest.raises(RuntimeError, match="injected staging failure"):
        runner.open()

    assert model.bind_calls[-1] is None, "a failed open left the KV cache bound"
    assert runner._kv_cache == []
    assert runner._prefill_page_table is None and runner._decode_page_table is None
    assert recorder.staged, "the failure happened after the cache was allocated, as intended"


def test_reopening_after_a_close_allocates_a_fresh_cache(monkeypatch):
    recorder = patch_direct_runner(monkeypatch)
    model = RecordingModel()
    runner = GalaxyDirectRunner(model)
    runner.open()
    first = [tensor for pair in runner._kv_cache for tensor in pair]
    runner.close()
    runner.open()
    try:
        second = [tensor for pair in runner._kv_cache for tensor in pair]
        assert all(tensor.is_allocated() is False for tensor in first)
        assert all(tensor.is_allocated() for tensor in second)
        assert all(a is not b for a, b in zip(first, second))
    finally:
        runner.close()


def test_open_is_idempotent_while_already_open(monkeypatch):
    recorder = patch_direct_runner(monkeypatch)
    model = RecordingModel()
    runner = GalaxyDirectRunner(model)
    runner.open()
    try:
        staged = len(recorder.staged)
        binds = len(model.bind_calls)
        runner.open()
        assert len(recorder.staged) == staged
        assert len(model.bind_calls) == binds
    finally:
        runner.close()


# ---------------------------------------------------------------------------
# L1: repeated construction and the global circular buffer
# ---------------------------------------------------------------------------


def test_cleanup_reports_no_owned_resources_while_never_freeing_the_global_cb():
    """Milestone A limitation L1, pinned on the host.

    ``cleanup()`` clears ``self._global_cb`` without adding it to the resources
    it deallocates, because ttnn exposes no free for a global circular buffer.
    The owner then reports ``owned_resources == ()`` - truthfully, by its own
    definition - while roughly 55 MB of L1 is still resident on the mesh. That
    is why the *next* owner's ``seal()`` OOMs, and why the honest reading of a
    clean ``cleanup()`` is "nothing this object still owns", not "nothing is
    left on the device".

    The OOM itself needs L1 and is not reproduced here.
    """

    resources = ResourceHarness()
    owner = initialized_owner(resources, expected_weight_count=1)
    seal_one(owner)

    created = list(resources.created_cbs)
    assert len(created) == 1
    assert owner.owned_resources, "a sealed owner owns the CB and the address metadata"

    owner.cleanup()

    assert owner.owned_resources == ()
    assert created[0] not in resources.deallocated, (
        "the global circular buffer was deallocated; if ttnn gained a free for it, "
        "L1 is closed and this test should become the assertion that it is"
    )


def test_a_second_owner_on_the_same_mesh_reallocates_a_second_global_cb():
    """Two constructions in one process allocate two CBs; only one can be freed.

    On the host both seals succeed, because nothing tracks L1. On a mesh the
    second ``seal()`` is the one that fails. Recording the host behaviour makes
    the difference between the two explicit rather than a surprise.
    """

    resources = ResourceHarness()
    first = initialized_owner(resources, expected_weight_count=1)
    seal_one(first)
    first.cleanup()

    second = initialized_owner(resources, expected_weight_count=1)
    seal_one(second)
    try:
        assert len(resources.created_cbs) == 2
        assert resources.created_cbs[0] != resources.created_cbs[1]
        assert not any(cb in resources.deallocated for cb in resources.created_cbs)
    finally:
        second.cleanup()


def test_cleanup_is_terminal_and_a_sealed_owner_cannot_be_resealed():
    resources = ResourceHarness()
    owner = initialized_owner(resources, expected_weight_count=1)
    seal_one(owner)
    owner.cleanup()
    owner.cleanup()  # idempotent

    assert owner.owned_resources == ()
    with pytest.raises(RuntimeError):
        owner.seal()


def test_the_galaxy_global_cb_size_is_a_topology_constant_not_a_model_choice():
    from models.common.models.galaxy.prefetch import GALAXY_GLOBAL_CB_SIZE

    # 728 pages of 1088 bytes: the bfloat8_b tile size times the Galaxy decode
    # receiver depth. Roughly 0.76 MiB per receiver core.
    assert GALAXY_GLOBAL_CB_SIZE == 728 * 1088
    assert GALAXY_GLOBAL_CB_SIZE % 1088 == 0
