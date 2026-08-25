# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The async boundary the compaction un-permutation has to survive.

`Qwen3CoderForCausalLM` may decode a compacted batch in a graph narrower than
`max_num_seqs`, which means the sampled tokens come back in **graph-row** order
and have to be scattered back to vLLM's slots. The mapping that does that is
chosen when a forward is *issued*; the scatter happens when the output is
*read*. Under `--async-scheduling` those are different steps.

So the hazard is not the permutation arithmetic -- that is covered by
`doc/batch_scaling/probes/compaction_identity.py` on real weights -- it is the
**pairing**: if a later step installs a new mapping before an earlier step's
tokens are read, the earlier tokens get scattered with the wrong permutation and
every one of them lands on the wrong request. Silently, and as a correctness
bug rather than a slowdown.

The vLLM plugin happens to order this safely today (it drains pending async
decodes on a layout change, and only a layout change can move the mapping), but
that invariant lives in a repository this one must not modify and cannot pin.
`_pending_orders` removes the dependency by pairing each output with the mapping
its own forward used, and these tests are what hold that property in place.

Deliberately **device-free**: the whole hazard is adapter bookkeeping, so a fake
generator exercises it exactly and the tests run in milliseconds. Three earlier
pieces of evidence -- a real-weights identity probe, a 158-test suite and a
serving A/B -- all passed while this bug was present, because every one of them
either used the synchronous read path or never moved the mapping. That is the
gap these tests close.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from models.demos.blackhole.qwen3_coder_30b_a3b.tt import generator_vllm as gv
from models.demos.blackhole.qwen3_coder_30b_a3b.tt.generator import Qwen3CoderGenerator
from models.demos.blackhole.qwen3_coder_30b_a3b.tt.generator_vllm import Qwen3CoderForCausalLM

SLOTS = 32


class _Handle:
    """Stands in for the device tensor a decode forward returns."""

    def __init__(self, tag: int):
        self.tag = tag


class _FakeGenerator:
    """Only the surface `decode_forward` / `process_decode_output_host` touch.

    `read_sampled_tokens` returns a vector that encodes **which forward** it came
    from and **which graph row** each entry is, so a mis-paired scatter is
    visible in the values rather than having to be inferred.
    """

    def __init__(self):
        self.model = object()
        self.mesh_device = object()
        self.pages_per_user = 8
        self.page_block_size = 32
        self.num_blocks = SLOTS * 8
        self.issued: list[_Handle] = []
        self.trace_stats: dict = {}

    # -- the calls the adapter makes on the decode path --------------------
    def set_sampling_params(self, **kwargs):
        return None

    def set_penalty_params(self, **kwargs):
        return False, False

    def decode_device_state(self):
        return None  # first install every time: the adapter takes the host view

    def prefill_forward(self, tokens, **kwargs):
        # Host-sampled shape: the adapter reshapes this to [active, 1, vocab].
        return torch.zeros((int(tokens.shape[0]), 8))

    def decode_forward(self, *args, **kwargs):
        if kwargs.get("sampling_mode") == "host":
            # Host-sampled decode returns logits, not a device handle; the
            # adapter reshapes them to [rows, 1, vocab].
            return torch.zeros((SLOTS, 8))
        handle = _Handle(len(self.issued))
        self.issued.append(handle)
        return handle

    def read_sampled_tokens(self, tt_out, count):
        # row r of forward `tag` -> 1000 * (tag + 1) + r
        return torch.tensor([1000 * (tt_out.tag + 1) + r for r in range(count)], dtype=torch.long)


class _Sampling:
    def __init__(self, rows: int):
        self.temperature = [0.0] * rows
        self.top_k = [1] * rows
        self.top_p = [1.0] * rows
        self.seed = [None] * rows
        self.repetition_penalty = [1.0] * rows
        self.presence_penalty = [0.0] * rows
        self.frequency_penalty = [0.0] * rows


def _adapter(widths: str = "1,2,4,8,16,32") -> Qwen3CoderForCausalLM:
    with patch.dict("os.environ", {"QWEN3_DECODE_WIDTHS": widths}):
        adapter = Qwen3CoderForCausalLM(_FakeGenerator(), max_model_len=4096, max_num_seqs=SLOTS)
    adapter.kv_cache = ["fake-cache"]
    return adapter


def _batch(live_rows):
    """vLLM's padded decode batch: position -1 on every unoccupied slot."""
    positions = torch.full((SLOTS,), -1, dtype=torch.int64)
    tokens = torch.zeros((SLOTS, 1), dtype=torch.int64)
    for row in live_rows:
        positions[row] = 128
        tokens[row, 0] = 5000 + row
    return tokens, positions


def _issue(adapter, live_rows, *, reset=True):
    """One decode forward, output deliberately NOT read (the async path)."""
    tokens, positions = _batch(live_rows)
    return adapter.decode_forward(
        tokens=tokens,
        page_table=torch.zeros((SLOTS, 8), dtype=torch.int32),
        kv_cache=adapter.kv_cache,
        start_pos=positions,
        sampling_params=_Sampling(SLOTS),
        reset_batch=reset,
        read_from_device=False,
    )


def _read(adapter, handle):
    with patch.object(gv.ttnn, "is_tensor_storage_on_device", lambda _t: False):
        return adapter.process_decode_output_host(handle, is_tokens=True).reshape(-1)


def test_output_uses_the_mapping_its_own_forward_was_issued_with():
    """Issue A, then issue B with a different mapping, then read A.

    This is the exact interleaving `--async-scheduling` produces and the one
    nothing else covers. With the mapping stored on the adapter rather than
    queued, reading A after B scatters A's tokens through **B's** permutation.
    """
    adapter = _adapter()

    handle_a = _issue(adapter, [3, 17, 29])  # width 4, order starts 3,17,29
    order_a = adapter._compaction.clone()
    handle_b = _issue(adapter, [0, 1, 2, 3, 4, 5])  # width 8, a different order
    order_b = adapter._compaction.clone()

    # The premise of the test: the adapter's live mapping is no longer A's.
    assert not torch.equal(
        order_a[: min(len(order_a), len(order_b))], order_b[: min(len(order_a), len(order_b))]
    ), "the two steps must have different mappings or this test proves nothing"

    tokens_a = _read(adapter, handle_a)
    # Forward 0, graph rows 0,1,2 -> vLLM slots 3,17,29.
    assert tokens_a[3] == 1000, tokens_a[[3, 17, 29]]
    assert tokens_a[17] == 1001, tokens_a[[3, 17, 29]]
    assert tokens_a[29] == 1002, tokens_a[[3, 17, 29]]

    tokens_b = _read(adapter, handle_b)
    # Forward 1, contiguous live rows -> identity over the first six slots.
    for row in range(6):
        assert tokens_b[row] == 2000 + row, tokens_b[:6]

    assert adapter._audit["compaction_fifo_underflows"] == 0
    assert len(adapter._pending_orders) == 0


def test_three_forwards_in_flight_are_read_in_issue_order():
    """FIFO depth > 2, so the pairing cannot be a lucky one-slot swap."""
    adapter = _adapter()
    handles = [_issue(adapter, rows) for rows in ([3, 17, 29], [0, 1], [7])]
    assert adapter._audit["compaction_fifo_max_depth"] == 3

    expected = ({3: 1000, 17: 1001, 29: 1002}, {0: 2000, 1: 2001}, {7: 3000})
    for handle, wanted in zip(handles, expected):
        got = _read(adapter, handle)
        for slot, value in wanted.items():
            assert got[slot] == value, (slot, value, got[slot])
    assert adapter._audit["compaction_fifo_underflows"] == 0


def test_full_width_steps_queue_a_null_mapping():
    """At full occupancy there is no permutation, and that must still be paired.

    A `None` entry is meaningful: it says "this step needs no un-permutation".
    Skipping the push for full-width steps would misalign the queue for every
    narrow step behind them.
    """
    adapter = _adapter()
    handle_full = _issue(adapter, list(range(SLOTS)))
    assert adapter._compaction is None
    handle_narrow = _issue(adapter, [9])
    assert adapter._compaction is not None

    tokens_full = _read(adapter, handle_full)
    for row in (0, 5, 31):
        assert tokens_full[row] == 1000 + row, tokens_full[:3]

    tokens_narrow = _read(adapter, handle_narrow)
    assert tokens_narrow[9] == 2000, tokens_narrow[:3]
    assert adapter._audit["compaction_fifo_underflows"] == 0


def _unset_widths_adapter() -> Qwen3CoderForCausalLM:
    """An adapter built with `QWEN3_DECODE_WIDTHS` genuinely absent."""
    import os

    with patch.dict("os.environ", {}, clear=False):
        os.environ.pop("QWEN3_DECODE_WIDTHS", None)
        adapter = Qwen3CoderForCausalLM(_FakeGenerator(), max_model_len=4096, max_num_seqs=SLOTS)
    adapter.kv_cache = ["fake-cache"]
    return adapter


def test_the_ladder_is_on_when_the_variable_is_unset():
    """Unset must mean the ladder, not the fixed-width graph.

    The previous default was off, and it was *known wrong*: a `max_num_seqs=32`
    server that simply does not set an environment variable decodes one user at
    4.3464 t/s/u instead of 49.3636. This test is what stops that default coming
    back by accident.
    """
    adapter = _unset_widths_adapter()

    assert adapter._decode_widths == [1, 2, 4, 8, 16, SLOTS]
    assert adapter._compaction_enabled is True
    handle = _issue(adapter, [3, 17, 29])
    assert adapter._compaction is not None, "unset must still compact"
    assert len(adapter._pending_orders) == 1
    tokens = _read(adapter, handle)
    # Un-permuted back to vLLM slots, exactly as with the ladder set explicitly.
    for row in (3, 17, 29):
        assert tokens[row] == 1000 + [3, 17, 29].index(row), tokens[[3, 17, 29]]


def test_a_single_width_restores_the_fixed_width_path_exactly():
    """`QWEN3_DECODE_WIDTHS=32` is the escape hatch back to the old behaviour.

    With one width there is nothing to compact, so the path takes none of the
    bookkeeping and cannot raise any of the pairing errors below -- which is the
    property the previous default provided and which must remain reachable.
    """
    adapter = _adapter(widths=str(SLOTS))

    assert adapter._decode_widths == [SLOTS]
    assert adapter._compaction_enabled is False
    handle = _issue(adapter, [3, 17, 29])
    assert adapter._compaction is None, "no compaction may happen with a single width"
    assert len(adapter._pending_orders) == 0, "the single-width path must not touch the queue"
    tokens = _read(adapter, handle)
    # Straight through: graph row r is slot r.
    for row in (0, 3, 17, 29):
        assert tokens[row] == 1000 + row, tokens[[0, 3, 17, 29]]


def test_widths_above_max_num_seqs_are_dropped_from_the_default():
    """A smaller server must not try to capture a graph wider than its slots."""
    import os

    with patch.dict("os.environ", {}, clear=False):
        os.environ.pop("QWEN3_DECODE_WIDTHS", None)
        adapter = Qwen3CoderForCausalLM(_FakeGenerator(), max_model_len=4096, max_num_seqs=8)

    assert adapter._decode_widths == [1, 2, 4, 8]


@pytest.mark.parametrize("live_rows", [[0], [3, 17, 29], list(range(16)), list(range(SLOTS))])
def test_queue_drains_exactly_once_per_forward(live_rows):
    adapter = _adapter()
    handle = _issue(adapter, live_rows)
    assert len(adapter._pending_orders) == 1
    _read(adapter, handle)
    assert len(adapter._pending_orders) == 0
    assert adapter._audit["compaction_fifo_underflows"] == 0


# -- the guards -------------------------------------------------------------
#
# The pairing rests on decode forwards being finalized exactly once and in issue
# order. Those are invariants of `vllm-tt-plugin`, which this repository must not
# modify, so they are *checked* here rather than trusted. The route that makes
# this concrete: `async_decode.py::ensure_finalized` sets `_finalized = True`
# only after `_get_output_impl()` returns, and the pop happens inside that call
# -- so a raise anywhere after the pop leaves the step un-finalized and a later
# `wait_for_all_pending_async_steps` finalizes it a second time.


def test_underflow_raises_instead_of_guessing(expect_error):
    """No queued mapping means the pairing is broken; a wrong guess is worse.

    Falling back to the adapter's current mapping here would apply exactly the
    permutation the queue exists to prevent, in the one state where it is known
    not to belong to these tokens -- every token to the wrong request, silently.
    """
    adapter = _adapter()
    handle = _issue(adapter, [3, 17, 29])
    _read(adapter, handle)
    with expect_error(RuntimeError, "no queued row mapping"):
        _read(adapter, handle)  # the second finalize of the same forward
    assert adapter._audit["compaction_fifo_underflows"] == 1


def test_double_finalize_is_caught_by_the_tag_before_it_can_mis_scatter(expect_error):
    """A re-finalize consumes the *next* step's mapping; the tag says so.

    This is the overflow-direction desync: the queue drains faster than it
    fills. Depth alone cannot see it -- the tags can.
    """
    adapter = _adapter()
    a = _issue(adapter, [3, 17, 29])
    _issue(adapter, [0, 1])
    _read(adapter, a)  # legitimate: pops tag 0
    with expect_error(RuntimeError, "out of step"):
        # A second finalize of forward A pops tag 1, which belongs to forward B.
        # Without the tag it would silently scatter A's tokens through B's map.
        adapter._pending_orders.appendleft((99, adapter._compaction))
        _read(adapter, a)


def test_queue_cap_raises_rather_than_growing_without_bound(expect_error):
    """Outputs never read is a leak; fail at the cap instead of mis-pairing later."""
    adapter = _adapter()
    adapter._pending_orders_cap = 4
    with expect_error(RuntimeError, "row-mapping queue reached"):
        for _ in range(adapter._pending_orders_cap + 2):
            _issue(adapter, [3, 17, 29])


def test_reset_realigns_tags_so_later_pops_still_pair():
    """A released trace makes queued outputs unreadable; the reset must not desync.

    Clearing without realigning the tags would make the next legitimate pop look
    like a skipped step and raise on a perfectly healthy server.
    """
    adapter = _adapter()
    _issue(adapter, [3, 17, 29])
    _issue(adapter, [0, 1])
    assert len(adapter._pending_orders) == 2
    adapter._reset_pending_orders()
    assert len(adapter._pending_orders) == 0

    handle = _issue(adapter, [7])
    got = _read(adapter, handle)
    assert got[7] == 3000, got[:8]  # third forward issued -> tag 1000*(2+1)
    assert adapter._audit["compaction_fifo_underflows"] == 0


def test_prefill_resets_the_queue():
    """Prefill may release the decode traces, so queued outputs die with them."""
    adapter = _adapter()
    _issue(adapter, [3, 17, 29])
    assert len(adapter._pending_orders) == 1
    adapter.prefill_forward(
        tokens=torch.zeros((1, 8), dtype=torch.int64),
        page_table=torch.zeros((SLOTS, 8), dtype=torch.int32),
        kv_cache=adapter.kv_cache,
        prompt_lens=[8],
        sampling_params=None,
    )
    assert len(adapter._pending_orders) == 0
    handle = _issue(adapter, [5])
    _read(adapter, handle)
    assert adapter._audit["compaction_fifo_underflows"] == 0


# ---------------------------------------------------------------------------
# The penalty path under the ladder.
#
# `_apply_penalties` reorders the token history into graph-row order alongside
# the per-row penalty scalars. The scalars are genuine python lists, so a list
# comprehension is right for them; the histories are vLLM `[rows, L]` **torch
# tensors**, and rebuilding one as a list of 1-D tensors makes
# `Qwen3CoderGenerator._row_token_ids` raise `TypeError: only integer tensors of
# a single element can be converted to an index` on its `torch.as_tensor` call.
#
# That crashed a real penalised request in production. Nothing caught it: the
# ladder must be ON (with it off the history is passed through untouched) *and*
# the request must carry a non-neutral penalty, and `_FakeGenerator` above
# accepts `set_penalty_params(**kwargs)` without ever looking at the history.
# These tests run the **real** `_row_token_ids` over whatever the adapter
# actually passed, so a type that the generator cannot consume fails here.
# ---------------------------------------------------------------------------


class _PenaltyRecordingGenerator(_FakeGenerator):
    """Captures the kwargs `_apply_penalties` hands to the generator."""

    def __init__(self):
        super().__init__()
        self.penalty_calls: list[dict] = []

    def set_penalty_params(self, **kwargs):
        self.penalty_calls.append(kwargs)
        return False, False


def _penalty_adapter(widths: str = "1,2,4,8,16,32") -> Qwen3CoderForCausalLM:
    with patch.dict("os.environ", {"QWEN3_DECODE_WIDTHS": widths}):
        adapter = Qwen3CoderForCausalLM(_PenaltyRecordingGenerator(), max_model_len=4096, max_num_seqs=SLOTS)
    adapter.kv_cache = ["fake-cache"]
    return adapter


def _history(rows: int, width: int = 6) -> torch.Tensor:
    """A vLLM `[rows, L]` history: slot r holds tokens 100*r+1.., -1 padded.

    The -1 padding and the batch padded to `max_num_reqs` are what
    `_row_token_ids` documents, so this mirrors the real contract.
    """
    hist = torch.full((rows, width), -1, dtype=torch.int64)
    for row in range(rows):
        hist[row, :3] = torch.tensor([100 * row + 1, 100 * row + 2, 100 * row + 3])
    return hist


def _penalised(rows: int) -> _Sampling:
    sampling = _Sampling(rows)
    sampling.repetition_penalty = [1.2] * rows  # non-neutral: takes the staged path
    return sampling


def test_penalised_decode_under_the_ladder_does_not_crash_on_a_tensor_history():
    """The production crash, reduced.

    A `[rows, L]` tensor history reordered into a python list of 1-D tensors
    reaches `_row_token_ids` as something `torch.as_tensor` cannot index.
    """
    adapter = _penalty_adapter()
    live = [3, 17, 29]
    order = adapter._compaction_order(_batch(live)[1], 4, SLOTS)

    adapter._apply_penalties(_penalised(SLOTS), SLOTS, _history(SLOTS), _history(SLOTS), order=order, graph_rows=4)

    call = adapter.generator.penalty_calls[-1]
    for name in ("prompt_tokens", "output_tokens"):
        # The real consumer, on the real object the adapter passed.
        ids = Qwen3CoderGenerator._row_token_ids(call[name], 0)
        assert ids.numel() == 3, f"{name} row 0 unreadable by the generator"


def test_penalty_history_follows_its_own_slot_through_the_compaction():
    """Graph row g must carry the history of the slot the mapping sent there.

    A reorder that is merely type-correct but inverted would apply one user's
    repetition penalty to another user's tokens -- wrong output, no crash.
    """
    adapter = _penalty_adapter()
    live = [3, 17, 29]
    order = adapter._compaction_order(_batch(live)[1], 4, SLOTS)

    adapter._apply_penalties(_penalised(SLOTS), SLOTS, _history(SLOTS), _history(SLOTS), order=order, graph_rows=4)

    call = adapter.generator.penalty_calls[-1]
    for graph_row, slot in enumerate(int(v) for v in order.tolist()):
        ids = Qwen3CoderGenerator._row_token_ids(call["prompt_tokens"], graph_row)
        expected = torch.tensor([100 * slot + 1, 100 * slot + 2, 100 * slot + 3])
        assert torch.equal(ids, expected), f"graph row {graph_row} carries slot {slot}'s history"


def test_penalty_scalars_and_history_are_reordered_the_same_way():
    """The scalar and the history for one slot must not come apart.

    They are reordered by separate statements; if only one of them tracked the
    mapping, a user would get another user's penalty strength.
    """
    adapter = _penalty_adapter()
    live = [3, 17, 29]
    order = adapter._compaction_order(_batch(live)[1], 4, SLOTS)

    sampling = _Sampling(SLOTS)
    # A distinct penalty per slot, so a mis-pairing is visible in the value.
    sampling.repetition_penalty = [1.0 + 0.01 * r for r in range(SLOTS)]

    adapter._apply_penalties(sampling, SLOTS, _history(SLOTS), None, order=order, graph_rows=4)

    call = adapter.generator.penalty_calls[-1]
    for graph_row, slot in enumerate(int(v) for v in order.tolist()):
        assert call["repetition"][graph_row] == pytest.approx(1.0 + 0.01 * slot)
        ids = Qwen3CoderGenerator._row_token_ids(call["prompt_tokens"], graph_row)
        assert int(ids[0]) == 100 * slot + 1, "history and scalar disagree about the slot"


@pytest.mark.parametrize("as_list", [False, True])
def test_history_reorder_preserves_the_type_it_was_given(as_list):
    """Tensors stay tensors; genuine python sequences keep working.

    `_row_token_ids` accepts both, but only via `torch.as_tensor`, which is what
    the list-of-tensors form breaks.
    """
    adapter = _penalty_adapter()
    live = [3, 17, 29]
    order = adapter._compaction_order(_batch(live)[1], 4, SLOTS)

    tensor_history = _history(SLOTS)
    history = tensor_history.tolist() if as_list else tensor_history

    adapter._apply_penalties(_penalised(SLOTS), SLOTS, history, None, order=order, graph_rows=4)

    passed = adapter.generator.penalty_calls[-1]["prompt_tokens"]
    if as_list:
        assert isinstance(passed, list)
    else:
        assert isinstance(passed, torch.Tensor), "a tensor history must stay a tensor"
    ids = Qwen3CoderGenerator._row_token_ids(passed, 0)
    assert int(ids[0]) == 100 * int(order[0]) + 1


def test_ladder_off_leaves_the_history_exactly_as_vllm_sent_it():
    """The shipped default path must not be touched by any of the above."""
    adapter = _penalty_adapter(widths=str(SLOTS))
    history = _history(SLOTS)

    adapter._apply_penalties(_penalised(SLOTS), SLOTS, history, None, order=None, graph_rows=None)

    passed = adapter.generator.penalty_calls[-1]["prompt_tokens"]
    assert passed is history, "with the ladder off the history is passed through unchanged"


# ---------------------------------------------------------------------------
# The host-sampling demotion has to be audible.
#
# vLLM decides per request, in `check_perform_device_sampling`, whether a
# request may sample on device. On this 4-die mesh any request carrying
# `logprobs` -- including `logprobs: 0`, because the guard tests
# `max_num_logprobs is not None` before it ever looks at the value -- is routed
# to eager host sampling. That bypasses the captured trace and the width ladder
# and costs ~14x (measured: 3.595 t/s/u against 49.345), and the plugin emits no
# log line for it. The server-level `sample_on_device_mode: all` stays correct
# and stays silent.
#
# A 14x cliff whose only symptom is "the model got slow" is precisely the
# failure this port was first reported with. The adapter cannot prevent the
# demotion -- the guard lives in a repository this one must not modify -- so the
# least it must do is say so.
# ---------------------------------------------------------------------------


def _host_sampled_step(adapter):
    """One decode step with `sampling_params=None`, i.e. vLLM's host-sampled route."""
    tokens, positions = _batch([3, 17, 29])
    return adapter.decode_forward(
        tokens=tokens,
        page_table=torch.zeros((SLOTS, 8), dtype=torch.int32),
        kv_cache=adapter.kv_cache,
        start_pos=positions,
        sampling_params=None,
        reset_batch=True,
        read_from_device=False,
    )


def test_host_sampled_decode_warns_once_with_the_cause_and_the_cost():
    """The demotion must name what happened, why, and what it costs."""
    adapter = _adapter()
    with patch.object(gv.logger, "warning") as warn:
        _host_sampled_step(adapter)

    assert warn.call_count == 1, "the demotion must be reported"
    message = warn.call_args[0][0]
    for needle in ("HOST sampling", "logprobs", "3.595", "49.345", "14x"):
        assert needle in message, f"the warning must mention {needle!r}"

    assert adapter._audit["host_sampled_decode_steps"] == 1


def test_host_sampled_warning_does_not_repeat_every_step():
    """One line, not one per token -- a per-step warning would be its own defect."""
    adapter = _adapter()
    with patch.object(gv.logger, "warning") as warn:
        for _ in range(5):
            _host_sampled_step(adapter)

    assert warn.call_count == 1, "the warning must be once per server, not per step"
    assert adapter._audit["host_sampled_decode_steps"] == 5


def test_device_sampled_steps_do_not_warn():
    """The traced path must stay silent; a false alarm here trains people to ignore it."""
    adapter = _adapter()
    with patch.object(gv.logger, "warning") as warn:
        _issue(adapter, [3, 17, 29])

    assert warn.call_count == 0
    assert adapter._audit["host_sampled_decode_steps"] == 0
