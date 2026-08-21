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


def test_disabled_by_default_does_not_queue_and_changes_nothing():
    """With `QWEN3_DECODE_WIDTHS` unset the adapter must not permute at all.

    It must also not use the queue: with no compaction there is nothing to pair,
    so the shipped path takes none of the bookkeeping and, more importantly,
    cannot raise any of the pairing errors below.
    """
    with patch.dict("os.environ", {}, clear=False):
        import os

        os.environ.pop("QWEN3_DECODE_WIDTHS", None)
        adapter = Qwen3CoderForCausalLM(_FakeGenerator(), max_model_len=4096, max_num_seqs=SLOTS)
    adapter.kv_cache = ["fake-cache"]

    assert adapter._decode_widths == [SLOTS]
    assert adapter._compaction_enabled is False
    handle = _issue(adapter, [3, 17, 29])
    assert adapter._compaction is None, "no compaction may happen when the feature is off"
    assert len(adapter._pending_orders) == 0, "the disabled path must not touch the queue"
    tokens = _read(adapter, handle)
    # Straight through: graph row r is slot r.
    for row in (0, 3, 17, 29):
        assert tokens[row] == 1000 + row, tokens[[0, 3, 17, 29]]


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
