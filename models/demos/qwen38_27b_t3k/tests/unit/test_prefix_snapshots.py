# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Saving and restoring one slot's GDN state, without a device.

A snapshot is only meaningful as the summary of a specific token prefix, so the length it stands
for travels with the rows and is reinstated on restore. Everything here is bookkeeping around the
clone and copy that `_capture` already performs; the device behaviour belongs in a device test.
"""

import importlib
import unittest
from collections import Counter
from functools import wraps
from types import MethodType, SimpleNamespace
from unittest.mock import patch

METHODS = (
    "remap_recurrent_slots",
    "_reserve_snapshot_pool",
    "save_slot_state",
    "restore_slot_state",
    "free_slot_state",
    "_write_recurrent_slots",
    "reset_recurrent_slots",
)


def load_methods(names, ops):
    module = importlib.import_module("models.demos.qwen38_27b_t3k.tt.generator")
    cls = module.Qwen38Generator

    def bind_ops(method):
        @wraps(method)
        def invoke(*args, **kwargs):
            with patch.object(module, "ttnn", ops):
                return method(*args, **kwargs)

        return invoke

    return {name: bind_ops(getattr(cls, name)) for name in names}, module


class Row:
    """Stands in for one layer's conv or recurrent tensor, sliceable by row."""

    def __init__(self, label, rows=4, layout="tile"):
        self.label, self.rows, self.layout = label, rows, layout
        self.shape = (rows, 2, 2)

    def __getitem__(self, key):
        return f"{self.label}[{key.start}]"


class Buffer:
    """A reserved snapshot row. A save overwrites what it holds rather than allocating."""

    def __init__(self, content):
        self.content = content

    def __repr__(self):
        return f"Buffer({self.content})"


class PrefixSnapshotTests(unittest.TestCase):
    LAYERS = 3

    def setUp(self):
        self.copies, self.fills = [], []
        self.ops = SimpleNamespace(
            zeros_like=lambda tensor: f"zeros({tensor})",
            clone=lambda tensor: Buffer(tensor),
            slice=self.write_row,
            fill_cache=lambda tensor, row, slot: self.fills.append((tensor.label, row, slot)),
            TILE_LAYOUT="tile",
            copy=lambda source, target: self.copies.append((source, target)),
            concat=lambda parts, **kw: ("concat", tuple(parts)),
        )
        self.methods, self.module = load_methods(METHODS, self.ops)
        layers = [
            SimpleNamespace(conv=Row(f"conv{i}", layout="row_major"), recurrent=Row(f"rec{i}", layout="tile"))
            for i in range(self.LAYERS)
        ]
        self.gen = SimpleNamespace(
            cache=SimpleNamespace(batch_size=4, layers=layers),
            model=SimpleNamespace(_resident_decode_bucket=None),
            counters=Counter(),
            _slot_prefix_len=[0, 0, 0, 0],
            _state_snapshots={},
            _snapshot_pool=[],
            _next_snapshot_handle=0,
            _recurrent_reset_warmed=None,
            _release_traces=lambda **kw: None,
        )
        for name in METHODS:
            setattr(self.gen, name, MethodType(self.methods[name], self.gen))
        self.gen._recurrent_reset_warmed = self.gen.cache
        self.gen._reserve_snapshot_pool()

    def write_row(self, tensor, start, end, output_tensor=None):
        """Stand in for ttnn.slice writing one row into an already-reserved buffer."""
        assert output_tensor is not None, "a save must never allocate its own destination"
        assert (start[0], end[0]) == (end[0] - 1, start[0] + 1), (start, end)
        output_tensor.content = f"{tensor.label}[{start[0]}]"
        return output_tensor

    def test_a_slot_holding_nothing_has_no_snapshot_worth_taking(self):
        self.assertIsNone(self.gen.save_slot_state(1))

    def test_a_slot_left_indeterminate_by_a_failed_chunk_is_not_saved(self):
        self.gen._slot_prefix_len[1] = -1
        self.assertIsNone(self.gen.save_slot_state(1))

    def test_a_held_prefix_is_saved_and_the_length_travels_with_it(self):
        self.gen._slot_prefix_len[2] = 4096
        handle = self.gen.save_slot_state(2)
        self.assertIsNotNone(handle)
        held, rows = self.gen._state_snapshots[handle]
        self.assertEqual(held, 4096)
        # One row per conv and recurrent on every layer, taken from slot 2.
        self.assertEqual(len(rows), 2 * self.LAYERS)
        self.assertEqual(rows[(0, "conv")].content, "conv0[2]")
        self.assertEqual(rows[(2, "recurrent")].content, "rec2[2]")

    def test_the_store_refuses_rather_than_evicting_something_still_referenced(self):
        self.gen._slot_prefix_len[0] = 64
        handles = [self.gen.save_slot_state(0) for _ in range(self.module.MAX_PREFIX_SNAPSHOTS)]
        self.assertNotIn(None, handles)
        self.assertEqual(len(set(handles)), len(handles))
        self.assertIsNone(self.gen.save_slot_state(0))

    def test_the_whole_store_is_reserved_before_any_request_can_need_it(self):
        # Allocation is what a live decode trace forbids, so every snapshot is taken at bind time.
        self.assertEqual(len(self.gen._snapshot_pool), self.module.MAX_PREFIX_SNAPSHOTS)
        self.assertEqual(len(self.gen._snapshot_pool[0]), 2 * self.LAYERS)
        reserved = {id(row) for rows in self.gen._snapshot_pool for row in rows.values()}
        self.gen._slot_prefix_len[0] = 64
        saved = self.gen._state_snapshots[self.gen.save_slot_state(0)][1]
        self.assertTrue({id(row) for row in saved.values()} <= reserved)

    def test_freeing_a_snapshot_returns_its_buffers_for_the_next_save(self):
        self.gen._slot_prefix_len[0] = 64
        handles = [self.gen.save_slot_state(0) for _ in range(self.module.MAX_PREFIX_SNAPSHOTS)]
        self.assertEqual(self.gen._snapshot_pool, [])
        rows = self.gen._state_snapshots[handles[0]][1]
        self.gen.free_slot_state(handles[0])
        self.assertEqual(self.gen._snapshot_pool, [rows])
        self.assertIsNotNone(self.gen.save_slot_state(0))

    def test_restoring_an_unknown_handle_reports_failure_without_touching_the_slot(self):
        self.assertFalse(self.gen.restore_slot_state(1, 999))
        self.assertEqual(self.copies, [])
        self.assertEqual(self.gen._slot_prefix_len, [0, 0, 0, 0])

    def test_a_restored_slot_holds_the_prefix_the_snapshot_stood_for(self):
        self.gen._slot_prefix_len[0] = 512
        handle = self.gen.save_slot_state(0)
        self.gen.reset_recurrent_slots([3])
        self.assertTrue(self.gen.restore_slot_state(3, handle))
        self.assertEqual(self.gen._slot_prefix_len[3], 512)
        self.assertEqual(self.gen._slot_prefix_len[0], 512)

    def test_a_restore_writes_the_saved_rows_into_the_target_slot(self):
        self.gen._slot_prefix_len[0] = 512
        handle = self.gen.save_slot_state(0)
        self.copies.clear()
        self.fills.clear()
        self.gen.restore_slot_state(3, handle)
        # Tiled state takes one in-place row write per layer; row-major state has no such op, so
        # its whole tensor is still rebuilt.
        self.assertEqual(len(self.fills), self.LAYERS)
        self.assertEqual(len(self.copies), self.LAYERS)
        self.assertEqual([(label, slot) for label, _, slot in self.fills], [(f"rec{i}", 3) for i in range(self.LAYERS)])
        self.assertEqual(self.fills[0][1].content, "rec0[0]")
        rebuilt = [source for source, _ in self.copies]
        self.assertTrue(all(kind == "concat" for kind, _ in rebuilt))
        # Slot 3's position carries the saved row; the other rows are the slot's own.
        first = rebuilt[0][1]
        self.assertEqual(first[3].content, "conv0[0]")
        self.assertEqual(first[1], "conv0[1]")

    def test_a_tiled_row_write_does_not_scale_with_how_many_slots_the_cache_holds(self):
        # The rebuild is O(batch) per layer; this is the path that stops restore growing with it.
        self.gen._slot_prefix_len[0] = 512
        handle = self.gen.save_slot_state(0)
        self.fills.clear()
        self.gen.cache.batch_size = 16
        self.gen.restore_slot_state(3, handle)
        self.assertEqual(len(self.fills), self.LAYERS)

    def test_a_remap_carries_each_slot_prefix_with_the_row_it_describes(self):
        # remap_recurrent_slots rebuilds row j from old row order[j]; the length each row stands
        # for has to travel the same way or the continuity gate judges the wrong request.
        self.gen._slot_prefix_len = [10, 20, 30, 40]
        self.gen._release_traces = lambda **kw: None
        self.gen.remap_recurrent_slots([2, 0, 3, 1])
        self.assertEqual(self.gen._slot_prefix_len, [30, 10, 40, 20])

    def test_an_identity_remap_leaves_the_prefix_lengths_alone(self):
        self.gen._slot_prefix_len = [10, 20, 30, 40]
        self.gen.remap_recurrent_slots([0, 1, 2, 3])
        self.assertEqual(self.gen._slot_prefix_len, [10, 20, 30, 40])

    def test_a_freed_handle_can_no_longer_be_restored(self):
        self.gen._slot_prefix_len[0] = 64
        handle = self.gen.save_slot_state(0)
        self.gen.free_slot_state(handle)
        self.assertFalse(self.gen.restore_slot_state(1, handle))

    def test_freeing_an_unknown_handle_is_harmless(self):
        self.gen.free_slot_state(4242)

    def test_a_slot_outside_the_cache_is_refused(
        self,
    ):
        for call in (lambda: self.gen.save_slot_state(4), lambda: self.gen.restore_slot_state(-1, 0)):
            with self.assertRaises(ValueError):
                call()

    def test_the_capacity_is_published_for_the_caller_to_size_its_index(self):
        self.assertEqual(
            self.module.Qwen38Generator.prefix_snapshot_capacity.fget(self.gen),
            self.module.MAX_PREFIX_SNAPSHOTS,
        )


if __name__ == "__main__":
    unittest.main()
