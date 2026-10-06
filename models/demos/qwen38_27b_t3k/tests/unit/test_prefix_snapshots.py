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

    def __init__(self, label, rows=4):
        self.label, self.rows = label, rows

    def __getitem__(self, key):
        return f"{self.label}[{key.start}]"


class PrefixSnapshotTests(unittest.TestCase):
    LAYERS = 3

    def setUp(self):
        self.copies = []
        self.ops = SimpleNamespace(
            zeros_like=lambda tensor: f"zeros({tensor})",
            clone=lambda tensor: f"clone({tensor})",
            copy=lambda source, target: self.copies.append((source, target)),
            concat=lambda parts, **kw: ("concat", tuple(parts)),
        )
        self.methods, self.module = load_methods(METHODS, self.ops)
        layers = [SimpleNamespace(conv=Row(f"conv{i}"), recurrent=Row(f"rec{i}")) for i in range(self.LAYERS)]
        self.gen = SimpleNamespace(
            cache=SimpleNamespace(batch_size=4, layers=layers),
            model=SimpleNamespace(_resident_decode_bucket=None),
            counters=Counter(),
            _slot_prefix_len=[0, 0, 0, 0],
            _state_snapshots={},
            _next_snapshot_handle=0,
            _recurrent_reset_warmed=None,
            _release_traces=lambda **kw: None,
        )
        for name in METHODS:
            setattr(self.gen, name, MethodType(self.methods[name], self.gen))
        self.gen._recurrent_reset_warmed = self.gen.cache

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
        self.assertEqual(rows[(0, "conv")], "clone(conv0[2])")
        self.assertEqual(rows[(2, "recurrent")], "clone(rec2[2])")

    def test_the_store_refuses_rather_than_evicting_something_still_referenced(self):
        self.gen._slot_prefix_len[0] = 64
        handles = [self.gen.save_slot_state(0) for _ in range(self.module.MAX_PREFIX_SNAPSHOTS)]
        self.assertNotIn(None, handles)
        self.assertEqual(len(set(handles)), len(handles))
        self.assertIsNone(self.gen.save_slot_state(0))

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
        self.gen.restore_slot_state(3, handle)
        # One rebuilt tensor per conv and recurrent on every layer.
        self.assertEqual(len(self.copies), 2 * self.LAYERS)
        parts = dict(self.copies)
        rebuilt = [source for source, _ in self.copies]
        self.assertTrue(all(kind == "concat" for kind, _ in rebuilt))
        # Slot 3's position carries the saved row; the other rows are the slot's own.
        first = rebuilt[0][1]
        self.assertEqual(first[3], "clone(conv0[0])")
        self.assertEqual(first[1], "conv0[1]")
        self.assertEqual(len(parts), 2 * self.LAYERS)

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
