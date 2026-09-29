# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU behavioral checks for the actual profile controller; no TTNN import."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

_PATH = Path(__file__).parents[1] / "models/ltx/audio_profile_drains.py"
_SPEC = importlib.util.spec_from_file_location("audio_profile_drains", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
ProfileDrains, ProfileSegmentComplete = _MODULE.ProfileDrains, _MODULE.ProfileSegmentComplete


class Module:
    def __init__(self, clock, cost=0, children=()):
        self.clock, self.cost, self.children = clock, cost, dict(children)

    def named_children(self):
        return self.children.items()

    def __call__(self, value):
        return self.forward(value)

    def forward(self, value):
        for child in self.children.values():
            value = child(value)
        self.clock[0] += self.cost
        return value + self.cost


class Leaf(Module):
    pass


class ProfileDrainContracts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.clock, self.reads = [0], []

    def controller(self, roots, types=(Module,), stop=None, read=None, minimum_gap=1):
        drains = ProfileDrains(
            roots,
            types,
            read or (lambda: self.reads.append(self.clock[0])),
            lambda: self.clock[0],
            Path(self.directory.name),
            stop_module=stop,
            minimum_operation_gap=minimum_gap,
        )
        self.addCleanup(drains.close)
        return drains

    def test_nested_forward_preserves_output_and_drains_tail(self):
        root = Module(self.clock, 2, [("a", Leaf(self.clock, 3)), ("b", Leaf(self.clock, 7))])
        drains = self.controller((("audio", root),))
        self.assertEqual(root(11), 23)
        self.assertEqual(self.reads, [3, 10, 12])
        self.assertEqual([r["operation_gap"] for r in drains.events], [3, 7, 2])
        self.assertEqual(drains.summary()["max_operation_gap"], 7)
        rows = [json.loads(line) for line in drains.events_path.read_text().splitlines()]
        self.assertEqual(rows, drains.events)

    def test_zero_gap_ancestors_do_not_reread_mesh(self):
        root = Module(self.clock, 0, [("leaf", Leaf(self.clock, 5))])
        drains = self.controller((("audio", root),))
        root(0)
        self.assertEqual(self.reads, [5])
        drains.drain("boundary", force=True)
        self.assertEqual(self.reads, [5, 5])

    def test_first_segment_drains_then_stops_before_later_module(self):
        first, later = Module(self.clock, 1, [("leaf", Leaf(self.clock, 4))]), Leaf(self.clock, 99)
        root = Module(self.clock, 0, [("first", first), ("later", later)])
        drains = self.controller((("audio", root),), stop=first)
        with self.assertRaises(ProfileSegmentComplete):
            root(0)
        self.assertEqual(self.reads, [4, 5])
        self.assertEqual(self.clock[0], 5)
        self.assertEqual(drains.completed_segment, "audio.first")
        drains.close()
        self.assertEqual(root(0), 104)

    def test_weight_preparation_must_keep_root_and_child_identity(self):
        root = Module(self.clock, children=[("leaf", Leaf(self.clock))])
        roots = (("audio", root),)
        drains = self.controller(roots)
        drains.assert_same_modules(roots)
        with self.assertRaisesRegex(AssertionError, "identities changed"):
            drains.assert_same_modules((("audio", Module(self.clock)),))
        root.children["leaf"] = Leaf(self.clock)
        with self.assertRaisesRegex(AssertionError, "identities changed"):
            drains.assert_same_modules(roots)

    def test_shared_modules_wrapped_once_and_instance_forward_restored(self):
        shared = Leaf(self.clock)
        original = lambda value: value + 17
        shared.forward = original
        root = Module(self.clock, children=[("a", shared), ("b", shared)])
        drains = self.controller((("audio", root),))
        self.assertEqual(len(drains.coverage), 2)
        self.assertEqual(root(0), 34)
        drains.close()
        self.assertIs(shared.forward, original)
        self.assertNotIn("forward", vars(root))
        drains.close()  # outer finalizers may repeat cleanup

    def test_device_drain_failure_is_not_segment_success(self):
        root = Leaf(self.clock, 4)

        def fail():
            raise RuntimeError("profiler read failed")

        drains = self.controller((("audio", root),), stop=root, read=fail)
        with self.assertRaisesRegex(RuntimeError, "profiler read failed"):
            root(0)
        self.assertIsNone(drains.completed_segment)
        self.assertFalse(drains.events)

    def test_refuse_stale_output_or_missing_hooks(self):
        root = Module(self.clock)
        drains = self.controller((("audio", root),))
        with self.assertRaises(FileExistsError):
            self.controller((("audio", root),))
        drains.close()
        drains.events_path.unlink()
        with self.assertRaisesRegex(AssertionError, "no audio profile hooks"):
            self.controller((("audio", root),), types=(Leaf,))
        self.assertNotIn("forward", vars(root))

    def test_accumulates_small_leaf_ops_and_forces_segment_tail(self):
        # Job092's23 tiny intervals covered only79 ops;32-chip reads cost~6s.
        # Keep those same operations while batching the expensive native reads.
        costs = [3] * 22 + [13]
        root = Module(self.clock, children=[(str(i), Leaf(self.clock, n)) for i, n in enumerate(costs)])
        drains = self.controller((("audio", root),), stop=root, minimum_gap=32)
        with self.assertRaises(ProfileSegmentComplete):
            root(0)
        self.assertEqual(self.clock[0], 79)
        self.assertEqual(self.reads, [33, 66, 79])
        self.assertEqual([row["operation_gap"] for row in drains.events], [33, 33, 13])
        self.assertEqual(sum(row["operation_gap"] for row in drains.events), 79)
        self.assertEqual(drains.summary()["minimum_operation_gap"], 32)

    def test_large_leaf_overshoot_is_observed_not_hidden(self):
        root = Module(self.clock, 2, [("small", Leaf(self.clock, 5)), ("large", Leaf(self.clock, 70))])
        drains = self.controller((("audio", root),), stop=root, minimum_gap=32)
        with self.assertRaises(ProfileSegmentComplete):
            root(0)
        self.assertEqual(self.reads, [75, 77])
        self.assertEqual(drains.summary()["max_operation_gap"], 75)

    def test_invalid_batching_threshold_fails_before_hooks(self):
        root = Module(self.clock)
        for gap in (0, 65, 1.5):
            with self.assertRaises(AssertionError):
                self.controller((("audio", root),), minimum_gap=gap)
        self.assertNotIn("forward", vars(root))


if __name__ == "__main__":
    unittest.main()
