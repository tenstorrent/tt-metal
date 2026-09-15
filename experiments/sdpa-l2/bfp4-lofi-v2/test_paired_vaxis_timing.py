"""CPU/static timing-schedule, memory-budget and failure-cleanup tests."""

import ast
import collections
import unittest
from pathlib import Path
from types import SimpleNamespace

PATH = Path(__file__).with_name("paired_vaxis_timing.py")


def helpers(ttnn=None):
    tree = ast.parse(PATH.read_text())
    names = {"candidates", "round_order", "memory_estimate", "release_captures"}
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = dict(ttnn=ttnn)
    exec(compile(tree, str(PATH), "exec"), scope)
    return SimpleNamespace(**{name: scope[name] for name in names})


class PairedTiming(unittest.TestCase):
    def test_round_order(self):
        helper = helpers()
        for count in (2, 4, 6):
            positions = collections.Counter()
            for ordinal in range(2 * count):
                order = helper.round_order(count, ordinal)
                self.assertEqual(sorted(order), list(range(count)))
                for position, candidate in enumerate(order):
                    positions[candidate, position] += 1
                if ordinal % 2:
                    self.assertEqual(order, list(reversed(helper.round_order(count, ordinal - 1))))
            self.assertEqual(set(positions.values()), {2})

    def test_candidate_bounds_and_memory(self):
        helper = helpers()
        choices = helper.candidates([2, 8, 16])
        self.assertEqual(choices, [("D", 2), ("N", 2), ("D", 8), ("N", 8), ("D", 16), ("N", 16)])
        for invalid in ([], [2, 2], [2, 8, 16, 0], [0], [64]):
            with self.assertRaises(AssertionError):
                helper.candidates(invalid)
        small = helper.memory_estimate(32768, 10, choices, 64 * 1024**2)
        large = helper.memory_estimate(262144, 10, choices, 64 * 1024**2)
        self.assertEqual(large["tensor_bytes"], 8 * small["tensor_bytes"])
        self.assertEqual(large["tensor_bytes"], 25417482240)
        self.assertLess(large["total_budget_bytes"], 24 * 1024**3)
        fewer = helper.memory_estimate(262144, 10, helper.candidates([2, 8]), 64 * 1024**2)
        self.assertEqual(3 * fewer["tensor_bytes"], 2 * large["tensor_bytes"])

    def test_cleanup_attempts_every_trace_even_after_failure(self):
        calls, records = [], []

        def end(device, trace, **kwargs):
            calls.append(("end", trace))
            raise RuntimeError("capture failure")

        def release(device, trace):
            calls.append(("release", trace))
            if trace == 11:
                raise RuntimeError("release failure")

        helper = helpers(SimpleNamespace(end_trace_capture=end, release_trace=release))
        errors = helper.release_captures(None, [10, 11, 12], 12, records.append)
        self.assertEqual(calls, [("end", 12), ("release", 12), ("release", 11), ("release", 10)])
        self.assertEqual(len(errors), 2)
        self.assertEqual(records[0]["trace_count"], 3)


if __name__ == "__main__":
    unittest.main()
