# SPDX-License-Identifier: Apache-2.0
"""CPU-only source/route guards; importing this test never imports TTNN."""

import ast
import importlib.util
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("padding_candidate", HERE / "benchmark_decode_padding_route_candidate.py")
CANDIDATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CANDIDATE)


class PaddingRouteTests(unittest.TestCase):
    def source(self):
        text = (HERE.parent / "tt/multichip_decoder.py").read_text()
        cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == "ParallelMoE")
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "group_packed")
        return "\n".join(text.splitlines()[fn.lineno - 1 : fn.end_lineno])

    def test_exact_source_insertion_and_shared_consumers(self):
        original = self.source()
        baseline = CANDIDATE.build_source(original, False)
        masked = CANDIDATE.build_source(original, True)
        gate = "    selected = ttnn.where(self._probe_active, selected, 0.0)\n"
        self.assertEqual(masked.count(gate), 1)
        self.assertEqual(masked.replace(gate, ""), baseline)
        ast.parse(masked)
        self.assertLess(masked.index(gate), masked.index("scores = ttnn.to_layout(selected"))
        self.assertEqual(masked.count("[source, ids, scores, self.rank, counts, mapping, capacity]"), 1)
        self.assertEqual(masked.count("[down, ids, scores, self.rank, mapping, compact]"), 1)
        self.assertIn("n != 8", masked)

    def test_source_drift_rejected(self):
        with self.assertRaises(ValueError):
            CANDIDATE.build_source(self.source().replace("n = x.shape[2]", "n = 8"), True)

    def test_counts_zero_padding_and_rank_partition(self):
        ids = [[(row * 64 + route) % 512 for route in range(32)] for row in range(8)]
        scores = [[0.125] * 10 + [0] * 22 for _ in range(8)]
        for active in (5, 6, 7, 8):
            with self.subTest(active=active):
                counts = [CANDIDATE.route_counts(ids, scores, rank, active) for rank in range(4)]
                self.assertEqual(sum(map(sum, counts)), 10 * active)
                gated = [row if i < active else [0] * 32 for i, row in enumerate(scores)]
                self.assertEqual(counts, [CANDIDATE.route_counts(ids, gated, rank) for rank in range(4)])

    def test_overlap_does_not_promise_weight_saving(self):
        ids = [list(range(32)) for _ in range(8)]
        scores = [[0.125] * 10 + [0] * 22 for _ in range(8)]
        before = CANDIDATE.route_counts(ids, scores, 0)
        after = CANDIDATE.route_counts(ids, scores, 0, 5)
        self.assertEqual(sum(x > 0 and y == 0 for x, y in zip(before, after)), 0)
        self.assertEqual(sum(before) - sum(after), 30)


if __name__ == "__main__":
    unittest.main()
