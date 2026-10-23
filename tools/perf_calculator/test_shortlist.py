# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
import copy
import unittest
from shortlist import make_shortlist


def row(group, time):
    return dict(configuration={"operation": "example_copy", "batch": group}, estimated_us=time)


class ShortlistTests(unittest.TestCase):
    def test_deduplicate_configurations_not_equal_times(self):
        rows = [row(3, 5), row(3, 5), row(4, 5), row(2, 6)]
        out = make_shortlist(rows, 2, {})
        self.assertEqual(out["distinct_candidates"], 3)
        self.assertEqual({c["prediction"]["configuration"]["batch"] for c in out["candidates"]}, {3, 4})
        self.assertEqual(out, make_shortlist(list(reversed(rows)), 2, {}))

    def test_conflicting_duplicate_rejected(self):
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            make_shortlist([row(3, 5), row(3, 6)], 2, {})

    def test_source_predictions_not_mutated(self):
        rows = [row(3, 5), row(4, 6)]
        old = copy.deepcopy(rows)
        out = make_shortlist(rows, 4, {"source": "test"})
        self.assertEqual(len(out["candidates"]), 2)
        self.assertEqual(rows, old)
        self.assertEqual(out["provenance"], {"source": "test"})

    def test_invalid_count_rejected(self):
        with self.assertRaises(ValueError):
            make_shortlist([row(3, 5)], 0, {})


if __name__ == "__main__":
    unittest.main()
