# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only metadata validation; device math needs sampling_state_probe."""

import subprocess
import sys
import unittest
from pathlib import Path

from models.autoports.ornith_ai_ornith_1_5_9b.tt.sampling_state import (
    normalize_histories,
    seed_selection_indices,
    validate_slots,
)


class SamplingStateTests(unittest.TestCase):
    def test_remap_covers_dormant_offbatch_slots(self):
        permutation = [1, 0, *range(3, 32), 2]
        self.assertEqual(validate_slots(permutation, permutation=True), tuple(permutation))
        for invalid in ([1, 0], list(range(31)) + [0], list(range(31)) + [32], [-1, *range(31)]):
            with self.assertRaises(ValueError):
                validate_slots(invalid, permutation=True)

    def test_seed_reset_and_inactive_keep_are_complementary(self):
        active = (0, 3, 31)
        current = tuple(range(32))
        saved = tuple(range(100, 132))
        combined = current + saved
        reset = tuple(combined[i] for i in seed_selection_indices(active, choose_second=True))
        preserved = tuple(combined[i] for i in seed_selection_indices(active, choose_second=False))
        for slot in range(32):
            self.assertEqual(reset[slot], saved[slot] if slot in active else current[slot])
            self.assertEqual(preserved[slot], current[slot] if slot in active else saved[slot])
        self.assertEqual(seed_selection_indices([], choose_second=False), tuple(range(32, 64)))
        self.assertEqual(seed_selection_indices(range(32), choose_second=False), tuple(range(32)))

    def test_histories_are_packed_in_supplied_slot_order(self):
        rows = normalize_histories([[9, 9, -1], [2]], slots=[31, 0], vocab_size=16)
        self.assertEqual(rows, ((9, 9), (2,)))
        self.assertEqual(normalize_histories(None, slots=[1, 3], vocab_size=16), ((), ()))
        for tokens in ([[1]], [[1], [16]], [[1], [-2]], [[True], [1]]):
            with self.assertRaises(ValueError):
                normalize_histories(tokens, slots=[31, 0], vocab_size=16)

    def test_no_device_dependency_during_metadata_import(self):
        # Repository pytest fixtures may already import device packages. Isolate
        # the production module import, including its transitive dependencies.
        # -I/-S exclude inherited Python paths and site initialization; only the
        # explicit checkout root is added, so this check also works outside cwd.
        probe = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "__import__(sys.argv[2]); "
            "found = sorted(name for name in sys.modules "
            "if name.split('.')[0] in ('ttnn', 'torch', 'vllm')); "
            "assert not found, 'Host-only module imported device packages: ' + repr(found)"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-c",
                probe,
                str(Path(__file__).resolve().parents[4]),
                "models.autoports.ornith_ai_ornith_1_5_9b.tt.sampling_state",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()
