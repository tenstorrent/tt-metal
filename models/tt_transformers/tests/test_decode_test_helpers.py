# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import unittest

from models.tt_transformers.tests.decode_test_helpers import decode_step_state


class DecodeStepStateTests(unittest.TestCase):
    def test_teacher_forced_prompt_advances_before_sampling(self):
        states = [decode_step_state(0, iteration, 5, 256) for iteration in range(6)]

        self.assertEqual([state[0] for state in states], [1, 2, 3, 4, 5, 6])
        self.assertEqual([state[1] for state in states], [1, 2, 3, 4, None, None])
        self.assertEqual([state[2] for state in states], [1, 2, 3, 4, 5, 6])

    def test_cache_window_is_capped_at_max_sequence_length(self):
        self.assertEqual(decode_step_state(254, 0, 5, 256), (255, 1, 1))
        self.assertEqual(decode_step_state(254, 1, 5, 256), (256, 2, 2))
        self.assertEqual(decode_step_state(254, 2, 5, 256), (257, 3, 2))


if __name__ == "__main__":
    unittest.main()
