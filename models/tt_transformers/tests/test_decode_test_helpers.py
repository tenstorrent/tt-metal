# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import unittest

from models.tt_transformers.tests.decode_test_helpers import decode_step_state, teacher_forced_decode_token


class DecodeStepStateTests(unittest.TestCase):
    def test_teacher_forced_prompt_advances_before_sampling(self):
        states = [decode_step_state(0, iteration, 5, 256) for iteration in range(6)]

        self.assertEqual([state[0] for state in states], [1, 2, 3, 4, 5, 6])
        self.assertEqual([state[1] for state in states], [1, 2, 3, 4, None, None])
        self.assertEqual([state[2] for state in states], [1, 2, 3, 4, 5, 6])

    def test_decode_position_is_rejected_before_exceeding_max_sequence_length(self):
        self.assertEqual(decode_step_state(254, 0, 5, 256), (255, 1, 1))
        self.assertEqual(decode_step_state(254, 1, 5, 256), (256, 2, 2))
        with self.assertRaisesRegex(ValueError, "decode position 256"):
            decode_step_state(254, 2, 5, 256)

    def test_reference_token_teacher_forces_a_shared_trajectory(self):
        reference_token = object()
        device_token = object()

        selected_token = teacher_forced_decode_token(
            reference_token=reference_token,
            device_token=device_token,
        )

        self.assertIs(selected_token, reference_token)

    def test_device_token_is_used_without_a_reference_model(self):
        device_token = object()

        self.assertIs(teacher_forced_decode_token(device_token=device_token), device_token)

    def test_decode_cannot_continue_without_a_sampled_token(self):
        with self.assertRaisesRegex(ValueError, "reference or device token"):
            teacher_forced_decode_token()


if __name__ == "__main__":
    unittest.main()
