# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Host-only invariants for experimental context compaction."""

import copy
import unittest

from probe_context_dedup import compact


def turn(call_id, command, output):
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": call_id, "function": {"name": "bash", "arguments": command}}],
        },
        {"role": "tool", "tool_call_id": call_id, "content": output},
    ]


class CompactionTests(unittest.TestCase):
    def test_preserves_first_copy_and_does_not_mutate_input(self):
        messages = turn("first", "cat a", "x" * 500) + turn("second", "cat a", "x" * 500)
        original = copy.deepcopy(messages)
        result, changes = compact(messages)
        self.assertEqual(messages, original)
        self.assertEqual(result[1], original[1])
        self.assertIn("first", result[3]["content"])
        self.assertEqual(len(changes), 1)

    def test_different_command_is_not_compacted(self):
        messages = turn("a", "cat a", "x" * 500) + turn("b", "cat b", "x" * 500)
        self.assertEqual(compact(messages), (messages, []))

    def test_changed_output_is_not_compacted(self):
        messages = turn("a", "cat a", "x" * 500) + turn("b", "cat a", "y" * 500)
        self.assertEqual(compact(messages), (messages, []))

    def test_short_output_is_retained(self):
        messages = turn("a", "false", "error") + turn("b", "false", "error")
        self.assertEqual(compact(messages), (messages, []))

    def test_user_and_unmatched_tool_messages_are_retained(self):
        messages = [
            {"role": "user", "content": "x" * 500},
            {"role": "tool", "tool_call_id": "unknown", "content": "x" * 500},
        ]
        self.assertEqual(compact(messages), (messages, []))


if __name__ == "__main__":
    unittest.main()
