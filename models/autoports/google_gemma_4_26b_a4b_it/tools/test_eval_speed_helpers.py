# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Host-only invariants for experimental context compaction."""

import copy
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from probe_context_dedup import compact
from probe_loop_recovery import wall_deadline


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


class WeightControlTests(unittest.TestCase):
    def test_only_configurable_weights_change(self):
        root = Path(__file__).resolve().parents[1]
        source = root / "doc/datatype_sweep/selected_precision_config.json"
        original = source.read_bytes()
        baseline = json.loads(original)
        with tempfile.TemporaryDirectory(prefix="gemma4-weight-control-test-") as directory:
            output = Path(directory) / "policy.json"
            subprocess.run(
                [
                    sys.executable,
                    str(root / "tools/prepare_eval_weight_control.py"),
                    "--source",
                    str(source),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
            )
            candidate = json.loads(output.read_text())
            changes = json.loads(output.with_suffix(".manifest.json").read_text())["changes"]
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(len(changes), 93)
        for change in changes:
            self.assertNotIn("fixed", change["path"].split("."))
            self.assertEqual((change["before"], change["after"]), ("bfloat4_b", "bfloat8_b"))
            node = candidate
            parts = change["path"].split(".")
            for part in parts[:-1]:
                node = node[part]
            node[parts[-1]] = change["before"]
        candidate["config_id"] = baseline["config_id"]
        self.assertEqual(candidate, baseline)


class DiagnosticDeadlineTests(unittest.TestCase):
    def test_interrupts_blocked_read_and_restores_timer(self):
        state = {"expired": False}
        with wall_deadline(0.02, state):
            time.sleep(0.2)
            self.fail("Deadline did not interrupt blocking work")
        self.assertTrue(state["expired"])
        next_state = {"expired": False}
        with wall_deadline(0.1, next_state):
            pass
        self.assertFalse(next_state["expired"])


if __name__ == "__main__":
    unittest.main()
