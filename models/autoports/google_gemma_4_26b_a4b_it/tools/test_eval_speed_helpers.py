# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Host-only invariants for experimental context compaction."""

import copy
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from probe_context_dedup import compact
from probe_loop_recovery import wall_deadline
from probe_native_eval_action import request_messages
from replay_eval_requests import post
from summarize_swe_suite import counter_delta


class SuiteCounterTests(unittest.TestCase):
    def test_only_matched_complete_counter_window_is_accepted(self):
        zero = {
            "request_success_total": 0,
            "time_to_first_token_seconds_count": 0,
            "time_to_first_token_seconds_sum": 0,
            "e2e_request_latency_seconds_sum": 0,
            "generation_tokens_total": 0,
            "prompt_tokens_total": 0,
        }
        final = dict(zip(zero, (1, 1, 2, 5, 3, 10)))
        events = [
            {"event": "server_metrics", "phase": "before_request", "counters": zero},
            {"event": "server_metrics", "phase": "after_response", "counters": final},
        ]
        responses = [{"usage": {"completion_tokens": 3, "prompt_tokens": 10}}]
        result = counter_delta(events, responses)
        self.assertTrue(result["valid"])
        self.assertEqual((result["ttft_s"], result["post_first_token_s"]), (2, 3))
        final["request_success_total"] = 0
        self.assertFalse(counter_delta(events, responses)["valid"])
        self.assertFalse(counter_delta([], responses)["valid"])


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


class NativeReplayTests(unittest.TestCase):
    def test_retains_both_reasoning_fields_but_not_local_metadata(self):
        messages = [
            {"role": "assistant", "content": None, "reasoning_content": "plan", "extra": {"timestamp": 1}},
            {"role": "assistant", "reasoning": "other plan", "provider_specific_fields": {}},
        ]
        original = copy.deepcopy(messages)
        self.assertEqual(
            request_messages(messages),
            [
                {"role": "assistant", "content": None, "reasoning_content": "plan"},
                {"role": "assistant", "reasoning": "other plan"},
            ],
        )
        self.assertEqual(messages, original)


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

    def test_interrupts_http_before_response_headers(self):
        class SlowHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                time.sleep(0.15)
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass

        with HTTPServer(("127.0.0.1", 0), SlowHandler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                state = {"expired": False}
                with wall_deadline(0.02, state):
                    with post(f"http://127.0.0.1:{server.server_port}", "/", {}) as response:
                        response.read()
                self.assertTrue(state["expired"])
            finally:
                server.shutdown()
                thread.join()


if __name__ == "__main__":
    unittest.main()
