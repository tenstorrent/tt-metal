# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Host-only invariants for experimental context compaction."""

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from measure_context_dedup import canonical, common_prefix_blocks
from probe_context_dedup import compact, compact_warning_blocks
from probe_loop_recovery import wall_deadline
from probe_native_eval_action import forwarded_feedback, request_messages
from replay_eval_requests import post
from run_local_eval_probe import sampling_kwargs
from summarize_swe_suite import completed_response_counters, counter_delta, deadline_audit


class DiagnosticSamplingTests(unittest.TestCase):
    def test_native_replay_restores_only_selected_request_feedback(self):
        events = [
            {"event": "request_start", "request_id": "old", "message_count": 4},
            {"event": "repeated_tool_feedback", "request_id": "old", "feedback": "old note"},
            {"event": "request_start", "request_id": "new", "message_count": 6},
            {"event": "repeated_tool_feedback", "request_id": "new", "feedback": "current note"},
        ]
        self.assertEqual(forwarded_feedback(events, 6), [{"role": "user", "content": "current note"}])
        with self.assertRaises(ValueError):
            forwarded_feedback(events, 8)
        events.append({"event": "reasoning_history_limited", "request_id": "new"})
        with self.assertRaises(ValueError):
            forwarded_feedback(events, 6)

    def test_default_does_not_change_or_alias_source_policy(self):
        original = {"config": {"model": {"model_kwargs": {"temperature": 1, "max_tokens": 32768}}}}
        candidate = sampling_kwargs(original)
        self.assertEqual(candidate, original)
        candidate["config"]["model"]["model_kwargs"]["max_tokens"] = 1
        self.assertEqual(original["config"]["model"]["model_kwargs"]["max_tokens"], 32768)

    def test_explicit_greedy_control_changes_only_temperature(self):
        model = {"temperature": 1, "top_p": 0.95, "max_tokens": 32768, "extra_body": {"top_k": 20}}
        original = {"config": {"model": {"model_kwargs": model}}}
        actual = sampling_kwargs(original, 0)["config"]["model"]["model_kwargs"]
        self.assertEqual(actual, {**model, "temperature": 0})
        self.assertEqual(model["temperature"], 1)
        for invalid in (-1, 3, float("nan")):
            with self.assertRaises(ValueError):
                sampling_kwargs(original, invalid)


class OfflineContextTests(unittest.TestCase):
    def test_warning_compaction_preserves_first_and_nonwarning_evidence(self):
        warning = "/testbed/module.py:12: DeprecationWarning: " + "deprecated interface " * 10 + "\n  import old\n"
        messages = [
            {"role": "tool", "tool_call_id": "call1", "content": warning + "test PASS\n"},
            {"role": "tool", "tool_call_id": "call2", "content": warning + "test FAIL\nTraceback: problem\n"},
            {"role": "assistant", "content": warning, "reasoning": "retain this"},
        ]
        original = copy.deepcopy(messages)
        candidate, changes = compact_warning_blocks(messages)
        self.assertEqual(messages, original)
        self.assertEqual(candidate[0], original[0])
        self.assertEqual(candidate[2], original[2])
        self.assertIn("call1", candidate[1]["content"])
        self.assertTrue(candidate[1]["content"].endswith("test FAIL\nTraceback: problem\n"))
        self.assertEqual(len(changes), 1)
        self.assertEqual(candidate[1]["tool_call_id"], "call2")

    def test_changed_or_unidentified_warning_is_not_compacted(self):
        warning = "/module.py:12: DeprecationWarning: " + "deprecated " * 20 + "\n"
        messages = [
            {"role": "tool", "tool_call_id": "a", "content": warning},
            {"role": "tool", "tool_call_id": "b", "content": warning.replace(":12:", ":13:")},
            {"role": "tool", "content": warning},
        ]
        self.assertEqual(compact_warning_blocks(messages), (messages, []))

    def test_harbor_json_output_wrapper_preserves_return_code_and_fields(self):
        warning = "/module.py:12: DeprecationWarning: " + "deprecated " * 20 + "\n"
        payload = {"returncode": 1, "output": warning + "FAIL\n", "extra": "retain"}
        messages = [
            {"role": "tool", "tool_call_id": "a", "content": json.dumps(payload)},
            {"role": "tool", "tool_call_id": "b", "content": json.dumps(payload)},
        ]
        candidate, changes = compact_warning_blocks(messages)
        self.assertEqual(candidate[0], messages[0])
        edited = json.loads(candidate[1]["content"])
        self.assertEqual(edited["returncode"], 1)
        self.assertEqual(edited["extra"], "retain")
        self.assertTrue(edited["output"].endswith("FAIL\n"))
        self.assertEqual(len(changes), 1)

    def test_canonical_preserves_reasoning_and_does_not_mutate_history(self):
        history = [
            {
                "role": "assistant",
                "content": "answer",
                "reasoning": "plan",
                "tool_calls": [{"function": {"name": "bash", "arguments": '{"command":"ls"}'}}],
            }
        ]
        saved = copy.deepcopy(history)
        converted = canonical(history)
        self.assertEqual(history, saved)
        self.assertEqual(converted[0]["reasoning"], "plan")
        self.assertEqual(converted[0]["content"], [{"type": "text", "text": "answer"}])
        self.assertEqual(converted[0]["tool_calls"][0]["function"]["arguments"], {"command": "ls"})

    def test_block_prefix_requires_matches_and_one_uncached_token(self):
        self.assertEqual(common_prefix_blocks([], [1] * 100), 0)
        self.assertEqual(common_prefix_blocks([1] * 64, [1] * 64), 32)
        self.assertEqual(common_prefix_blocks([1] * 64, [1] * 65), 64)
        self.assertEqual(common_prefix_blocks([1] * 40 + [2], [1] * 100), 32)


class PrefillPrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("gemma_precision_test", root / "tt/precision_policy.py")
        cls.policy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.policy)

    def test_schema_one_retains_fixed_prefill_contract(self):
        baseline = self.policy.baseline_precision_config()
        self.assertEqual(self.policy.resolve_precision_config(baseline), baseline)
        baseline["layer_types"]["full_attention"]["fixed"]["prefill_expert_down_dtype"] = "bfloat8_b"
        with self.assertRaisesRegex(ValueError, "Unsupported fixed"):
            self.policy.resolve_precision_config(baseline)

    def test_schema_two_changes_only_explicit_prefill_weight_groups(self):
        original = self.policy.baseline_precision_config()
        upgraded = self.policy.baseline_precision_config(2)
        for kind in upgraded["layer_types"]:
            layer = upgraded["layer_types"][kind]
            for key in self.policy.PREFILL_WEIGHT_FIELDS:
                self.assertEqual(layer.pop(key), original["layer_types"][kind]["fixed"][key])
                layer["fixed"][key] = original["layer_types"][kind]["fixed"][key]
        upgraded["schema_version"] = 1
        self.assertEqual(upgraded, original)
        candidate = self.policy.resolve_precision_config(
            {"schema_version": 2, "layer_overrides": {"5": {"prefill_expert_down_dtype": "bfloat8_b"}}}
        )
        layer = self.policy.layer_precision_config(candidate, 5, "full_attention")
        self.assertEqual(layer["prefill_expert_down_dtype"], "bfloat8_b")
        self.assertEqual(layer["fixed"]["prefill_expert_fidelity"], "LoFi")
        self.assertEqual(layer["kv_cache_dtype"], "bfloat8_b")

    def test_schema_two_rejects_unimplemented_changes(self):
        for override in (
            {"prefill_expert_down_dtype": "float32"},
            {"fixed": {"prefill_expert_fidelity": "HiFi4"}},
            {"prefill_expert_unknown": "bfloat8_b"},
        ):
            with self.assertRaisesRegex(ValueError, "Unknown|Unsupported"):
                self.policy.resolve_precision_config({"schema_version": 2, "layer_types": {"full_attention": override}})


class SuiteCounterTests(unittest.TestCase):
    def test_deadline_audit_distinguishes_late_proxy_work_from_saved_agent_actions(self):
        events = [
            {"event": "request_start", "request_id": "a", "unix_s": 1},
            {"event": "response", "request_id": "a", "unix_s": 7},
            {"event": "request_start", "request_id": "b", "unix_s": 8},
            {"event": "response", "request_id": "b", "unix_s": 10},
        ]
        report = deadline_audit(events, 5)
        self.assertEqual(report["proxy_response_seconds_after_agent_end"], [2, 5])
        self.assertEqual(report["requests_started_after_agent_end"], 1)
        self.assertEqual(report["completed_proxy_wait_after_agent_end_s"], 4)

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
        trailing = events + [{"event": "server_metrics", "phase": "before_request", "counters": final}]
        self.assertFalse(counter_delta(trailing, responses)["valid"])
        saved = completed_response_counters(trailing, responses)
        self.assertTrue(saved["valid"])
        self.assertEqual((saved["ttft_s"], saved["post_first_token_s"]), (2, 3))
        self.assertIn("not clipped agent time", saved["scope"])
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
    def test_prefill_control_is_explicit_and_preserves_source(self):
        root = Path(__file__).resolve().parents[1]
        source = root / "doc/datatype_sweep/selected_precision_config.json"
        original = source.read_bytes()
        with tempfile.TemporaryDirectory(prefix="gemma4-prefill-control-test-") as directory:
            output = Path(directory) / "policy.json"
            subprocess.run(
                [
                    sys.executable,
                    str(root / "tools/prepare_eval_weight_control.py"),
                    "--source",
                    str(source),
                    "--output",
                    str(output),
                    "--prefill-bfp8",
                ],
                check=True,
                capture_output=True,
            )
            candidate = json.loads(output.read_text())
            manifest = json.loads(output.with_suffix(".manifest.json").read_text())
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(candidate["schema_version"], 2)
        self.assertEqual(len(manifest["changes"]), 96)
        self.assertEqual(len(manifest["schema_migrations"]), 4)
        for layer in candidate["layer_types"].values():
            for key in ("prefill_expert_gate_dtype", "prefill_expert_down_dtype"):
                self.assertEqual(layer[key], "bfloat8_b")
                self.assertNotIn(key, layer["fixed"])
            self.assertEqual(layer["fixed"]["prefill_expert_fidelity"], "LoFi")

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
