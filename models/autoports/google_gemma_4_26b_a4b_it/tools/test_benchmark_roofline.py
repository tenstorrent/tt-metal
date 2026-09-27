# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic collector integration with the actual installed report validator."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import benchmark_roofline as collector
from benchmark_stage.roofline import load_roofline
from benchmark_stage.subsets import digest


class FakeWork:
    source_sha256 = {"synthetic": "not-device-evidence"}

    def prefill(self, lengths):
        return {"useful_flops": sum(lengths) * 100}

    def decode(self, positions, batch_slots):
        return {"dram_bytes": sum(positions) * batch_slots, "terms": {"synthetic": sum(positions) * batch_slots}}

    def peaks(self):
        return {
            "peak_flops_per_s": 1e12,
            "peak_dram_bytes_per_s": 1e11,
            "reference": "synthetic host test",
            "peak_basis": "not measured hardware",
        }


def fixture(root, n):
    count = max(8, n * 3)
    prefix = f"gemma4-perf-b{n}-"
    server = {"phase_control": f"synthetic-{n}", "model": "synthetic", "max_num_seqs": n}
    (root / f"perf-b{n}-server.json").write_text(json.dumps(server))
    (root / f"perf-b{n}-request-map.json").write_text(json.dumps({"request_id_prefix": prefix}))
    requests, events = {}, []
    timestamp = 2_000_000_000
    for index in range(count):
        rid = f"cmpl-{prefix}{index}-0-unique"
        requests[rid] = {
            "prompt_tokens": 4096,
            "max_tokens": 128,
            "temperature": 0,
            "prompt_sha256": f"distinct-{index}",
        }
        for step in range(128):
            event = {
                "submission_id": f"{index}:{step}",
                "request_ids": [rid],
                "phase": "prefill" if step == 0 else "decode",
                "positions": [] if step == 0 else [4095 + step],
                "prompt_lens": [4096] if step == 0 else [],
                "batch_slots": 1,
                "wire_rows": n,
                "device_sampling": True,
            }
            events.append({**event, "event": "dispatch", "timestamp_ns": timestamp})
            events.append({**event, "event": "completion", "timestamp_ns": timestamp + 10_000})
            timestamp += 20_000
    raw = {
        "completed": count,
        "total_input_tokens": count * 4096,
        "total_output_tokens": count * 128,
        "start_times": [1.0] * count,
        "duration": 20.0,
    }
    (root / f"perf-b{n}.json").write_text(json.dumps(raw))
    return {
        "identity": {"layer_count": 30, "max_num_seqs": n, "mesh": [1, 4]},
        "errors": [],
        "pending": [],
        "requests": requests,
        "events": events,
        "export_time_ns": timestamp,
    }


def collect(root, n, data):
    with (
        patch.object(
            sys, "argv", ["collector", "--run-dir", str(root), "--concurrency", str(n), "--action", "collect"]
        ),
        patch.object(collector, "query", return_value=data),
        patch.object(collector, "WorkAccounting", FakeWork),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        collector.main()


class CollectorTests(unittest.TestCase):
    def test_digest_matches_packaged_validator(self):
        value = {"nested": {"slots": 32, "list": [1, 4]}, "unicode": "μ"}
        self.assertEqual(collector.digest(value), digest(value))

    def test_both_profiles_validate_and_first_profile_is_retained(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for n in (32, 1):
                data = fixture(root, n)
                collect(root, n, data)
            evidence = load_roofline(root, required=("1", "32"))
            self.assertEqual(set(evidence), {"1", "32"})
            for n in (1, 32):
                self.assertGreater(evidence[str(n)]["prefill"]["percent"], 0)
                self.assertGreater(evidence[str(n)]["decode"]["percent"], 0)

    def test_readiness_transmits_accuracy_policy_and_checks_host_route(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = fixture(root, 32)
            server_path = root / "perf-b32-server.json"
            server = json.loads(server_path.read_text())
            server["base_url"] = "http://synthetic.invalid"
            server_path.write_text(json.dumps(server))
            bodies = []

            def respond(request, timeout):
                body = json.loads(request.data)
                bodies.append(body)
                rid = "cmpl-" + request.get_header("X-request-id") + "-0"
                is_accuracy = "messages" in body
                data["requests"][rid] = {"top_k": body.get("top_k", -1), "temperature": body["temperature"]}
                for phase in ("prefill", "decode"):
                    for event in ("dispatch", "completion"):
                        data["events"].append(
                            {"request_ids": [rid], "phase": phase, "event": event, "device_sampling": not is_accuracy}
                        )
                return io.StringIO(
                    json.dumps(
                        {
                            "usage": {"completion_tokens": 2},
                            "choices": [{"message": {"content": "Four"}, "finish_reason": "stop"}],
                        }
                    )
                )

            with (
                patch.object(
                    sys, "argv", ["collector", "--run-dir", str(root), "--concurrency", "32", "--action", "check"]
                ),
                patch.object(collector, "query", return_value=data),
                patch.object(collector, "WorkAccounting", FakeWork),
                patch.object(collector, "urlopen", side_effect=respond),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                collector.main()
            self.assertEqual(len(bodies), 2)
            accuracy = bodies[1]
            self.assertEqual(accuracy["top_k"], 64)
            self.assertEqual(accuracy["top_p"], 0.95)
            self.assertTrue(accuracy["logprobs"])
            self.assertEqual(accuracy["top_logprobs"], 0)
            self.assertFalse(accuracy["chat_template_kwargs"]["enable_thinking"])
            self.assertNotIn("ignore_eos", accuracy)
            evidence = json.loads((root / "phase-readiness-b32.json").read_text())
            self.assertIn("response", evidence["accuracy_host_route_probe"])

    def test_duplicate_prompts_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = fixture(root, 1)
            for state in data["requests"].values():
                state["prompt_sha256"] = "identical"
            with self.assertRaisesRegex(ValueError, "not distinct"):
                collect(root, 1, data)

    def test_incomplete_completion_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = fixture(root, 1)
            data["events"].pop()
            with self.assertRaisesRegex(ValueError, "missing dispatch or completion"):
                collect(root, 1, data)

    def test_host_sampling_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = fixture(root, 1)
            data["events"][0]["device_sampling"] = False
            with self.assertRaisesRegex(ValueError, "host sampling"):
                collect(root, 1, data)

    def test_extra_complete_decode_is_accounted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = fixture(root, 1)
            extra = [dict(e) for e in data["events"][-2:]]
            for event in extra:
                event["submission_id"] = "extra-step"
                event["timestamp_ns"] += 20_000
                event["positions"] = [4223]
            data["events"].extend(extra)
            collect(root, 1, data)
            accounting = json.loads((root / "phase-accounting-b1.json").read_text())
            self.assertEqual(max(accounting["output_steps_per_request"].values()), 129)


if __name__ == "__main__":
    unittest.main(verbosity=2)
