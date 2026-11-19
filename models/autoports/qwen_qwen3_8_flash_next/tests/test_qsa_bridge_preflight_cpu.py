# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only rejection checks for the ordered QSA bridge preflight."""

import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from .qsa_state_fixture import load_prefix
from .verify_qsa_bridge_inputs import verify_closed_producer, verify_hf_bindings


class BridgePreflightChecks(unittest.TestCase):
    def setUp(self):
        self.provenance = {"status": "finished", "exit_code": 0}
        self.report = {
            "passed": True,
            "prefill": {"passed": True},
            "full_boundary_branch": {
                "full_prefill_length": 262144,
                "full_prefill": {"passed": True},
                "tail_prefill_rows": 31,
                "tail_prefill": {"passed": True},
            },
            "decode": [{"passed": True}],
            "decode_repeat_equal": [True, True],
            "position_after_prefill": 262143,
            "runtime_audit": {"full_prefill": True, "decode_capture_and_replays": True},
        }

    def test_closed_success_gate(self):
        verify_closed_producer(self.provenance, self.report)

    def test_running_or_failed_producer_rejected(self):
        for provenance in [{"status": "running", "exit_code": 0}, {"status": "finished", "exit_code": 1}]:
            with self.assertRaises(ValueError):
                verify_closed_producer(provenance, self.report)

    def test_export_before_final_failure_rejected(self):
        for target in ["passed", "decode", "tail", "audit", "repeat"]:
            report = copy.deepcopy(self.report)
            if target == "passed":
                report["passed"] = False
            elif target == "decode":
                report["decode"][0]["passed"] = False
            elif target == "tail":
                report["full_boundary_branch"]["tail_prefill"]["passed"] = False
            elif target == "audit":
                report["runtime_audit"]["full_prefill"] = False
            else:
                report["decode_repeat_equal"][0] = False
            with self.assertRaises(ValueError):
                verify_closed_producer(self.provenance, report)

    def test_actual_attention_and_full_decoder_provenance_schemas(self):
        root = Path(__file__).resolve().parents[1] / "doc/functional_decoder"
        producer = json.loads((root / "qsa_ordered_device_full_boundary_v1.json").read_text())
        cpu = json.loads((root / "qsa_ordered_full_decoder_cpu_v1.json").read_text())
        recorded = json.loads((root / "provenance/qsa_ordered_device_full_boundary_v1.json").read_text())
        source = cpu["CPU_fixture"]["key"]["source_fixture_key"]
        self.assertNotIn("snapshot", producer["provenance"])
        verify_hf_bindings(cpu["provenance"], producer["provenance"], source, recorded["argv"])
        for field in ["snapshot", "config_sha256", "reference_sha256", "transformers"]:
            changed = copy.deepcopy(source)
            changed[field] = "different"
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_hf_bindings(cpu["provenance"], producer["provenance"], changed, recorded["argv"])
        for field in ["real_weight_keys", "checkpoint_shards"]:
            changed = copy.deepcopy(producer["provenance"])
            changed[field] = ["not-in-full-decoder"]
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_hf_bindings(cpu["provenance"], changed, source, recorded["argv"])

    def test_recovery_version_rejected(self):
        with patch("torch.load", return_value={"version": 2}):
            with self.assertRaises(AssertionError):
                load_prefix("unused", source_key={}, prefix=262143, capacity=262144)

    def test_stale_runtime_prefix_rejected(self):
        payload = {
            "version": 1,
            "origin": "TT QSA prefill from initially empty state; exported before final decode",
            "prefix": 262143,
            "capacity": 262144,
            "page_size": 32,
            "source_fixture_key": {},
            "runtime_source_sha256": {"stale.py": "invalid"},
        }
        with patch("torch.load", return_value=payload):
            with self.assertRaises(AssertionError):
                load_prefix("unused", source_key={}, prefix=262143, capacity=262144)

    def tearDown(self):
        self.assertNotIn("ttnn", sys.modules)


if __name__ == "__main__":
    unittest.main()
