# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for qualification evidence and HTTP result acceptance."""

import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.demos.qwen38_27b_qb2.demo import prefix_http_probe as http
from models.demos.qwen38_27b_qb2.demo import prefix_qualification as gate


class QualificationEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.bundle = Path(temporary.name)
        self.source = self.bundle / "source" / gate.MODEL
        (self.source / "tt").mkdir(parents=True)
        (self.source / "tt/model.py").write_text("frozen_model = True\n")
        (self.source / "config").mkdir()
        (self.source / "config/precision.json").write_text("{}")
        (self.bundle / "source" / gate.POLICY).write_text('{"recurrent_dtype": "float32"}')
        self.plan = dict(device_ids=[0, 4, 12, 8])
        self.physical = dict(
            passed=True,
            cleanup_completed=True,
            batched_transfer=True,
            device_ids=[0, 4, 12, 8],
            all_rank_bytes_exact=True,
            neighbours_unchanged=True,
            corrupt_restore_unpublished=True,
        )
        hashes = {str(p.relative_to(self.source)): gate.sha(p) for p in (self.source / "tt").rglob("*.py")}
        hashes["config/precision.json"] = gate.sha(self.source / "config/precision.json")
        hashes["effective_precision_override"] = gate.sha(self.bundle / "source" / gate.POLICY)
        self.continuation = dict(
            passed=True,
            cleanup_completed=True,
            batched_transfer=True,
            layers=64,
            device_ids=[0, 4, 12, 8],
            precision={"recurrent_dtype": "float32"},
            source_sha256=hashes,
            cases=[{"name": name} for name in ("suffix_prefill", "traced_decode", "post_decode_checkpoint")],
        )

    def test_serial_other_mesh_or_unclean_physical_evidence_is_rejected(self):
        gate.validate_native_result(self.physical, None, self.plan, self.bundle)
        for key, bad in [
            ("batched_transfer", False),
            ("cleanup_completed", False),
            ("neighbours_unchanged", False),
            ("device_ids", [0, 1, 2, 3]),
        ]:
            with self.subTest(key=key):
                mutated = dict(self.physical, **{key: bad})
                with self.assertRaises(ValueError):
                    gate.validate_native_result(mutated, None, self.plan, self.bundle)

    def test_continuation_requires_complete_matching_source_and_precision(self):
        gate.validate_native_result(self.continuation, 64, self.plan, self.bundle)
        for change in ("missing_source", "source_changed", "precision_changed", "missing_case"):
            with self.subTest(change=change):
                mutated = copy.deepcopy(self.continuation)
                if change == "missing_source":
                    mutated["source_sha256"].pop("tt/model.py")
                elif change == "source_changed":
                    mutated["source_sha256"]["tt/model.py"] = "0" * 64
                elif change == "precision_changed":
                    mutated["precision"]["recurrent_dtype"] = "bfloat16"
                else:
                    mutated["cases"].pop()
                with self.assertRaises(ValueError):
                    gate.validate_native_result(mutated, 64, self.plan, self.bundle)

    def completed_queue(self):
        gate.write(self.bundle / "bundle.json", {"frozen": True})
        report = dict(
            passed=True, cleanup_completed=True, bundle_sha256=gate.sha(self.bundle / "bundle.json"), stages=[]
        )
        for name, layers in [("transfer", None), ("layers-4", 4), ("layers-64", 64)]:
            receipt = self.bundle / "native" / name / "result.json"
            receipt.parent.mkdir(parents=True)
            measured = dict(self.physical) if layers is None else dict(self.continuation, layers=layers)
            gate.write(receipt, measured)
            report["stages"].append(
                dict(path=str(receipt.relative_to(self.bundle)), layers=layers, sha256=gate.sha(receipt))
            )
        gate.write(self.bundle / "native/queue.json", report)
        return report

    def test_http_rejects_missing_full64_and_changed_receipts(self):
        report = self.completed_queue()
        gate.require_native(self.bundle, self.plan)
        incomplete = copy.deepcopy(report)
        incomplete["stages"].pop()
        gate.write(self.bundle / "native/queue.json", incomplete)
        with self.assertRaisesRegex(ValueError, "full64"):
            gate.require_native(self.bundle, self.plan)
        gate.write(self.bundle / "native/queue.json", report)
        (self.bundle / report["stages"][0]["path"]).write_text("{}")
        with self.assertRaisesRegex(ValueError, "receipt changed"):
            gate.require_native(self.bundle, self.plan)

    def test_receipts_cannot_substitute_historical_or_external_files(self):
        report = self.completed_queue()
        report["stages"][0]["path"] = "../../old-passed-receipt.json"
        gate.write(self.bundle / "native/queue.json", report)
        with self.assertRaisesRegex(ValueError, "fresh task-owned"):
            gate.require_native(self.bundle, self.plan)

    def test_manifest_rejects_changed_bytes_and_path_escape(self):
        file = self.source / "tt/model.py"
        files = {str(file.relative_to(self.bundle)): gate.sha(file)}
        gate.verify_files(self.bundle, files)
        file.write_text("modified\n")
        with self.assertRaises(ValueError):
            gate.verify_files(self.bundle, files)
        with self.assertRaisesRegex(ValueError, "escapes"):
            gate.verify_files(self.bundle, {"../foreign-file": "0" * 64})

    def test_inherited_credentials_are_not_recorded(self):
        result = gate.logged_environment(
            {
                "PATH": "/bin",
                "OPENROUTER_API_KEY": "secret",
                "AWS_SECRET_ACCESS_KEY": "secret",
                "QWEN_DECODE_BUCKETS": "1",
            }
        )
        self.assertEqual(result, {"PATH": "/bin", "QWEN_DECODE_BUCKETS": "1"})

    def test_serving_must_log_exact_precision(self):
        expected = {"recurrent_dtype": "float32"}
        receipt = gate.validate_serving_precision(
            "worker: Qwen3.8 vLLM precision: {'recurrent_dtype': 'float32'}", expected
        )
        self.assertEqual(receipt["precision_confirmations"], 1)
        for log in ("ready", "worker: Qwen3.8 vLLM precision: {'recurrent_dtype': 'bfloat16'}"):
            with self.assertRaisesRegex(ValueError, "precision policy"):
                gate.validate_serving_precision(log, expected)

    @unittest.skipUnless(sys.platform == "linux", "Process-group supervision targets the Linux device host")
    def test_stop_reaps_owned_process_group(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        try:
            self.assertTrue(gate.stop(child))
            self.assertFalse(gate.group_alive(child.pid))
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()


class HTTPAcceptanceTests(unittest.TestCase):
    def test_token_id_mismatch_is_a_correctness_failure(self):
        with self.assertRaisesRegex(ValueError, "outputs differ"):
            http.same({"token_ids": [1], "text": "cedar"}, {"token_ids": [2], "text": "cedar"})

    def test_truncated_or_tokenless_http_response_is_rejected(self):
        response = {
            "choices": [{"text": "cedar", "token_ids": [1], "finish_reason": "length"}],
            "usage": {"completion_tokens": 1},
        }
        with patch.object(http.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(response).encode())):
            with self.assertRaisesRegex(ValueError, "wrong output count"):
                http.completion("http://127.0.0.1:18086", [1, 2], "case", output_tokens=64)

    def test_absent_cache_metric_is_not_a_warm_pass(self):
        with patch.object(http, "get", return_value=b"# help\nvllm:other_counter 2\n"):
            with self.assertRaisesRegex(ValueError, "not exposed"):
                http.metric("http://127.0.0.1:18086", http.HITS)

    def test_warm_pass_requires_measured_hit_and_exact_output_count(self):
        result = {"token_ids": [1] * 64, "text": "cedar"}
        with patch.object(http, "metric", side_effect=[8, 4104]), patch.object(http, "completion", return_value=result):
            measured = http.hit_completion("http://127.0.0.1:18086", [1, 2], "case")
        self.assertEqual(measured["external_hit_tokens_delta"], 4096)


if __name__ == "__main__":
    unittest.main()
