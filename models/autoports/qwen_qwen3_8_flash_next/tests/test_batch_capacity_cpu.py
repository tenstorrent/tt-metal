# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed CPU checks for the normal/watcher batch-capacity distinction."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from .test_format19_dependency_cpu import restore_preformat_fixture
from .validate_batch_capacity import CASES, DEPENDENCIES, validate_bound
from .validate_trace_evidence import digest


class BatchCapacityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        original = Path(__file__).resolve().parents[1]
        source = original / "doc/functional_decoder"
        self.base = self.root / "doc/functional_decoder"
        self.base.mkdir(parents=True)
        for path in (original / "tt").rglob("*"):
            if path.is_file() and path.suffix in (".py", ".cpp", ".hpp", ".h"):
                target = self.root / path.relative_to(original)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        for name in DEPENDENCIES:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original / name, target)
        restore_preformat_fixture(self.root)
        self.bound_path = self.base / "bound.json"
        self.bound = json.loads((source / "ple_private_batch_capacity_bound.json").read_text())
        for name in CASES:
            case = self.bound["cases"][name]
            shutil.copyfile(source / case["artifact"], self.base / case["artifact"])
            record = json.loads((source / case["provenance"]).read_text())
            log = Path(record["cwd"]) / record["log"]
            copied_log = self.base / (name + ".log")
            shutil.copyfile(log, copied_log)
            record["log"] = str(copied_log)
            record["argv"][record["argv"].index("--output") + 1] = str(self.base / case["artifact"])
            target = self.base / case["provenance"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(record))
        self.save_bound()

    def save_bound(self):
        self.bound_path.write_text(json.dumps(self.bound))

    def validate(self):
        return validate_bound(self.root, self.bound_path)

    def test_closed_evidence_keeps_control_and_scope(self):
        result = self.validate()
        self.assertEqual(result["verified_bound"]["exceeds_device_by_bytes"], 12072320)
        self.assertFalse(result["literal_hf_strict_passed"])
        self.assertEqual(result["stage_acceptance"], "not_evaluated")
        self.assertTrue(result["watcher5_is_separate_instrumentation_failure"])

    def test_changed_runtime_or_log_rejected(self):
        path = self.root / "tt/decoder_components.py"
        original = path.read_bytes()
        path.write_bytes(original + b"\n# altered\n")
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.validate()
        path.write_bytes(original)
        log = self.base / "ple_private_batch6_normal_v1.log"
        log.write_text(log.read_text() + "changed\n")
        with self.assertRaisesRegex(ValueError, "log changed"):
            self.validate()

    def test_watcher_failure_cannot_be_normal_limit(self):
        path = self.base / self.bound["cases"]["ple_private_batch6_normal_v1"]["provenance"]
        record = json.loads(path.read_text())
        record["environment"]["TT_METAL_WATCHER"] = "10"
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "Normal/watcher"):
            self.validate()

    def test_source_hash_omission_cannot_hide_runtime_change(self):
        path = self.base / self.bound["cases"]["ple_private_batch6_normal_v1"]["provenance"]
        record = json.loads(path.read_text())
        record["source_sha256"].pop("tt/ple.py")
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.validate()

    def test_trace_and_workspace_arithmetic_rejected(self):
        self.bound["batch6_bound"]["activated_plus_down_bytes"] += 80 * 1024 * 1024
        self.save_bound()
        with self.assertRaisesRegex(ValueError, "Bound arithmetic"):
            self.validate()
        self.bound["batch6_bound"]["activated_plus_down_bytes"] -= 80 * 1024 * 1024
        self.save_bound()
        case = self.bound["cases"]["ple_private_batch6_normal_v1"]
        path = self.base / case["artifact"]
        document = json.loads(path.read_text())
        document["memory_snapshots"]["exception_before_release_trace"]["allocated_bytes"] += 8
        path.write_text(json.dumps(document))
        case["sha256"] = digest(path)
        self.save_bound()
        with self.assertRaisesRegex(ValueError, "Trace allocation delta"):
            self.validate()

    def test_wrong_failed_process_outcome_rejected(self):
        path = self.base / self.bound["cases"]["ple_private_batch6_normal_v1"]["provenance"]
        record = json.loads(path.read_text())
        record["exit_code"] = 0
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "process outcome"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
