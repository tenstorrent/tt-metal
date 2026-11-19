# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only negative controls: missing/pending/dirty evidence is never clean."""

import json
import tempfile
import unittest
from pathlib import Path

from .close_watcher_evidence import PROFILER_ENV, close_watcher, scan
from .validate_trace_evidence import digest


class WatcherEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("tt/model.py", "tests/probe.py", "tests/run_recorded.py"):
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("# source fixture\n")
        sources = {str(p.relative_to(self.root)): digest(p) for p in self.root.rglob("*.py")}
        self.manifest = {"runs": []}
        for layer in (0, 1, 3):
            artifact = self.root / f"run{layer}.json"
            artifact.write_text(json.dumps({"layer_idx": layer, "passed": True}))
            log = self.root / f"run{layer}.log"
            log.write_text("Device closed\nRUN_RECORDED_FINISHED exit_code=0\n")
            wr = self.root / f"watcher{layer}"
            w = wr / "generated/watcher/watcher.log"
            w.parent.mkdir(parents=True)
            w.write_text(
                "At 0.1s attach device 0\nDump #1 at 0.2s\nDevice 0 worker core(x=0,y=0): R\nDump #1 completed at 0.3s\nAt 0.4s detach device 0\n"
            )
            record = {
                "status": "finished",
                "exit_code": 0,
                "cwd": str(self.root),
                "argv": [
                    "python",
                    "-m",
                    "models.autoports.qwen_qwen3_8_flash_next.tests.probe",
                    "--output",
                    str(artifact),
                ],
                "log": str(log),
                "log_sha256": digest(log),
                "source_sha256": sources,
                "environment": {
                    **{k: None for k in PROFILER_ENV},
                    "TT_METAL_WATCHER": "10",
                    "TT_METAL_LOGS_PATH": str(wr),
                },
            }
            (self.root / f"provenance{layer}.json").write_text(json.dumps(record))
            self.manifest["runs"].append(
                {
                    "layer": layer,
                    "artifact": artifact.name,
                    "provenance": f"provenance{layer}.json",
                    "required_sources": ["tt/model.py"],
                }
            )

    def mutate_record(self, action):
        path = self.root / "provenance3.json"
        value = json.loads(path.read_text())
        action(value)
        path.write_text(json.dumps(value))

    def validate(self):
        return close_watcher(self.root, self.manifest)

    def test_complete_coverage(self):
        result = self.validate()
        self.assertTrue(result["all_clean"])
        self.assertEqual(result["layers"], [0, 1, 3])
        self.assertEqual(result["stage_acceptance"], "not_evaluated")

    def test_missing_or_duplicate_layer(self):
        self.manifest["runs"].pop()
        with self.assertRaisesRegex(ValueError, "exactly layers"):
            self.validate()

    def test_pending_or_failed_process(self):
        self.mutate_record(lambda r: r.update(status="running"))
        with self.assertRaisesRegex(ValueError, "Unfinished"):
            self.validate()
        self.mutate_record(lambda r: r.update(status="finished", exit_code=1))
        with self.assertRaisesRegex(ValueError, "unsuccessful"):
            self.validate()

    def test_failed_result_and_wrong_layer(self):
        p = self.root / "run3.json"
        p.write_text(json.dumps({"layer_idx": 3, "passed": False}))
        with self.assertRaisesRegex(ValueError, "did not pass"):
            self.validate()
        p.write_text(json.dumps({"layer_idx": 1, "passed": True}))
        with self.assertRaisesRegex(ValueError, "Wrong report layer"):
            self.validate()

    def test_no_watcher_or_profiler_collision(self):
        self.mutate_record(lambda r: r["environment"].update(TT_METAL_WATCHER="0"))
        with self.assertRaisesRegex(ValueError, "not enabled"):
            self.validate()
        self.mutate_record(lambda r: r["environment"].update(TT_METAL_WATCHER="10", TT_METAL_DEVICE_PROFILER="1"))
        with self.assertRaisesRegex(ValueError, "Profiler"):
            self.validate()

    def test_missing_empty_or_incomplete_watcher(self):
        p = self.root / "watcher3/generated/watcher/watcher.log"
        original = p.read_text()
        p.unlink()
        with self.assertRaises(FileNotFoundError):
            self.validate()
        p.write_text("")
        with self.assertRaisesRegex(ValueError, "Empty"):
            self.validate()
        p.write_text(original.replace("Dump #1 completed at 0.3s\n", ""))
        with self.assertRaisesRegex(ValueError, "incomplete Watcher dumps"):
            self.validate()
        p.write_text(original.replace("At 0.4s detach device 0\n", ""))
        with self.assertRaisesRegex(ValueError, "attach/detach"):
            self.validate()

    def test_actual_failure_signatures(self):
        for message in (
            "Watcher detected NOC error and stopped device:",
            "Watcher unexpected noc debug state (corrupted noc sanitization state)",
            "Watcher stopped the device due to tripped assert",
            "Watcher detected stack overflow on Device 0",
            "NOC transaction overflows a circular buffer",
            "TT_FATAL: failure",
            "Traceback (most recent call last):",
        ):
            with self.subTest(message=message), self.assertRaises(ValueError):
                scan(message, "fixture")

    def test_changed_source_or_extra_runtime(self):
        p = self.root / "tt/model.py"
        p.write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.validate()
        p.write_text("# source fixture\n")
        (self.root / "tt/new.py").write_text("pass\n")
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.validate()

    def test_log_identity_and_completion(self):
        p = self.root / "run3.log"
        p.write_text("incomplete\n")
        with self.assertRaisesRegex(ValueError, "Command log changed"):
            self.validate()
        self.mutate_record(lambda r: r.update(log_sha256=digest(p)))
        with self.assertRaisesRegex(ValueError, "final completion"):
            self.validate()

    def test_linear_exception_forbidden_on_qsa(self):
        self.manifest["runs"][-1]["source_closure"] = "linear_qsa_constructor_v1"
        with self.assertRaisesRegex(ValueError, "linear-only"):
            self.validate()
