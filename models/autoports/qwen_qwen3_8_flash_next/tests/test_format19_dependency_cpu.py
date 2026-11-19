# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact format-only source closure and integration negative controls; CPU only."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import verify_format19_dependency as fmt
from .summarize_functional_evidence import evaluate_gate
from .validate_trace_evidence import source_closure


def restore_preformat_fixture(root):
    """Keep historical negative-control fixtures historical after repo formatting."""
    manifest = json.loads((fmt.ROOT / fmt.MANIFEST).read_text())
    for name, item in manifest["files"].items():
        target = root / name
        if target.exists():
            target.write_bytes((fmt.ROOT / item["before_snapshot"]).read_bytes())


class FormatDependencyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = json.loads((fmt.ROOT / fmt.MANIFEST).read_text())
        for path in (fmt.ROOT / "tt").rglob("*"):
            if path.is_file() and path.suffix in fmt.SUFFIXES:
                target = self.root / path.relative_to(fmt.ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        self.install("before")
        self.recorded = {
            str(p.relative_to(self.root)): fmt.digest(p.read_bytes())
            for p in (self.root / "tt").rglob("*")
            if p.is_file() and p.suffix in fmt.SUFFIXES
        }

    def install(self, which):
        for name, item in self.manifest["files"].items():
            (self.root / name).write_bytes((fmt.ROOT / item[which + "_snapshot"]).read_bytes())

    def test_exact_before_after_and_recorded_after(self):
        self.assertEqual(fmt.FormatSourceClosure(self.root, fmt.BINDING).proof["state"], "before")
        self.install("after")
        proof = fmt.FormatSourceClosure(self.root, fmt.BINDING)
        self.assertEqual(proof.proof["state"], "after")
        self.assertFalse(proof.proof["compiled_binary_identity_asserted"])
        for name in self.recorded:
            proof.verify(name, self.recorded[name])
            proof.verify(name, fmt.digest((self.root / name).read_bytes()))
        for name in self.manifest["files"]:
            with self.assertRaises(ValueError):
                fmt.FormatSourceClosure(self.root).verify(name, self.recorded[name])

    def test_unknown_or_stale_binding(self):
        for binding in [
            {},
            {**fmt.BINDING, "extra": True},
            {**fmt.BINDING, "manifest_sha256": "0" * 64},
            {"kind": "generic_whitespace"},
        ]:
            with self.subTest(binding=binding), self.assertRaises(ValueError):
                fmt.FormatSourceClosure(self.root, binding)

    def test_partial_transition_and_unknown_file(self):
        name, item = next(iter(self.manifest["files"].items()))
        (self.root / name).write_bytes((fmt.ROOT / item["after_snapshot"]).read_bytes())
        with self.assertRaisesRegex(ValueError, "Partially"):
            fmt.FormatSourceClosure(self.root, fmt.BINDING)
        self.install("before")
        (self.root / "tt/extra.py").write_text("x=1\n")
        with self.assertRaisesRegex(ValueError, "inventory"):
            fmt.FormatSourceClosure(self.root, fmt.BINDING)

    def test_changed_literal_operator_brace_or_directive(self):
        for name, needle, replacement in [
            ("tt/kernels/ordered_projection_reader.cpp", b"row < count", b"row <= count"),
            ("tt/kernels/router_topk_algorithm.hpp", b"0x7fffffffu", b"0x7ffffffeu"),
            ("tt/kernels/gdn_compute_sfpu.cpp", b"if (emit_state) {", b"if (emit_state)"),
            ("tt/kernels/gdn_compute.cpp", b"#include", b"# include"),
        ]:
            self.install("after")
            p = self.root / name
            raw = p.read_bytes()
            self.assertIn(needle, raw)
            p.write_bytes(raw.replace(needle, replacement, 1))
            with self.subTest(name=name), self.assertRaises(ValueError):
                fmt.FormatSourceClosure(self.root, fmt.BINDING)

    def test_wrong_binary_config_or_snapshot(self):
        paths = [
            Path(self.manifest["formatter"]["path"]),
            Path(self.manifest["configuration"]["path"]),
            fmt.ROOT / next(iter(self.manifest["files"].values()))["before_snapshot"],
        ]
        original = Path.read_bytes
        for bad in paths:

            def read(path):
                data = original(path)
                return data + b"changed" if path == bad else data

            with self.subTest(path=bad), mock.patch.object(Path, "read_bytes", read), self.assertRaises(ValueError):
                fmt.FormatSourceClosure(self.root, fmt.BINDING)
        with mock.patch.object(
            fmt.subprocess, "check_output", return_value="clang-format version 20.0"
        ), self.assertRaises(ValueError):
            fmt.FormatSourceClosure(self.root, fmt.BINDING)

    def test_shared_source_and_missing_recorded_hash_fail(self):
        self.install("after")
        proof = fmt.FormatSourceClosure(self.root, fmt.BINDING)
        with self.assertRaises(ValueError):
            proof.verify("tt/qsa.py", None)
        p = self.root / "tt/decoder_components.py"
        p.write_bytes(p.read_bytes() + b"\n# extra\n")
        with self.assertRaises(ValueError):
            proof.verify("tt/decoder_components.py", self.recorded["tt/decoder_components.py"])
        with self.assertRaises(ValueError):
            source_closure(self.root, {"source_sha256": self.recorded}, [], fmt.BINDING)

    def test_gate_requires_opt_in_preserves_numeric_checks(self):
        self.install("after")
        (self.root / "result.json").write_text(json.dumps({"passed": True, "pcc": 0.999}))
        (self.root / "log").write_text("closed\n")
        record = {
            "status": "finished",
            "exit_code": 0,
            "cwd": str(self.root),
            "argv": ["--output", "result.json"],
            "log": "log",
            "log_sha256": fmt.digest((self.root / "log").read_bytes()),
            "source_sha256": self.recorded,
        }
        (self.root / "provenance.json").write_text(json.dumps(record))
        gate = {
            "id": "format",
            "scope": "CPU mock strict gate",
            "artifact": "result.json",
            "provenance": "provenance.json",
            "required_sources": list(self.recorded),
            "checks": [{"pointer": "/passed", "equals": True}, {"pointer": "/pcc", "ge": 0.995}],
        }
        self.assertEqual(evaluate_gate(self.root, gate)["status"], "unmet")
        gate["format_source_closure"] = dict(fmt.BINDING)
        result = evaluate_gate(self.root, gate)
        self.assertEqual(result["status"], "declared_checks_satisfied_review_required")
        self.assertTrue(result["format_source_closure"]["recorded_hashes_preserved"])
        (self.root / "result.json").write_text(json.dumps({"passed": True, "pcc": 0.99}))
        self.assertEqual(evaluate_gate(self.root, gate)["status"], "unmet")
        (self.root / "result.json").write_text(json.dumps({"passed": True, "pcc": 0.999}))
        del record["source_sha256"]["tt/ple.py"]
        (self.root / "provenance.json").write_text(json.dumps(record))
        self.assertEqual(evaluate_gate(self.root, gate)["status"], "unmet")

    def test_composed_linear_qsa_and_format_proofs(self):
        from . import verify_linear_qsa_dependency as linear

        self.install("after")
        baseline = json.loads((linear.ROOT / linear.RUNTIME_BASELINE).read_text())
        config = b'{"text_config":{"layer_types":["linear_attention","linear_attention"]}}'
        (self.root / "config.json").write_bytes(config)
        (self.root / "result.json").write_text(json.dumps({"layer": 0, "passed": True}))
        (self.root / "log").write_text("closed\n")
        record = {
            "status": "finished",
            "exit_code": 0,
            "cwd": str(self.root),
            "argv": ["--output", "result.json", "--snapshot", "."],
            "log": "log",
            "log_sha256": fmt.digest((self.root / "log").read_bytes()),
            "source_sha256": baseline,
        }
        (self.root / "provenance.json").write_text(json.dumps(record))
        gate = {
            "id": "linear",
            "scope": "CPU composed proof",
            "artifact": "result.json",
            "provenance": "provenance.json",
            "required_sources": list(baseline),
            "checks": [{"pointer": "/passed", "equals": True}],
            "source_closure": {
                "kind": "linear_qsa_constructor_v1",
                "historical_qsa_sha256": linear.BASELINE_SHA,
                "current_qsa_sha256": fmt.digest((self.root / "tt/qsa.py").read_bytes()),
                "baseline_inventory_sha256": linear.RUNTIME_BASELINE_SHA,
            },
            "format_source_closure": dict(fmt.BINDING),
        }
        with mock.patch.object(linear, "CONFIG_SHA", fmt.digest(config)):
            result = evaluate_gate(self.root, gate)
            self.assertEqual(result["status"], "declared_checks_satisfied_review_required", result)
            self.assertEqual(result["source_closure"]["live_record"]["format_source_closure"]["state"], "after")
            (self.root / "result.json").write_text(json.dumps({"layer": 3, "passed": True}))
            self.assertEqual(evaluate_gate(self.root, gate)["status"], "unmet")

    def test_physical_failure_records_stay_failures_after_format(self):
        from .test_batch_capacity_cpu import BatchCapacityTests
        from .validate_batch_capacity import validate_bound

        case = BatchCapacityTests("test_closed_evidence_keeps_control_and_scope")
        case.setUp()
        try:
            for name, item in self.manifest["files"].items():
                (case.root / name).write_bytes((fmt.ROOT / item["after_snapshot"]).read_bytes())
            result = validate_bound(case.root, case.bound_path, fmt.BINDING)
            self.assertEqual(result["verified_bound"]["exceeds_device_by_bytes"], 12072320)
            self.assertFalse(result["literal_hf_strict_passed"])
            self.assertEqual(result["format_source_closure"]["state"], "after")
            self.assertEqual(result["stage_acceptance"], "not_evaluated")
        finally:
            case.doCleanups()

    def test_synthetic_summary_discloses_equivalence_not_identical_hashes(self):
        from .summarize_synthetic_evidence import summarize

        self.install("after")
        for name in [
            "tests/test_synthetic_decoder.py",
            "tests/synthetic_weights.py",
            "doc/functional_decoder/target_text_config.json",
            "doc/functional_decoder/weight_stats.json",
        ]:
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fmt.ROOT / name, p)
        xml = self.root / "synthetic.xml"
        shutil.copy2(fmt.ROOT / "doc/functional_decoder/synthetic_final_decoders_v4.xml", xml)
        record = json.loads(
            (fmt.ROOT / "doc/functional_decoder/provenance/synthetic_final_decoders_v4.json").read_text()
        )
        log = self.root / "run.log"
        shutil.copyfile(Path(record["cwd"]) / record["log"], log)
        record.update(
            cwd=str(self.root),
            argv=["python", "-m", "pytest", "tests/test_synthetic_decoder.py", f"--junitxml={xml}"],
            log=str(log),
        )
        record["source_sha256"] = dict(self.recorded)
        for name in ["tests/test_synthetic_decoder.py", "tests/synthetic_weights.py"]:
            record["source_sha256"][name] = fmt.digest((self.root / name).read_bytes())
        provenance = self.root / "pytest.json"
        provenance.write_text(json.dumps(record))
        rejected = summarize(self.root, xml, provenance, fmt.digest(xml.read_bytes()))
        self.assertFalse(rejected["passed"])
        accepted = summarize(self.root, xml, provenance, fmt.digest(xml.read_bytes()), fmt.BINDING)
        self.assertTrue(accepted["passed"], accepted["errors_found"])
        self.assertFalse(accepted["source_identity_matches_pytest_run"])
        self.assertTrue(accepted["source_identity_or_exact_format_closure"])
