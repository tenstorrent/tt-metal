# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Negative CPU controls for the deliberately narrow QSA dependency closure."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import verify_linear_qsa_dependency as dependency
from .summarize_functional_evidence import evaluate_gate
from .test_format19_dependency_cpu import restore_preformat_fixture
from .verify_linear_qsa_dependency import (
    ADDED_RUNTIME,
    BASELINE,
    BASELINE_SHA,
    CHANGES,
    ROOT,
    RUNTIME_BASELINE,
    RUNTIME_BASELINE_SHA,
    digest,
    verify_branch,
    verify_record,
    verify_transition,
)

CONFIG_BYTES = b'{"text_config":{"layer_types":["linear_attention","linear_attention"]}}'


class LinearQSADependencyTests(unittest.TestCase):
    def setUp(self):
        self.original = (ROOT / BASELINE).read_bytes()
        self.expected = self.original.decode()
        for old, new in CHANGES:
            self.expected = self.expected.replace(old, new)
        self.expected = self.expected.encode()

    def test_exact_local_transition(self):
        self.assertTrue(verify_transition(self.original, self.expected)["new_import_is_function_local"])

    def test_rejects_unchanged_or_extra_changes(self):
        mutations = [
            self.original,
            self.expected + b"\nSIDE_EFFECT = True\n",
            self.expected.replace(b"self.eps, self.context", b"self.changed, self.context"),
            self.expected.replace(b"from .ordered_projection import", b"from .other_projection import"),
        ]
        for changed in mutations:
            with self.subTest(changed=changed[-60:]), self.assertRaises(ValueError):
                verify_transition(self.original, changed)

    def test_rejects_different_baseline(self):
        with self.assertRaises(ValueError):
            verify_transition(self.original + b"\n", self.expected)

    def test_record_rejects_shared_change_wrong_layer_and_missing_binding(self):
        baseline = json.loads((ROOT / RUNTIME_BASELINE).read_text())
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            dependency, "CONFIG_SHA", digest(CONFIG_BYTES)
        ):
            root = Path(directory)
            (root / "config.json").write_bytes(CONFIG_BYTES)
            for name in set(baseline) | set(ADDED_RUNTIME):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
                restore_preformat_fixture(root)
            (root / "tt/qsa.py").write_bytes(self.expected)
            (root / "log.txt").write_text("closed\n")
            report = {"layer": 0, "passed": True}
            provenance = {
                "status": "finished",
                "exit_code": 0,
                "cwd": directory,
                "argv": [
                    "--output",
                    "result.json",
                    "--snapshot",
                    ".",
                ],
                "log": "log.txt",
                "log_sha256": digest((root / "log.txt").read_bytes()),
                "source_sha256": baseline,
            }

            def write():
                (root / "result.json").write_text(json.dumps(report))
                (root / "provenance.json").write_text(json.dumps(provenance))

            write()
            self.assertEqual(verify_record(root, "result.json", "provenance.json")["layer"], 0)
            report["layer"] = 3
            write()
            with self.assertRaises(ValueError):
                verify_record(root, "result.json", "provenance.json")
            report["layer"] = 0
            provenance["source_sha256"] = dict(baseline)
            del provenance["source_sha256"]["tt/precise_projection.py"]
            write()
            with self.assertRaises(ValueError):
                verify_record(root, "result.json", "provenance.json")
            provenance["source_sha256"] = baseline
            write()
            with (root / "tt/precise_projection.py").open("a") as stream:
                stream.write("\n# changed shared router implementation\n")
            with self.assertRaises(ValueError):
                verify_record(root, "result.json", "provenance.json")

    def test_evaluator_opt_in_and_fail_closed_controls(self):
        baseline = json.loads((ROOT / RUNTIME_BASELINE).read_text())
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            dependency, "CONFIG_SHA", digest(CONFIG_BYTES)
        ):
            root = Path(directory)
            (root / "config.json").write_bytes(CONFIG_BYTES)
            for name in set(baseline) | set(ADDED_RUNTIME):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
                restore_preformat_fixture(root)
            (root / "tt/qsa.py").write_bytes(self.expected)
            (root / "log.txt").write_text("closed\n")
            report = {"layer": 1, "passed": True, "pcc": 0.996}
            record = {
                "status": "finished",
                "exit_code": 0,
                "cwd": directory,
                "argv": [
                    "--output",
                    "result.json",
                    "--snapshot",
                    ".",
                ],
                "log": "log.txt",
                "log_sha256": digest((root / "log.txt").read_bytes()),
                "source_sha256": baseline,
            }
            gate = {
                "id": "linear_success",
                "artifact": "result.json",
                "provenance": "record.json",
                "scope": "Explicit linear layer test",
                "required_sources": list(baseline),
                "checks": [{"pointer": "/passed", "equals": True}, {"pointer": "/pcc", "ge": 0.995}],
            }
            (root / "result.json").write_text(json.dumps(report))
            (root / "record.json").write_text(json.dumps(record))
            self.assertEqual(evaluate_gate(root, gate)["status"], "unmet")
            closure = {
                "kind": "linear_qsa_constructor_v1",
                "historical_qsa_sha256": BASELINE_SHA,
                "current_qsa_sha256": digest(self.expected),
                "baseline_inventory_sha256": RUNTIME_BASELINE_SHA,
            }
            gate["source_closure"] = closure
            evaluated = evaluate_gate(root, gate)
            self.assertEqual(evaluated["status"], "declared_checks_satisfied_review_required")
            self.assertTrue(evaluated["source_closure"]["recorded_hashes_preserved"])
            for mutation in [
                "wrong_layer",
                "extra_qsa_edit",
                "shared_edit",
                "stale_binding",
                "unknown_kind",
                "missing_source",
                "failed_pcc",
                "extra_module",
            ]:
                with self.subTest(mutation=mutation):
                    altered_report = dict(report)
                    altered_record = {**record, "source_sha256": dict(baseline)}
                    altered_gate = {**gate, "source_closure": dict(closure)}
                    (root / "tt/qsa.py").write_bytes(self.expected)
                    shutil.copyfile(ROOT / "tt/precise_projection.py", root / "tt/precise_projection.py")
                    extra = root / "tt/__init__.py"
                    if extra.exists():
                        extra.unlink()
                    if mutation == "extra_module":
                        extra.write_text("SIDE_EFFECT = True\n")
                    elif mutation == "wrong_layer":
                        altered_report["layer"] = 3
                    elif mutation == "extra_qsa_edit":
                        changed = self.expected + b"\nSIDE_EFFECT = True\n"
                        (root / "tt/qsa.py").write_bytes(changed)
                        altered_gate["source_closure"]["current_qsa_sha256"] = digest(changed)
                    elif mutation == "shared_edit":
                        with (root / "tt/precise_projection.py").open("a") as stream:
                            stream.write("\n# changed shared router\n")
                    elif mutation == "stale_binding":
                        altered_gate["source_closure"]["current_qsa_sha256"] = "0" * 64
                    elif mutation == "unknown_kind":
                        altered_gate["source_closure"]["kind"] = "ignore_runtime_changes"
                    elif mutation == "missing_source":
                        del altered_record["source_sha256"]["tt/ple.py"]
                    else:
                        altered_report["pcc"] = 0.994
                    (root / "result.json").write_text(json.dumps(altered_report))
                    (root / "record.json").write_text(json.dumps(altered_record))
                    self.assertEqual(evaluate_gate(root, altered_gate)["status"], "unmet")

    def test_frozen_linear_branch_and_negative(self):
        source = (ROOT / "tt/functional_decoder.py").read_text()
        self.assertTrue(verify_branch(source)["linear_kind_never_constructs_qsa"])
        with self.assertRaises(ValueError):
            verify_branch(source.replace('kind == "linear_attention" else QSA', 'kind == "other" else QSA'))


if __name__ == "__main__":
    unittest.main()
