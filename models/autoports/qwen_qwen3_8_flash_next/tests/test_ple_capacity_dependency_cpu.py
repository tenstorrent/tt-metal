# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reject wider source exceptions and preserve deliberate capacity failures."""

import json
import shutil
from pathlib import Path

from .close_capacity_evidence import CAPACITY_SOURCES
from .test_batch_capacity_cpu import BatchCapacityTests
from .verify_ple_capacity_dependency import ple_source_closure


class PleDependencyTests(BatchCapacityTests):
    def setUp(self):
        super().setUp()
        self.original = Path(__file__).resolve().parents[1]
        for name in CAPACITY_SOURCES:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.original / name, target)

    def test_expected_failures_remain_failures(self):
        result = self.validate()
        for name in ("ple_private_batch6_normal_v1", "ple_private_batch5_controlled_v1"):
            proof = result["case_source_closure"][name]["qsa_unreachable_dependency_proof"]
            self.assertEqual(proof["recorded_exit_code"], 1)
            self.assertIsNot(proof["recorded_report_passed"], True)
        self.assertEqual(result["normal_passing_batch"], 5)

    def test_qsa_delta_must_be_exact(self):
        path = self.root / "tt/qsa.py"
        path.write_text(path.read_text() + "\n# additional edit\n")
        with self.assertRaisesRegex(ValueError, "two authorized constructor lines"):
            self.validate()

    def test_added_runtime_inventory_is_exact(self):
        (self.root / "tt/unreviewed.py").write_text("pass\n")
        with self.assertRaisesRegex(ValueError, "current runtime inventory"):
            self.validate()

    def test_added_candidate_identity_is_bound(self):
        path = self.root / "tt/ordered_projection.py"
        path.write_text(path.read_text() + "\n# modified candidate\n")
        with self.assertRaisesRegex(ValueError, "candidate source changed"):
            self.validate()

    def test_layer3_command_cannot_inherit_ple_exception(self):
        case = self.bound["cases"]["ple_private_batch6_normal_v1"]
        path = self.base / case["provenance"]
        record = json.loads(path.read_text())
        record["argv"][record["argv"].index("--layer") + 1] = "3"
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "layer1 batch command"):
            self.validate()

    def test_capacity_without_layer_requires_exact_harness(self):
        base = self.original / "doc/functional_decoder"
        document = json.loads((base / "ple_capacity_private_public_4768_v1.json").read_text())
        record = json.loads((base / "provenance/ple_capacity_private_public_4768_v1.json").read_text())
        self.assertNotIn("layer", document)
        self.assertNotIn("layer_idx", document)
        _, proof = ple_source_closure(self.root, record, CAPACITY_SOURCES, document, "capacity")
        self.assertEqual(proof["layer_proof"]["layer"], 1)
        document["layer"] = 3
        with self.assertRaisesRegex(ValueError, "Conflicting report layer"):
            ple_source_closure(self.root, record, CAPACITY_SOURCES, document, "capacity")
        del document["layer"]
        record["argv"][record["argv"].index("-m") + 1] = "models.other_probe"
        with self.assertRaisesRegex(ValueError, "Wrong executed PLE harness"):
            ple_source_closure(self.root, record, CAPACITY_SOURCES, document, "capacity")

    def test_required_helper_change_is_rejected(self):
        path = self.root / "tests/probe_ple_quantization.py"
        path.write_text(path.read_text() + "\n# changed\n")
        with self.assertRaisesRegex(ValueError, "Required source identity changed"):
            self.validate()

    def test_failure_cannot_be_relabeled_success(self):
        name = "ple_private_batch6_normal_v1"
        case = self.bound["cases"][name]
        path = self.base / case["artifact"]
        document = json.loads(path.read_text())
        document["passed"] = True
        path.write_text(json.dumps(document))
        from .validate_trace_evidence import digest

        case["sha256"] = digest(path)
        self.save_bound()
        with self.assertRaisesRegex(ValueError, "Failed capture called passed"):
            self.validate()
