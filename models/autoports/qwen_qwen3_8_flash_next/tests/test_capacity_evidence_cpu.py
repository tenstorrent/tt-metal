# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only negative controls for capacity evidence granularity and scope."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from .close_capacity_evidence import PATCH_FIELDS, full_public_success, metric_check, prepared_capacity
from .prove_capacity_import_closure import DEPENDENCY, IMPORTER, prove_import_closure


class CapacityEvidenceTests(unittest.TestCase):
    def test_exact_unused_cli_export_proof_rejects_executed_source_change(self):
        model_root = Path(__file__).resolve().parents[1]
        recorded = model_root / "doc/functional_decoder/provenance/ple_capacity_residues_private_buffers_v1.json"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tests").mkdir()
            for name in (DEPENDENCY, IMPORTER):
                (root / name).write_bytes((model_root / name).read_bytes())
            provenance = root / "record.json"
            provenance.write_bytes(recorded.read_bytes())
            proof = prove_import_closure(root, provenance)
            self.assertTrue(proof["imported_execution_unchanged"])
            self.assertEqual(proof["recorded_dependency_sha256"], proof["historical_source_reconstructed_sha256"])
            original = (root / DEPENDENCY).read_text()
            (root / DEPENDENCY).write_text(original.replace("reference.eval()", "reference.train()"))
            with self.assertRaisesRegex(ValueError, "historical full-file"):
                prove_import_closure(root, provenance)
            (root / DEPENDENCY).write_text(original)
            (root / IMPORTER).write_text((root / IMPORTER).read_text() + "\n# changed importer\n")
            with self.assertRaisesRegex(ValueError, "importer changed"):
                prove_import_closure(root, provenance)
            (root / IMPORTER).write_bytes((model_root / IMPORTER).read_bytes())
            record = json.loads(provenance.read_text())
            record["source_sha256"][DEPENDENCY] = "0" * 64
            provenance.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "historical full-file"):
                prove_import_closure(root, provenance)

    def test_global_pcc_cannot_hide_failed_row_or_nonfinite(self):
        axis = {
            "rows": 65,
            "failed_row_count": 0,
            "nonfinite_row_count": 0,
            "global_pcc": 0.999,
            "minimum_row_pcc": 0.996,
        }
        value = {"passed": True, "finite": True, "whole": copy.deepcopy(axis), "contribution": copy.deepcopy(axis)}
        metric_check(value, 65)
        for key, failure in (
            ("minimum_row_pcc", 0.994),
            ("global_pcc", float("nan")),
            ("failed_row_count", 1),
            ("nonfinite_row_count", 1),
            ("rows", 64),
        ):
            changed = copy.deepcopy(value)
            changed["contribution"][key] = failure
            with self.assertRaises(ValueError):
                metric_check(changed, 65)

    def test_experimental_wrappers_cannot_close_default_runtime(self):
        value = {
            name: True
            for name in (
                "prepare_programs",
                "prepare_all_tails",
                "reserve_decode_input",
                "decode_lifecycle",
                "chunk_io_prepared_before_large_io",
            )
        }
        value.update(
            hash_lengths=list(range(1, 33)),
            implicit_probe32=False,
            search_is_candidate_only=True,
            capability_reduction_accepted=False,
        )
        value.update({name: {"enabled": False} for name in PATCH_FIELDS})
        prepared_capacity(value)
        for name in PATCH_FIELDS:
            changed = copy.deepcopy(value)
            changed[name]["enabled"] = True
            with self.assertRaisesRegex(ValueError, "Experimental"):
                prepared_capacity(changed)
        value["capability_reduction_accepted"] = True
        with self.assertRaisesRegex(ValueError, "promotes"):
            prepared_capacity(value)

    def test_offset_zero_proxy_cannot_be_called_complete_public(self):
        plan = [{"start": start, "count": min(32, 65 - start), "stage": "actual_public_chunk"} for start in (0, 32, 64)]
        value = {
            "length": 65,
            "passed": True,
            "phase": "complete",
            "preparation_state_restored": True,
            "decode_lifecycle_completed": True,
            "decode_output_finite": True,
            "prepared_signatures": [{"phase": "prefill", "tokens": n} for n in range(1, 33)]
            + [{"phase": "decode", "tokens": 1}],
            "chunk_plan": plan,
            "completed_chunk_checks": [
                {**row, "pass": phase} for phase in ("decoder_chunk", "repeated_decoder_chunk") for row in plan
            ],
            "full_public_passes": [{"output_finite": True}, {"output_finite": True}],
        }
        full_public_success(value)
        changed = copy.deepcopy(value)
        changed["chunk_plan"][1]["start"] = 0
        for row in changed["completed_chunk_checks"]:
            if row["start"] == 32:
                row["start"] = 0
        with self.assertRaisesRegex(ValueError, "complete public offset"):
            full_public_success(changed)
        changed = copy.deepcopy(value)
        changed["completed_chunk_checks"].pop()
        with self.assertRaisesRegex(ValueError, "Incomplete chunk"):
            full_public_success(changed)
        changed = copy.deepcopy(value)
        changed["decode_lifecycle_completed"] = False
        with self.assertRaisesRegex(ValueError, "lifecycle incomplete"):
            full_public_success(changed)


if __name__ == "__main__":
    unittest.main()
