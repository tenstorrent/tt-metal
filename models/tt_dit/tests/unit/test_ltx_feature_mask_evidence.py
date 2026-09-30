# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only negatives for literal BF16 and C13 evidence persistence/provenance."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

SOURCE = Path(__file__).resolve().parents[1] / "encoders/gemma3/test_feature_mask_projection.py"
spec = importlib.util.spec_from_file_location("feature_mask_evidence", SOURCE)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


class FeatureEvidenceTest(unittest.TestCase):
    def test_signed_zero_is_not_literal_equal(self):
        positive = torch.tensor([0.0], dtype=torch.bfloat16)
        negative = torch.tensor([-0.0], dtype=torch.bfloat16)
        self.assertTrue(torch.equal(positive, negative))
        self.assertNotEqual(harness._tensor_sha(positive), harness._tensor_sha(negative))
        self.assertFalse(torch.equal(harness._bits(positive), harness._bits(negative)))

    def test_undefined_zero_bias_diagnostics_are_explicit(self):
        for value in (0.0, 2.0):
            actual = torch.full((16,), value, dtype=torch.bfloat16)
            metrics = harness._diagnostic_metrics(actual, actual.float())
            self.assertIsNone(metrics["pcc_diagnostic"])
            self.assertFalse(metrics["pcc_defined"])
            self.assertEqual(metrics["max_abs"], 0.0)
            self.assertEqual(metrics["relative_l2_defined"], value != 0.0)
            self.assertEqual(metrics["relative_l2"], 0.0 if value else None)
            json.dumps(metrics, allow_nan=False)

    def test_masks_exercise_tile_boundary_and_changed_prompts(self):
        masks = {"a": torch.zeros((1, 1024), dtype=torch.long), "b": torch.ones((1, 1024), dtype=torch.long)}
        boundary = harness._mask("tile_boundary", masks)
        self.assertEqual(boundary[0, 30:35].tolist(), [0, 1, 0, 0, 1])
        self.assertEqual(boundary[0, 62:66].tolist(), [1, 0, 0, 1])
        self.assertEqual(harness._mask("all_pad", masks).sum(), 0)
        self.assertEqual(harness._mask("all_valid", masks).sum(), 1024)
        self.assertIs(harness._mask("a", masks), masks["a"])
        self.assertIs(harness._mask("b", masks), masks["b"])

    def test_unreviewed_tracked_edit_is_rejected(self):
        with patch.dict(harness.os.environ, {}, clear=True):
            with patch.object(harness.subprocess, "check_output", return_value=" M production.py\n"):
                with self.assertRaisesRegex(AssertionError, "tracked source edits"):
                    harness._tracked_source_binding()
            with patch.object(harness.subprocess, "check_output", return_value=""):
                self.assertEqual(harness._tracked_source_binding()["tracked_status"], "")

    def test_fixture_bytes_and_preparer_copy_are_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            names = ("states-a.pt", "states-b.pt", "tokens.pt", "weights.pt", "references.pt")
            for name in names:
                (directory / name).write_bytes(name.encode())
            (directory / "preparer-source.py").write_bytes(b"immutable preparer")
            manifest = {
                "schema": 1,
                "cases": list(harness.CASES),
                "files": {n: harness._sha(directory / n) for n in names},
                "preparer_sha256": harness._sha(directory / "preparer-source.py"),
            }
            (directory / "manifest.json").write_text(json.dumps(manifest))
            harness._fixture(directory, references=True)
            (directory / "states-b.pt").write_bytes(b"changed prompt states")
            with self.assertRaisesRegex(AssertionError, "fixture changed"):
                harness._fixture(directory)
            (directory / "states-b.pt").write_bytes(b"states-b.pt")
            (directory / "preparer-source.py").write_bytes(b"changed preparer")
            with self.assertRaises(AssertionError):
                harness._fixture(directory)

    def test_pair_validation_rejects_stale_or_reused_evidence(self):
        # Isolate outer pair/persistence checks; per-record arithmetic validation
        # is deliberately mocked here, not represented as tested hardware math.
        base = {
            "mode": "0",
            "cgroup": "job-1",
            "trace_device_ms": [1.0] * 5,
            "outputs": {"eager_a": {"video": torch.tensor([0.0], dtype=torch.bfloat16)}},
        }
        for key in (
            "manifest_sha256",
            "sources",
            "mesh",
            "physical_ids",
            "topology",
            "num_links",
            "commit",
            "source_sha256",
            "tracked_source_binding",
            "binary_sha256",
        ):
            base[key] = "SYNTHETIC"
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "manifest.json").write_text("{}")
            base_path, result = directory / "base.pt", directory / "candidate.pt"
            torch.save(base, base_path)
            candidate = {**base, "mode": "1", "cgroup": "job-2"}
            with patch.object(harness, "_validate", return_value={}):
                torch.save(candidate, result)
                harness._verify(result, directory, base_path)
                self.assertTrue(result.with_suffix(".equivalence.json").exists())
                for changed in (
                    {"cgroup": "job-1"},
                    {"commit": "STALE"},
                    {"outputs": {"eager_a": {"video": torch.tensor([-0.0], dtype=torch.bfloat16)}}},
                ):
                    result.with_suffix(".equivalence.json").write_text("previous success")
                    torch.save({**candidate, **changed}, result)
                    with self.assertRaises(AssertionError):
                        harness._verify(result, directory, base_path)
                    self.assertFalse(result.with_suffix(".equivalence.json").exists())


if __name__ == "__main__":
    unittest.main()
