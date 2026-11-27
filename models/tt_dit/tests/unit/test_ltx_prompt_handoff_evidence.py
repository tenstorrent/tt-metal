# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU negative checks of the actual C09 verifier, with no device imports."""

import ast
import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

SOURCE = Path(__file__).parents[1] / "encoders/gemma3/test_prompt_device_handoff.py"
NAMES = {"_bits", "_tensor_sha", "_sha", "_verify", "_cast_values"}
NAMESPACE = {"torch": torch, "hashlib": hashlib, "math": math, "json": json, "Path": Path, "__file__": str(SOURCE)}
TREE = ast.parse(SOURCE.read_text())
exec(
    compile(
        ast.Module(body=[n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in NAMES], type_ignores=[]),
        str(SOURCE),
        "exec",
    ),
    NAMESPACE,
)


class HandoffEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        values = {
            name: {
                axis: torch.full((1, 1, 1024, width), value, dtype=torch.bfloat16)
                for axis, width in (("video", 4096), ("audio", 2048))
            }
            for name, value in (("a", 1.0), ("b", -2.0))
        }
        rows = []
        labels = ["native_cast_boundary"] + [
            f"{p}_{i}_{n}" for p in ("eager", "traced") for i, n in enumerate(("a", "b", "a"))
        ]
        labels += [f"timed_{i}_a" for i in range(5)] + ["recreated_a"]
        for label in labels:
            tensors = (
                {"cast": NAMESPACE["_cast_values"]().bfloat16()}
                if label == "native_cast_boundary"
                else values["b" if label.endswith("_b") else "a"]
            )
            row = dict(label=label, modalities={})
            for axis, tensor in tensors.items():
                digest = NAMESPACE["_tensor_sha"](tensor)
                row["modalities"][axis] = dict(
                    baseline=tensor,
                    baseline_sha256=digest,
                    failed_replica_values={},
                    replicas=[
                        dict(coord=[i, j], sha256=digest, finite=True, exact=True) for i in range(4) for j in range(8)
                    ],
                )
            rows.append(row)
        cls.template = dict(
            metadata=dict(schema=1, capture_only=False, mesh=[4, 8], phase="component"),
            failure=None,
            guard_checked=True,
            rows=rows,
            samples={"baseline_host_roundtrip_ms": [1.0] * 5, "device_handoff_ms": [1.0] * 5},
            media=[],
        )

    def verify(self, data, succeeds):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "evidence.pt").write_bytes(b"mocked CPU fixture bytes")
            (path / "verified.json").write_text("stale PASS")
            with patch.object(torch, "load", return_value=data):
                if succeeds:
                    NAMESPACE["_verify"](path)
                else:
                    with self.assertRaises(AssertionError):
                        NAMESPACE["_verify"](path)
            self.assertEqual((path / "verified.json").exists(), succeeds)

    def test_valid_complete_evidence(self):
        self.verify(self.template, True)

    def test_last_mesh_replica_difference_rejected(self):
        data = copy.deepcopy(self.template)
        data["rows"][1]["modalities"]["video"]["replicas"][-1]["sha256"] = "0" * 64
        self.verify(data, False)

    def test_restored_reference_drift_rejected_even_if_candidate_matches_it(self):
        data = copy.deepcopy(self.template)
        data["rows"][-1]["modalities"] = copy.deepcopy(data["rows"][2]["modalities"])
        self.verify(data, False)

    def test_missing_guard_or_saved_failure_rejected(self):
        for change in (dict(guard_checked=False), dict(failure="RuntimeError: saved failure")):
            data = {**self.template, **change}
            self.verify(data, False)

    def test_missing_observation_rejected(self):
        data = {**self.template, "rows": self.template["rows"][:-1]}
        self.verify(data, False)

    def test_missing_timing_arm_rejected(self):
        for samples in ({}, {"device_handoff_ms": [1.0] * 5}):
            self.verify({**self.template, "samples": samples}, False)

    def test_wrong_sink_shape_rejected(self):
        data = copy.deepcopy(self.template)
        value = data["rows"][1]["modalities"]["video"]["baseline"]
        data["rows"][1]["modalities"]["video"]["baseline"] = value.reshape(1, 1024, 4096)
        self.verify(data, False)

    def test_native_probe_has_rounding_boundaries_and_signed_zero(self):
        actual = NAMESPACE["_cast_values"]().bfloat16().view(torch.int16).flatten()
        unsigned = actual.to(torch.int32) & 0xFFFF
        self.assertEqual(
            unsigned[:13].tolist(),
            [0, 0x8000, 0x3F80, 0xBF80, 0x3F80, 0x3F80, 0x3F81, 0x3F81, 0x3F82, 0x3F82, 0xBF80, 0xBF80, 0xBF81],
        )


if __name__ == "__main__":
    unittest.main()
