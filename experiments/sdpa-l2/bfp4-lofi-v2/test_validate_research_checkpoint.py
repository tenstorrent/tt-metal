# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Standard-library mutation tests; existing evidence is only read.

All corruptions are in-memory copies. No fixtures, source, or records are written.
"""
import ast
import copy
import importlib.util
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("checkpoint_readonly", HERE / "validate_research_checkpoint.py")
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)


def read(name):
    text = (HERE / name).read_text()
    return [V.decode(x) for x in text.splitlines() if x.strip()] if name.endswith(".jsonl") else V.decode(text)


class CheckpointTests(unittest.TestCase):
    def evaluate(self, family, record, **kwargs):
        e = V.Evidence("in-memory-only", family)
        audit = V.Audit(V.ROOT, HERE)
        getattr(V, family if not family.startswith("identity4") else "identity4")(audit, e, copy.deepcopy(record), **kwargs)
        return e.export()

    def native(self):
        return read("native-suite-256k-v2.jsonl")

    def primitive(self):
        return read("adaptive_bfp4_round/adaptive-pm-b4-normal-v1.json")

    def identity(self):
        return read("identity4_streaming/identity4-resident-perf-v1.json")

    def test_native_complete_and_failure_flags(self):
        original = self.native()
        good = self.evaluate("native_suite", original, length=262144)
        self.assertEqual(good["status"], "PASS")
        self.assertEqual(good["summary"]["cases"], 82)
        for field, value in (("finite", False), ("trace_equal", False), ("nonfinite_count", 1)):
            changed = copy.deepcopy(original)
            changed[1][field] = value
            self.assertEqual(self.evaluate("native_suite", changed, length=262144)["status"], "FAIL")

    def test_native_missing_duplicate_and_unfinished(self):
        original = self.native()
        missing = original[:1] + original[2:]
        self.assertEqual(self.evaluate("native_suite", missing, length=262144)["status"], "FAIL")
        duplicate = original[:-1] + [copy.deepcopy(original[1]), original[-1]]
        self.assertEqual(self.evaluate("native_suite", duplicate, length=262144)["status"], "FAIL")
        self.assertEqual(self.evaluate("native_suite", missing[:-1], length=262144)["status"], "PENDING")

    def test_during_run_stability_is_not_current_drift(self):
        r = self.native()
        r[-1]["sources_unchanged"] = False
        self.assertEqual(self.evaluate("native_suite", r, length=262144)["status"], "FAIL")
        e = V.Evidence("memory", "adaptive_primitive")
        r = self.primitive()
        r["source_sha256"][V.V2 + "adaptive_bfp4_round/compute.cpp"] = "0" * 64
        V.Audit(V.ROOT, HERE).provenance(e, r)
        self.assertEqual(e.export()["status"], "PASS")
        self.assertEqual(e.provenance["status"], "WARN")
        self.assertTrue(e.provenance["current_source_drift"])

    def test_manifest_omission_and_absence(self):
        e = V.Evidence("memory", "adaptive_primitive")
        r = self.primitive()
        name = V.V2 + "adaptive_bfp4_round/compute.cpp"
        del r["source_sha256"][name]
        V.Audit(V.ROOT, HERE).provenance(e, r)
        self.assertIn(name, e.provenance["manifest_omissions"])
        self.assertEqual(e.provenance["current_source_drift"], [])
        audit = V.Audit(V.ROOT, HERE)
        audit.load("this-intentionally-missing-checkpoint-record.json", "adaptive_primitive", V.adaptive_primitive)
        self.assertEqual(audit.items[0]["status"], "PENDING")

    def test_primitive_hash_and_oracle_gates(self):
        r = self.primitive()
        kwargs = dict(search="pm", fmt="b4", distribution="normal")
        self.assertEqual(self.evaluate("adaptive_primitive", r, **kwargs)["status"], "PASS")
        bad = copy.deepcopy(r)
        bad["actual_sha256"] = "0" * 64
        self.assertEqual(self.evaluate("adaptive_primitive", bad, **kwargs)["status"], "FAIL")
        for field in ("induced_exponent_mismatches", "native_grid_roundtrip_mismatches"):
            bad = copy.deepcopy(r)
            bad["oracle"][field] = 1
            self.assertEqual(self.evaluate("adaptive_primitive", bad, **kwargs)["status"], "FAIL")
        r["oracle"]["selection_differs_fp64_groups"] = 1
        self.assertEqual(self.evaluate("adaptive_primitive", r, **kwargs)["status"], "PASS")

    def test_identity_bits_and_performance_arithmetic(self):
        r = self.identity()
        self.assertEqual(self.evaluate("identity4_resident", r, resident=True)["status"], "PASS")
        for field in ("attention_tflops", "combined_tflops"):
            bad = copy.deepcopy(r)
            bad["cases"][1][field] *= 1.01
            self.assertEqual(self.evaluate("identity4_resident", bad, resident=True)["status"], "FAIL")
        bad = copy.deepcopy(r)
        bad["cases"][1]["attention"]["median_ms"] += 0.001
        self.assertEqual(self.evaluate("identity4_resident", bad, resident=True)["status"], "FAIL")
        bad = copy.deepcopy(r)
        bad["cases"][1]["output_sha256"] = "0" * 64
        self.assertEqual(self.evaluate("identity4_resident", bad, resident=True)["status"], "FAIL")

    def test_constant_v_undefined_and_epilogue(self):
        r = read("valuecenter-32768-matched_mean-constant_v-v1.json")
        kwargs = dict(length=32768, vfmt="b4", mode="matched_mean", distribution="constant_v")
        self.assertEqual(self.evaluate("value_centering", r, **kwargs)["status"], "PASS")
        bad = copy.deepcopy(r)
        bad["centered_output_accuracy"]["l2_pct"] = 0.0
        self.assertEqual(self.evaluate("value_centering", bad, **kwargs)["status"], "FAIL")
        r["epilogue_check"]["mismatch"] = 1
        self.assertEqual(self.evaluate("value_centering", r, **kwargs)["status"], "FAIL")

    def test_accurate_immutable_inputs(self):
        r = read("accurate-kcenter-1024-v1.jsonl")
        self.assertEqual(self.evaluate("accurate_kcenter", r, length=1024)["status"], "PASS")
        r[1]["immutable_inputs_verified"] = False
        self.assertEqual(self.evaluate("accurate_kcenter", r, length=1024)["status"], "FAIL")

    def test_json_rejects_duplicate_and_nonfinite(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
            with self.assertRaises(ValueError):
                V.decode(text)

    def test_validator_has_no_write_or_device_calls(self):
        tree = ast.parse((HERE / "validate_research_checkpoint.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertFalse({x.name.split(".")[0] for x in node.names} & {"torch", "ttnn", "subprocess"})
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, {"write_text", "write_bytes", "unlink", "rename", "mkdir", "open_device"})
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {"open", "exec", "eval"})

    def test_repeated_audit_preserves_record_hashes(self):
        first, second = V.run_audit(), V.run_audit()
        digest = lambda r: {x["file"]: x["summary"].get("record_sha256") for x in r["evidence"]}
        self.assertEqual(digest(first), digest(second))
        self.assertEqual(first["sources_changed_during_audit"], [])
        self.assertEqual(second["sources_changed_during_audit"], [])


if __name__ == "__main__":
    unittest.main()
