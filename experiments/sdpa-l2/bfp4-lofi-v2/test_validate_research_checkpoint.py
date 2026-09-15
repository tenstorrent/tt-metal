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
from unittest import mock

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
        # Isolate omission handling from legitimate historical formatting drift.
        r["source_sha256"] = {p: V.hashlib.sha256((V.ROOT / p).read_bytes()).hexdigest()
                              for p in r["source_sha256"]}
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

    def test_adaptive_fullchip_exact_checks_layout_and_timing(self):
        original = read("adaptive32k-b8_b4-pm-normal-v1.json")
        kwargs = dict(length=32768, fmt="b8_b4", search="pm", distribution="normal")
        self.assertEqual(self.evaluate("adaptive_fullchip", original, **kwargs)["status"], "PASS")
        for mutate in (
            lambda r: r["preprocessing_checks"][2].update(actual_sha256="0" * 64),
            lambda r: r["preprocessing_checks"][2].update(decoded_bit_mismatches=1),
            lambda r: r["preprocessing_checks"][2]["selected_counts"].update({"1": -1}),
            lambda r: r.update(cpu_input_transform=True),
            lambda r: r.update(input_slots=1),
            lambda r: r["cb_audit"][1].update(tiles=64),
            lambda r: r["assignments"][0].update(jobs=1),
            lambda r: r["combined"].update(median_ms=r["combined"]["median_ms"] + 0.1),
            lambda r: r.update(attention_tflops=r["attention_tflops"] + 1),
        ):
            changed = copy.deepcopy(original)
            mutate(changed)
            self.assertEqual(self.evaluate("adaptive_fullchip", changed, **kwargs)["status"], "FAIL")
        # A declared relaxation is disclosed, not fabricated as an exact preprocessor check.
        relaxed = copy.deepcopy(original)
        relaxed.update(check_preprocess=False, preprocessing_checks=[])
        result = self.evaluate("adaptive_fullchip", relaxed, **kwargs)
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["summary"]["preprocessing_exact_checked"])

    def test_lut_pairs_bit_identity_controls_and_arithmetic(self):
        original = [read(f"lut-macro-{mode}-resident-v1.json") for mode in ("raw", "macro")]
        self.assertEqual(self.evaluate("lut_macro", original, scope="resident")["status"], "PASS")
        for field, value in (("output_sha256", "0" * 64), ("clock_mhz", 1000),
                             ("probability_pack_width", 1), ("tflops_per_core", 999)):
            changed = copy.deepcopy(original)
            changed[1][field] = value
            self.assertEqual(self.evaluate("lut_macro", changed, scope="resident")["status"], "FAIL")
        drifted = copy.deepcopy(original)
        for r in drifted:
            r["source_sha256"][V.V2 + "exp_lut.hpp"] = "0" * 64
        result = self.evaluate("lut_macro", drifted, scope="resident")
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["provenance"]["status"], "WARN")

    def test_lut_smoke_no_fabricated_timing_or_replay(self):
        original = [read(f"lut-macro-{mode}-smoke-v1.json") for mode in ("raw", "macro")]
        result = self.evaluate("lut_macro", original, scope="smoke")
        self.assertEqual(result["status"], "PASS")
        self.assertIsNone(result["summary"]["attention_speedup"])
        original[1]["trace_equal"] = True
        self.assertEqual(self.evaluate("lut_macro", original, scope="smoke")["status"], "FAIL")

    def test_missing_paired_record_is_pending(self):
        audit = V.Audit(V.ROOT, HERE)
        audit.load_pair(["lut-macro-raw-smoke-v1.json", "intentionally-absent-lut-macro.json"],
                        "lut_macro", V.lut_macro, scope="smoke")
        self.assertEqual(audit.items[0]["status"], "PENDING")
        self.assertEqual(len(audit.items[0]["summary"]["record_sha256"]), 1)

    def test_codec_cpu_coverage_and_scope(self):
        for suite, count in (("vaxis", 76), ("formats", 88)):
            original = read(f"codec-{suite}-v1.jsonl")
            result = self.evaluate("codec_cpu", original, suite=suite)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["summary"]["cases"], count)
            self.assertEqual(result["summary"]["execution"], "CPU_ONLY")
            self.assertFalse(result["summary"]["device_qualification"])
            self.assertEqual(self.evaluate("codec_cpu", original[:2] + original[3:], suite=suite)["status"], "FAIL")
            self.assertEqual(self.evaluate("codec_cpu", original[:-1], suite=suite)["status"], "PENDING")
            for mutate in (
                lambda r: r[0].update(execution="Blackhole device"),
                lambda r: r[0]["self_tests"].update(all_midpoint_ties_even=False),
                lambda r: r[2].update(attention_tflops=5),
                lambda r: r[3]["V"].update(group_axis="invalid"),
                lambda r: r[2]["fp64_output"].update(l2_pct=1),
            ):
                changed = copy.deepcopy(original)
                mutate(changed)
                self.assertEqual(self.evaluate("codec_cpu", changed, suite=suite)["status"], "FAIL")

    def test_required_long_adaptive_missing_is_pending(self):
        target = HERE / "adaptive256k-b8_b4-native-normal-v1.json"
        original_is_file = Path.is_file
        with mock.patch.object(Path, "is_file", lambda p: False if p == target else original_is_file(p)):
            audit = V.Audit(V.ROOT, HERE)
            audit.load(target.name, "adaptive_fullchip", V.adaptive_fullchip,
                       length=262144, fmt="b8_b4", search="native", distribution="normal")
        self.assertEqual(audit.items[0]["status"], "PENDING")
        self.assertFalse(audit.items[0]["optional"])

    def test_captured_interface_coverage_and_integrity(self):
        original = read("captured-interface-smoke-v1.jsonl")
        result = self.evaluate("captured_interface", original)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["summary"]["cases"], 6)
        self.assertFalse(result["summary"]["model_activation_qualification"])
        self.assertFalse(result["summary"]["performance_measured"])
        self.assertEqual(self.evaluate("captured_interface", original[:1] + original[2:])["status"], "FAIL")
        self.assertEqual(self.evaluate("captured_interface", original[:-1])["status"], "PENDING")
        for mutate in (
            lambda r: r[1].update(all_output_finite=False),
            lambda r: r[1].update(trace_equal=False),
            lambda r: r[1].update(immutable_inputs_verified=False),
            lambda r: r[1]["original_input_sha256"].update(q="0" * 64),
            lambda r: r[1].update(sampled_query_rows=list(range(1023))),
            lambda r: r[2].update(fast_private_correction_reset=False),
            lambda r: r[5].update(preprocessing_exact_checked=False),
            lambda r: r[1].update(attention_tflops=10),
            lambda r: r[-1].update(original_inputs_unchanged=False),
        ):
            changed = copy.deepcopy(original)
            mutate(changed)
            self.assertEqual(self.evaluate("captured_interface", changed)["status"], "FAIL")

    def test_captured_synthetic_semantics_and_metadata_hash(self):
        original = read("captured-interface-smoke-v1.jsonl")
        for mutate in (
            lambda r: r[0]["capture"]["metadata"]["provenance"].update(source_kind="model"),
            lambda r: r[0]["capture"]["metadata"].update(causal=True),
            lambda r: r[0]["capture"]["metadata"].update(mask="causal"),
            lambda r: r[0]["capture"].update(metadata_sha256="0" * 64),
            lambda r: r[0]["capture"].update(shape=[1, 2, 2048, 128]),
        ):
            changed = copy.deepcopy(original)
            mutate(changed)
            self.assertEqual(self.evaluate("captured_interface", changed)["status"], "FAIL")

    def test_captured_v1_gap_and_required_v2(self):
        historical = self.evaluate("captured_interface", read("captured-interface-smoke-v1.jsonl"))
        self.assertEqual(historical["status"], "PASS")
        self.assertEqual(historical["provenance"]["status"], "WARN")
        self.assertTrue(any("bf16-denom-pair-v3" in p for p in historical["provenance"]["manifest_omissions"]))
        target = HERE / "captured-interface-smoke-v2.jsonl"
        original_is_file = Path.is_file
        with mock.patch.object(Path, "is_file", lambda p: False if p == target else original_is_file(p)):
            audit = V.Audit(V.ROOT, HERE)
            audit.load(target.name, "captured_interface", V.captured_interface)
        self.assertEqual(audit.items[0]["status"], "PENDING")
        self.assertFalse(audit.items[0]["optional"])

    def test_captured_v2_current_and_bit_identical(self):
        current = read("captured-interface-smoke-v2.jsonl")
        result = self.evaluate("captured_interface", current)
        self.assertEqual(result["status"], "PASS")
        # V2 fixed the missing-pin issue. Subsequent formatting may legitimately
        # create current-source drift without invalidating historical evidence.
        self.assertEqual(result["provenance"]["manifest_omissions"], [])
        self.assertEqual(result["provenance"]["malformed_manifest"], [])
        historical = read("captured-interface-smoke-v1.jsonl")
        for old, new in zip(historical[1:-1], current[1:-1]):
            self.assertEqual(old["variant"], new["variant"])
            self.assertEqual(old["output_sha256"], new["output_sha256"])
            self.assertEqual(old["metrics"], new["metrics"])

    def test_v_transpose_complete_axes_and_coverage(self):
        count = 0
        for n, seed in ((32768, 1240), (262144, 1240), (32768, 1241)):
            pairs = []
            for axis in ("D", "N"):
                suffix = "-seed1241" if seed == 1241 else ""
                record = read(f"vt-full-{n}-{axis}{suffix}-v1.jsonl")
                result = self.evaluate("v_transpose", record, length=n, seed=seed, axis=axis)
                self.assertEqual(result["status"], "PASS", result["failures"])
                self.assertTrue(result["summary"]["input_immutability_producer_assertions"])
                count += result["summary"]["cases"]
                pairs.append(record)
            for d, nrow in zip(pairs[0][1:-1], pairs[1][1:-1]):
                self.assertEqual(d["distribution"], nrow["distribution"])
                self.assertEqual(d["original_input_sha256"], nrow["original_input_sha256"])
                self.assertEqual(d["sampled_query_rows"], nrow["sampled_query_rows"])
        self.assertEqual(count, 16)

    def test_v_transpose_integrity_mutations(self):
        original = read("vt-full-32768-N-v1.jsonl")
        kwargs = dict(length=32768, seed=1240, axis="N")
        for mutate in (
            lambda r: r[0]["args"].update(v_transposed=False),
            lambda r: r[1]["kernel"].update(v_group_axis="D: 16 consecutive channels within one token"),
            lambda r: r[1]["kernel"].update(pv_transpose_in1=False),
            lambda r: r[1]["kernel"].update(input_slots=1),
            lambda r: r[1]["kernel"]["preprocessing_checks"][0].update(mismatch=1),
            lambda r: r[1]["kernel"]["preprocessing_checks"].pop(),
            lambda r: r[1].update(check_preprocess=False),
            lambda r: r[1].update(all_output_finite=False),
            lambda r: r[1].update(trace_equal=False),
            lambda r: r[1].update(sources_unchanged=False),
            lambda r: r[1].update(immutable_inputs_verified=False),
            lambda r: r[1].update(accuracy_scope="Quantized-input reference"),
            lambda r: r[1].update(combined_tflops=r[1]["combined_tflops"] * 1.01),
            lambda r: r[1]["preprocessing_stages"]["v_transpose"].update(median_ms=123),
            lambda r: r[-1].update(sources_unchanged=False),
        ):
            changed = copy.deepcopy(original)
            mutate(changed)
            self.assertEqual(self.evaluate("v_transpose", changed, **kwargs)["status"], "FAIL")
        self.assertEqual(self.evaluate("v_transpose", original[:1] + original[2:], **kwargs)["status"], "FAIL")
        self.assertEqual(self.evaluate("v_transpose", original[:-1], **kwargs)["status"], "PENDING")

    def test_v_transpose_provenance_separate_from_gates(self):
        original = read("vt-full-32768-N-v1.jsonl")
        kwargs = dict(length=32768, seed=1240, axis="N")
        missing = copy.deepcopy(original)
        name = V.V2 + "vtransposed/pv_transpose.hpp"
        del missing[0]["source_sha256"][name]
        result = self.evaluate("v_transpose", missing, **kwargs)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["provenance"]["status"], "WARN")
        self.assertIn(name, result["provenance"]["manifest_omissions"])
        drift = copy.deepcopy(original)
        drift[0]["source_sha256"][V.V2 + "Vtransposed_fullchip.py"] = "0" * 64
        result = self.evaluate("v_transpose", drift, **kwargs)
        self.assertEqual(result["status"], "PENDING")
        self.assertFalse(result["summary"]["input_immutability_producer_assertions"])

    def test_historical_witness_snapshots_minimal_and_exact(self):
        needed = set()
        original = V.Audit.assertion_witness

        def collect(audit, e, r, driver, expression, node_type=ast.Assert):
            needed.add(r["source_sha256"][V.V2 + driver])
            return original(audit, e, r, driver, expression, node_type)

        with mock.patch.object(V.Audit, "assertion_witness", collect):
            result = V.run_audit()
        self.assertEqual(result["evidence_status"], "PASS")
        snapshots = list((HERE / "witness_sources").glob("*.source"))
        self.assertEqual({p.stem for p in snapshots}, needed)
        self.assertEqual(len(snapshots), 13)
        for path in snapshots:
            self.assertEqual(V.hashlib.sha256(path.read_bytes()).hexdigest(), path.stem)

    def test_formatting_drift_uses_historical_witness_without_hiding_drift(self):
        drivers = {"identity4_streaming.py", "value_centered_fullchip.py", "value_centered_b8_fullchip.py",
                   "adaptive_fullchip.py", "exp_lut_macro_streaming.py", "exp_lut_macro_resident.py",
                   "Vtransposed_fullchip.py", "fullchip.py", "hifi2_native_fullchip.py", "hifi2_lut_fullchip.py",
                   "hifi2_bf16_lut_fullchip.py", "combined_recipe_fullchip.py", "paired_vaxis_timing.py"}
        paths = {HERE / name for name in drivers}
        read_bytes = Path.read_bytes

        def formatted(path):
            data = read_bytes(path)
            return data + b"\n# Simulated formatting drift, in memory only.\n" if path in paths else data

        with mock.patch.object(Path, "read_bytes", formatted):
            result = V.run_audit()
            record = read("vt-full-32768-N-v1.jsonl")
            witness = self.evaluate("v_transpose", record, length=32768, seed=1240, axis="N")
        self.assertEqual(result["evidence_status"], "PASS")
        self.assertEqual(result["provenance_status"], "WARN")
        self.assertEqual(result["sources_changed_during_audit"], [])
        self.assertEqual(witness["status"], "PASS")
        self.assertTrue(any(x["source"].endswith("Vtransposed_fullchip.py")
                            for x in witness["provenance"]["current_source_drift"]))
        self.assertTrue(all(x["basis"] == "historical_sha256_snapshot" for x in witness["summary"]["producer_witnesses"]))

    def test_corrupt_or_wrong_hash_snapshot_rejected_before_ast(self):
        driver = "Vtransposed_fullchip.py"
        record = read("vt-full-32768-N-v1.jsonl")[0]
        producer = HERE / driver
        digest = record["source_sha256"][V.V2 + driver]
        snapshot = HERE / "witness_sources" / (digest + ".source")
        original_read = Path.read_bytes
        for payload in (b"not Python and not the expected bytes", original_read(HERE / "adaptive_fullchip.py")):
            def altered(path):
                if path == producer:
                    return original_read(path) + b"\n# Formatting simulation\n"
                return payload if path == snapshot else original_read(path)
            with mock.patch.object(Path, "read_bytes", altered), mock.patch.object(V.ast, "parse") as parse:
                e = V.Evidence("memory", "v_transpose")
                actual = V.Audit(V.ROOT, HERE).assertion_witness(
                    e, record, driver, "[tensor_hash(x) for x in inputs] == original_hashes")
                self.assertFalse(actual)
                self.assertEqual(e.export()["status"], "FAIL")
                self.assertTrue(any("SHA mismatch" in f for f in e.failures))
                parse.assert_not_called()

    def test_missing_current_source_uses_snapshot_and_missing_both_is_pending(self):
        driver = "Vtransposed_fullchip.py"
        record = read("vt-full-32768-N-v1.jsonl")[0]
        producer = HERE / driver
        snapshot = HERE / "witness_sources" / (record["source_sha256"][V.V2 + driver] + ".source")
        original_is_file = Path.is_file
        for missing, expected in (({producer}, "PASS"), ({producer, snapshot}, "PENDING")):
            with mock.patch.object(Path, "is_file", lambda p: False if p in missing else original_is_file(p)):
                e = V.Evidence("memory", "v_transpose")
                V.Audit(V.ROOT, HERE).assertion_witness(
                    e, record, driver, "[tensor_hash(x) for x in inputs] == original_hashes")
                self.assertEqual(e.export()["status"], expected)

    def test_late_mean_error_gates_and_undefined_uniform(self):
        r = read("mean-error-32768-b8_b8-uniform-mean_error-v1.json")
        kwargs = dict(length=32768, fmt="b8_b8", mode="mean_error", distribution="uniform")
        self.assertEqual(self.evaluate("mean_error", r, **kwargs)["status"], "PASS")
        for mutate in (
            lambda r: r.update(all_output_finite=False),
            lambda r: r.update(cpu_inputs_unchanged=False),
            lambda r: r.update(trace_equal=False),
            lambda r: r["preprocessing_checks"][0].update(mismatch=1),
            lambda r: r["value_mean_error_correction"]["check"].update(bias_mismatch=1),
            lambda r: r["correctness_trace_qualification"]["replay_output_sha256"].__setitem__(0, "0" * 64),
            lambda r: r["centered_output_accuracy"].update(l2_pct=0),
            lambda r: r.update(combined_tflops=999),
        ):
            changed = copy.deepcopy(r)
            mutate(changed)
            self.assertEqual(self.evaluate("mean_error", changed, **kwargs)["status"], "FAIL")

    def test_late_hifi2_routes_and_exactness_scope(self):
        r = read("hi2-native-rne-32768-v1.json")
        kwargs = dict(length=32768, route="native", option="rne")
        result = self.evaluate("hifi2_late", r, **kwargs)
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["summary"]["preprocessing_exact_checked"])
        r = read("hi2-lut-on-32768-v1.json")
        kwargs = dict(length=32768, route="lut", option="on")
        result = self.evaluate("hifi2_late", r, **kwargs)
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["summary"]["preprocessing_exact_checked"])
        for mutate in (
            lambda r: r.update(check_preprocess=False),
            lambda r: r["defines"].update(SDPA_LOFI_LUT_EXP=None),
            lambda r: r.update(b8_rne=False),
            lambda r: r.update(finite=False),
            lambda r: r.update(sources_unchanged=False),
            lambda r: r["attention"].update(median_ms=999),
        ):
            changed = copy.deepcopy(r)
            mutate(changed)
            self.assertEqual(self.evaluate("hifi2_late", changed, **kwargs)["status"], "FAIL")
        bf16 = read("hi2-bf16-lut-on-1024-v1.json")
        self.assertEqual(self.evaluate("hifi2_late", bf16, length=1024, route="bf16", option="on")["status"], "PASS")
        bf16["preprocessing_checks"][1]["identity_bits_preserved"] = False
        self.assertEqual(self.evaluate("hifi2_late", bf16, length=1024, route="bf16", option="on")["status"], "FAIL")

    def test_late_combined_recipe_and_v8_scope(self):
        r = read("recipe-32768-h16-vN-pm-v1.jsonl")
        kwargs = dict(length=32768, seed=1240, axis="N", recipe=(True, "pm"))
        self.assertEqual(self.evaluate("combined_recipe", r, **kwargs)["status"], "PASS")
        for mutate in (
            lambda r: r[1]["kernel"].update(score_scale=1),
            lambda r: r[1]["kernel"]["qk_rotation_metadata"][1].update(matrix_sha256="0" * 64),
            lambda r: r[1]["kernel"]["preprocessing_checks"][-1]["adaptive_statistics"].update(induced_exponent_mismatches=1),
            lambda r: r[1]["kernel"]["preprocessing_checks"][1].update(oracle_source="Ideal FP64 input"),
            lambda r: r[1]["preprocessing_stages"].pop("q_rotation"),
        ):
            changed = copy.deepcopy(r)
            mutate(changed)
            self.assertEqual(self.evaluate("combined_recipe", changed, **kwargs)["status"], "FAIL")
        self.assertEqual(self.evaluate("combined_recipe", r[:-1], **kwargs)["status"], "PENDING")
        smoke = read("vt-b8-smoke-N-v1.jsonl")
        kwargs = dict(length=1024, seed=1240, axis="N", vfmt="b8")
        result = self.evaluate("v8_axis", smoke, **kwargs)
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["summary"]["all_query_accuracy"])
        smoke[1]["kernel"]["v_format"] = "b4"
        self.assertEqual(self.evaluate("v8_axis", smoke, **kwargs)["status"], "FAIL")

    def test_late_paired_timing_order_hashes_and_arithmetic(self):
        r = read("vt-paired-32k-v1.jsonl")
        self.assertEqual(self.evaluate("paired_vaxis", r, length=32768)["status"], "PASS")
        for mutate in (
            lambda r: next(x for x in r if x["kind"] == "qualified").update(exact_preprocessing=False),
            lambda r: next(x for x in r if x["kind"] == "qualified")["kernel"]["preprocessing_checks"][0].update(mismatch=1),
            lambda r: next(x for x in r if x["kind"] == "round_order")["candidates"].reverse(),
            lambda r: next(x for x in r if x["kind"] == "summary").update(original_inputs_immutable=False),
            lambda r: next(x for x in r if x["kind"] == "summary")["candidates"][0].update(median_combined_ms=999),
            lambda r: next(x for x in r if x["kind"] == "trace_cleanup")["errors"].append("failed"),
        ):
            changed = copy.deepcopy(r)
            mutate(changed)
            self.assertEqual(self.evaluate("paired_vaxis", changed, length=32768)["status"], "FAIL")
        self.assertEqual(self.evaluate("paired_vaxis", r[:-1], length=32768)["status"], "PENDING")

    def test_missing_late_final_record_remains_pending(self):
        target = HERE / "hi2-bf16-lut-on-262144-v1.json"
        original = Path.is_file
        with mock.patch.object(Path, "is_file", lambda p: False if p == target else original(p)):
            audit = V.Audit(V.ROOT, HERE)
            audit.load(target.name, "hifi2_late", V.hifi2_late, length=262144, route="bf16", option="on")
        self.assertEqual(audit.items[0]["status"], "PENDING")

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
