import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

P = pathlib.Path(__file__).with_name("sfpu_corpus.py")
S = importlib.util.spec_from_file_location("sfpu_corpus", P)
M = importlib.util.module_from_spec(S)
S.loader.exec_module(M)


class CorpusTest(unittest.TestCase):
    def test_inventory_and_no_substring_mapping(self):
        rows = {r["id"]: r for r in M.inventory()}
        self.assertEqual(len(rows), 164)
        # Rows with no explicit per-row audit must remain unmapped (no basename
        # guessing).  legacy comp's kernels are not dispatched by any harness
        # (the zero-comp ops use the metal calculate_comp); zero_pad has no test
        # include anywhere.  legacy clamp gained an explicit coverage-parity
        # audit (SfpuType::clamp -> _calculate_clamp_) and is now mapped.
        self.assertEqual(
            rows["legacy__ckernel_sfpu_zero_pad"]["mapping_state"], "unmapped"
        )
        self.assertEqual(rows["legacy__ckernel_sfpu_comp"]["mapping_state"], "unmapped")
        self.assertEqual(
            rows["legacy__ckernel_sfpu_zero_pad"]["semantic_cpp_class"], "unmapped"
        )
        self.assertEqual(
            rows["legacy__ckernel_sfpu_comp"]["semantic_cpp_class"], "unmapped"
        )
        self.assertEqual(rows["legacy__ckernel_sfpu_clamp"]["mapping_state"], "mapped")
        self.assertIn(
            "mathop:Clamp", rows["legacy__ckernel_sfpu_clamp"]["functional_modules"]
        )

    def test_every_row_has_a_complete_explicit_semantic_audit(self):
        rows = M.read_manifest()
        errors, _ = M.validate(rows)
        self.assertEqual(errors, [])
        self.assertEqual(len(rows), 164)
        for row in rows:
            self.assertEqual(row["version"], "2")
            self.assertIn(row["semantic_cpp_class"], M.SEMANTIC_CPP_CLASSES)
            self.assertTrue(row["semantic_cpp_blocker"])
            self.assertIn(row["correctness_metric"], M.CORRECTNESS_METRICS)
            if row["mapping_state"] == "mapped":
                self.assertNotEqual(row["semantic_cpp_class"], "unmapped")

    def test_measured_silicon_is_correctness_gated(self):
        rows = {r["id"]: r for r in M.read_manifest()}
        expected = {
            "legacy__ckernel_sfpu_welfords": "win",
            "legacy__ckernel_sfpu_reduce_custom": "win",
            "legacy__ckernel_sfpu_binary_bcast": "parity",
            "legacy__ckernel_sfpu_where": "loss",
            "legacy__ckernel_sfpu_mul_int": "loss",
            "metal__ckernel_sfpu_mul_int32": "loss",
            "metal__ckernel_sfpu_exp": "loss",
            "metal__ckernel_sfpu_sigmoid_appx": "loss",
            "metal__ckernel_sfpu_recip": "win",
            "metal__ckernel_sfpu_lerp": "win",
            "legacy__ckernel_sfpu_topk": "blocked",
        }
        for row_id, status in expected.items():
            row = rows[row_id]
            self.assertEqual(row["silicon_status"], status)
            if status in {"win", "parity", "loss"}:
                self.assertEqual(row["test_status"], "pass")
                self.assertNotEqual(row["correctness_metric"], "none")

    def test_plan_formats_are_machine_and_human_readable(self):
        row = M.read_manifest()[0]
        import contextlib
        import io

        for fmt in ("tsv", "json", "markdown"):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                M.emit_plan([row], "bh", fmt)
            self.assertIn(row["id"], out.getvalue())
        payload = json.loads(self._capture_plan([row], "bh", "json"))
        self.assertEqual(payload["schema"], 2)
        self.assertIn("correctness_threshold", payload["rows"][0])

    @staticmethod
    def _capture_plan(rows, arch, fmt):
        import contextlib
        import io

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            M.emit_plan(rows, arch, fmt)
        return out.getvalue()

    def test_comparator(self):
        base = {
            "id": "x",
            "arch": "bh",
            "metric": "device_cycles",
            "scope": "body",
            "selector": "generated",
            "cycles": 100,
        }
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "b.json"
            p.write_text(json.dumps({"results": [base]}))

            def result(c=None):
                r = dict(base)
                r["cycles"] = c
                return M.compare_baseline([r], p, 2)[0]["status"]

            self.assertEqual(result(99), "PASS")
            self.assertEqual(result(103), "REGRESSION")
            self.assertEqual(result(None), "SKIP_NO_DEVICE_CYCLES")
            current = dict(base)
            current["cycles"] = 103
            c = pathlib.Path(d) / "c.json"
            c.write_text(json.dumps({"results": [current]}))
            self.assertNotEqual(
                subprocess.run(
                    [
                        sys.executable,
                        str(P),
                        "--compare-results",
                        str(c),
                        "--baseline",
                        str(p),
                        "--max-regression-pct",
                        "2",
                    ],
                    stdout=subprocess.DEVNULL,
                ).returncode,
                0,
            )

    def test_compiler_ab_artifact_classification(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            off = root / "off"
            on = root / "on"
            off.mkdir()
            on.mkdir()
            (off / "math.elf").write_bytes(b"same")
            (on / "math.elf").write_bytes(b"same")
            (off / "ignored.txt").write_text("off")
            (on / "ignored.txt").write_text("on")
            off_manifest = M.artifact_manifest(off)
            on_manifest = M.artifact_manifest(on)
            self.assertEqual(
                M.classify_artifact_pair(off_manifest, on_manifest)["status"],
                "BYTE_IDENTICAL",
            )
            (on / "math.elf").write_bytes(b"changed")
            pair = M.classify_artifact_pair(off_manifest, M.artifact_manifest(on))
            self.assertEqual(pair["status"], "CHANGED_BINARY")
            self.assertEqual(pair["changed_artifacts"], ["math.elf"])

    def test_compiler_ab_hashes_text_not_debug_paths(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            off = root / "off"
            on = root / "on"
            off.mkdir()
            on.mkdir()
            (off / "math.elf").write_bytes(b"TEXTSAMEdebug=/off")
            (on / "math.elf").write_bytes(b"TEXTSAMEdebug=/on")

            def fake_objcopy(cmd, **kwargs):
                source = pathlib.Path(cmd[-2])
                output = pathlib.Path(cmd[-1])
                output.write_bytes(source.read_bytes()[:8])
                return subprocess.CompletedProcess(cmd, 0, stdout="")

            off_manifest, off_errors = M.elf_text_manifest(
                off, root / "objcopy", root / "off-text", fake_objcopy
            )
            on_manifest, on_errors = M.elf_text_manifest(
                on, root / "objcopy", root / "on-text", fake_objcopy
            )
            self.assertEqual(off_errors, {})
            self.assertEqual(on_errors, {})
            self.assertEqual(
                M.classify_artifact_pair(off_manifest, on_manifest)["status"],
                "BYTE_IDENTICAL",
            )

    def test_compiler_ab_requires_compile_execution_and_both_options(self):
        cases = (
            [
                "--mode",
                "compile",
                "--compiler-ab-off-options=-mno-x",
                "--compiler-ab-on-options=-mx",
            ],
            [
                "--mode",
                "craq",
                "--execute",
                "--compiler-ab-off-options=-mno-x",
                "--compiler-ab-on-options=-mx",
            ],
            ["--mode", "compile", "--execute", "--compiler-ab-on-options=-mx"],
            ["--mode", "compile", "--execute", "--require-changed-binary"],
        )
        for args in cases:
            result = subprocess.run(
                [sys.executable, str(P), *args],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.assertNotEqual(result.returncode, 0)

    def test_checked_in_tsv_baseline(self):
        rows = M.load_baseline(M.DEVICE_BASELINE)
        ids = {r["id"] for r in M.inventory()}
        self.assertTrue(rows)
        self.assertTrue({r["id"] for r in rows} <= ids)
        current = {
            "id": "legacy__ckernel_sfpu_welfords",
            "arch": "bh",
            "metric": "device_cycles",
            "scope": "WELFORD_BODY",
            "selector": "generated",
            "cycles": 324,
        }
        result = M.compare_baseline([current], M.DEVICE_BASELINE, 0)[0]
        self.assertEqual(result["status"], "REGRESSION")
        self.assertAlmostEqual(result["delta_pct"], 100.0 / 323.0)

    def test_reciprocal_mapping_and_correctness_contract_are_explicit(self):
        row = {r["id"]: r for r in M.inventory()}["metal__ckernel_sfpu_recip"]
        self.assertEqual(row["mapping_state"], "mapped")
        self.assertEqual(row["paired_selector_status"], "implemented")
        self.assertEqual(row["correctness_metric"], "pcc")
        self.assertIn("rtol=0.05 atol=0.05", row["correctness_threshold"])
        self.assertEqual(row["silicon_status"], "win")
        self.assertIn("459 cycles vs production 467", row["silicon_result"])

    def test_lerp_mapping_and_compiler_ab_contract_are_explicit(self):
        row = {r["id"]: r for r in M.inventory()}["metal__ckernel_sfpu_lerp"]
        self.assertEqual(row["mapping_state"], "mapped")
        self.assertEqual(row["paired_selector_status"], "implemented")
        self.assertEqual(row["correctness_metric"], "pcc")
        self.assertIn("rtol=0.05 atol=0.05", row["correctness_threshold"])
        self.assertEqual(row["silicon_status"], "win")
        self.assertIn("compiler-pass ON 564.984375 cycles/tile", row["silicon_result"])

    def test_v1_is_retained_as_an_immutable_migration_source(self):
        old = P.with_name("sfpu_corpus_v1.tsv")
        with old.open() as f:
            rows = list(
                __import__("csv").DictReader(
                    (x for x in f if not x.startswith("#")), delimiter="\t"
                )
            )
        self.assertEqual(len(rows), 164)
        self.assertNotIn("semantic_cpp_class", rows[0])
        self.assertTrue(M.MANIFEST.name.endswith("_v2.tsv"))

    def test_empty_mapped_compile_lane_cannot_report_green(self):
        with tempfile.TemporaryDirectory() as d:
            run = pathlib.Path(d) / "qsr"
            result = subprocess.run(
                [
                    sys.executable,
                    str(P),
                    "--mode",
                    "compile",
                    "--arch",
                    "qsr",
                    "--execute",
                    "--require-executed-mapped",
                    "--run-root",
                    str(run),
                ],
                stdout=subprocess.DEVNULL,
            )
            self.assertNotEqual(result.returncode, 0)
            data = json.loads((run / "results.json").read_text())
            self.assertEqual(data["provenance"]["executed_mapped_gate"], "FAIL")

    def test_global_pytest_failure_is_attributed_to_exact_nodeid(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / "test_lanes.py").write_text(
                "def test_pass():\n    assert True\n\n"
                "def test_fail():\n    assert False\n"
            )
            env = os.environ.copy()
            nodes = {}
            for name in ("test_pass", "test_fail"):
                report = root / f"collect-{name}.json"
                rc, payload = M.invoke_pytest_report(
                    pathlib.Path(sys.executable),
                    root,
                    [f"test_lanes.py::{name}"],
                    [],
                    report,
                    root / f"collect-{name}.log",
                    env,
                    collect_only=True,
                )
                self.assertEqual(rc, 0)
                nodes[name] = set(payload["collected"])
            rc, payload = M.invoke_pytest_report(
                pathlib.Path(sys.executable),
                root,
                ["test_lanes.py"],
                [],
                root / "run.json",
                root / "run.log",
                env,
            )
            self.assertNotEqual(rc, 0)
            passed = {"status": "QUEUED"}
            failed = {"status": "QUEUED"}
            M.attribute_pytest_row(
                passed, nodes["test_pass"], payload["reports"], "test", root / "run.log"
            )
            M.attribute_pytest_row(
                failed, nodes["test_fail"], payload["reports"], "test", root / "run.log"
            )
            self.assertEqual(passed["status"], "PASS")
            self.assertEqual(failed["status"], "FAIL")
            self.assertEqual(failed["failing_nodeids"], ["test_lanes.py::test_fail"])

    def test_function_selector_collects_every_parameterized_nodeid(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / "test_params.py").write_text(
                "import pytest\n"
                "@pytest.mark.parametrize('value', [1, 2, 3])\n"
                "def test_param(value):\n    assert value\n"
            )
            rc, payload = M.invoke_pytest_report(
                pathlib.Path(sys.executable),
                root,
                ["test_params.py::test_param"],
                [],
                root / "collect.json",
                root / "collect.log",
                os.environ.copy(),
                collect_only=True,
            )
            self.assertEqual(rc, 0)
            self.assertEqual(len(payload["collected"]), 3)
            self.assertTrue(
                all(
                    x.startswith("test_params.py::test_param[")
                    for x in payload["collected"]
                )
            )

    def test_collection_failure_is_isolated_per_row(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / "test_one.py").write_text("def test_ok():\n    pass\n")
            bad_rc, bad = M.invoke_pytest_report(
                pathlib.Path(sys.executable),
                root,
                ["test_one.py::missing"],
                [],
                root / "bad.json",
                root / "bad.log",
                os.environ.copy(),
                collect_only=True,
            )
            good_rc, good = M.invoke_pytest_report(
                pathlib.Path(sys.executable),
                root,
                ["test_one.py::test_ok"],
                [],
                root / "good.json",
                root / "good.log",
                os.environ.copy(),
                collect_only=True,
            )
            self.assertNotEqual(bad_rc, 0)
            self.assertEqual(bad["collected"], [])
            self.assertEqual(good_rc, 0)
            self.assertEqual(good["collected"], ["test_one.py::test_ok"])

    def test_compiler_capability_blocks_only_declared_row(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            compiler = root / "compiler"
            compiler.write_text(
                "#!/bin/sh\n"
                'if [ "$1" = "--version" ]; then echo fake-sfpi; exit 0; fi\n'
                "cat >/dev/null\n"
                "exit 1\n"
            )
            compiler.chmod(0o755)
            expected = root / "sfpi-version"
            expected.write_text("sfpi_version='9.9.9'\n")
            installed = root / "sfpi.version"
            installed.write_text("9.9.9\n")
            preflight = M.compiler_preflight(
                compiler, "bh", {"indexed_topk"}, subprocess.run, installed, expected
            )
            self.assertTrue(preflight["pin_match"])
            self.assertFalse(preflight["capabilities"]["indexed_topk"]["available"])
            self.assertEqual(
                M.missing_row_capabilities("legacy__ckernel_sfpu_topk", preflight),
                ["indexed_topk"],
            )
            self.assertEqual(
                M.missing_row_capabilities(
                    "metal__ckernel_sfpu_sigmoid_appx", preflight
                ),
                [],
            )

    def test_compiler_pin_mismatch_is_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            compiler = root / "compiler"
            compiler.write_text("#!/bin/sh\necho fake-sfpi\nexit 0\n")
            compiler.chmod(0o755)
            expected = root / "sfpi-version"
            expected.write_text("sfpi_version='2.0'\n")
            installed = root / "sfpi.version"
            installed.write_text("1.0\n")
            preflight = M.compiler_preflight(
                compiler, "wh", set(), subprocess.run, installed, expected
            )
            self.assertEqual(preflight["status"], "PIN_MISMATCH")
            self.assertFalse(preflight["pin_match"])
            self.assertEqual(preflight["expected_sfpi_version"], "2.0")
            self.assertEqual(preflight["installed_sfpi_version"], "1.0")


if __name__ == "__main__":
    unittest.main()
