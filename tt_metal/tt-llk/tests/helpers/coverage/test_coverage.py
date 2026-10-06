# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks: PYTHONPATH=tests/python_tests LLK_AST_ANALYZER=... python -m unittest discover -s tests/helpers/coverage."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from helpers.coverage.ast_scan import llk_ast
from helpers.coverage.baseline import source_baseline
from helpers.coverage.gcov_json import function_records
from helpers.coverage.report import write_report
from helpers.coverage.store import Store
from helpers.coverage.sweep import audit_group

SOURCE = """namespace test {
enum class Mode { A = 1 << 2, Alias = A, B = A + 3 };
using Alias = Mode;
template<bool Enabled, Alias M> int unused(Mode mode, int count) {
    if constexpr (Enabled) {
        if (mode == Mode::A && count > 0) return count;
    } else {
        return M == Mode::B ? count : 0;
    }
    return 0;
}
template<bool Enabled> int observed(int value) {
    if constexpr (Enabled) return value;
    else return -value;
}
template<Mode M> struct Wrapper {
    template<bool Enabled> int method(int value) { return observed<Enabled>(value); }
};
int helper(int value) { return value + 1; }
int helper(double value) { return int(value) - 1; }
#if ACTIVE
int active() { return 1; }
#else
int inactive() { return 0; }
#endif
int use() { return Wrapper<Mode::B>{}.method<true>(1); }
}
"""


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        header = self.root / "tt_llk_quasar/llk_lib/fixture.h"
        header.parent.mkdir(parents=True)
        header.write_text(SOURCE)
        self.header = header

    def scan(self, source=SOURCE):
        self.header.write_text(source)
        inventory = llk_ast.scan(
            str(self.header), ["-x", "c++", "-std=c++20", "-DACTIVE=1"]
        )
        return inventory, json.loads(llk_ast.to_json(inventory))

    def test_uninstantiated_templates_and_runtime_decisions(self):
        inventory, scan = self.scan()
        self.assertTrue(inventory.complete, inventory.diagnostics)
        self.assertTrue(scan["complete"])
        definitions = scan["definitions"]
        unused = next(d for d in definitions if d["function"] == "unused")
        self.assertFalse(
            any(i["definition"] == unused["id"] for i in scan["instances"].values())
        )
        enum = unused["parameters"][1]
        self.assertEqual(enum["kind"], "enum")
        self.assertEqual(
            enum["values"],
            [{"value": "4", "names": ["A", "Alias"]}, {"value": "7", "names": ["B"]}],
        )
        self.assertEqual(unused["runtime_parameters"][0]["enum_type"], "test::Mode")
        self.assertEqual(
            [(d["kind"], d["compile_time"]) for d in unused["decisions"]],
            [("if", True), ("if", False), ("&&", False), ("conditional", False)],
        )
        self.assertIn(8, unused["executable_lines"])
        self.assertEqual(
            len({d["id"] for d in definitions if d["function"] == "helper"}), 2
        )
        self.assertIn("active", {d["function"] for d in definitions})
        self.assertNotIn("inactive", {d["function"] for d in definitions})

    def test_member_instantiation_maps_to_primary_definition(self):
        _, scan = self.scan()
        methods = [d for d in scan["definitions"] if d["function"] == "method"]
        self.assertEqual(len(methods), 1)
        method = methods[0]
        self.assertEqual([p["name"] for p in method["parameters"]], ["M", "Enabled"])
        instances = [
            i for i in scan["instances"].values() if i["definition"] == method["id"]
        ]
        self.assertEqual(
            [i["arguments"] for i in instances], [["(test::Mode)7", "true"]]
        )

    def test_failed_parse_is_explicit(self):
        inventory, scan = self.scan(SOURCE + "\nint broken = unknown;\n")
        self.assertFalse(inventory.complete)
        self.assertFalse(scan["complete"])
        self.assertGreater(scan["errors"], 0)

    def test_macro_expansion_changes_definition_identity(self):
        bodies = []
        for value in (1, 2):
            _, scan = self.scan(
                f"#define VALUE {value}\nint helper() {{ return VALUE; }}"
            )
            bodies.append(scan["definitions"][0]["body_digest"])
        self.assertNotEqual(*bodies)

    def constraints(self, body):
        """Scanned functions as analyzer objects and as the dicts the coverage store keeps."""
        macro = "do { if (!(condition)) __builtin_trap(); } while (0)"
        inventory, scan = self.scan(
            "#define LLK_ASSERT(condition, message) " + macro + "\n" + body
        )
        self.assertTrue(inventory.complete, inventory.diagnostics)
        for definition in scan["definitions"]:
            for parameter in (
                definition["parameters"] + definition["runtime_parameters"]
            ):
                for value in parameter["values"]:
                    value["value"] = int(value["value"])
        return (
            {f.name: f for f in inventory.functions},
            {d["function"]: d for d in scan["definitions"]},
        )

    def test_integer_domain_and_joint_constraint_filter(self):
        _, definitions = self.constraints(
            """
template<bool B, int N> void choices() {
    static_assert(N == 1 || N == 2 || N == 4);
    constexpr bool enabled = B;
    static_assert(!enabled || N == 4);
}
"""
        )
        definition = definitions["choices"]
        definition.update(arch="quasar", header="fixture.h")
        row = audit_group(definition, definition["parameters"], [], ("math", "{}", ()))
        self.assertEqual(row["declared_combinations"], 6)
        self.assertEqual(row["assertion_excluded_combinations"], 2)
        self.assertEqual(row["candidate_combinations"], 4)
        self.assertEqual(row["constraint_unknown_combinations"], 0)
        self.assertEqual(len(row["missing_preview"]), 4)

    def test_filter_keeps_conflicts_and_handles_unbounded_axes(self):
        _, definitions = self.constraints(
            """
template<bool B> void conflict() { static_assert(B); }
template<int N> void unbounded() { LLK_ASSERT(N != 0, "nonzero"); }
"""
        )
        definition = definitions["conflict"]
        definition.update(arch="quasar", header="fixture.h")
        row = audit_group(
            definition,
            definition["parameters"],
            [
                {
                    "arguments": ["false"],
                    "execution_count": 0,
                    "signature": "conflict<false>",
                }
            ],
            ("math", "{}", ()),
        )
        self.assertEqual(row["status"], "unresolved")
        self.assertEqual(row["candidate_combinations"], 2)
        self.assertEqual(len(row["constraint_conflicts"]), 1)
        definition = definitions["unbounded"]
        definition.update(arch="quasar", header="fixture.h")
        row = audit_group(
            definition, definition["parameters"], [], ("math", "{}", (("N", "0"),))
        )
        self.assertEqual(row["candidate_combinations"], 0)
        self.assertEqual(row["missing_combinations"], 0)

    def test_static_filter_agrees_with_sfpi_gcc(self):
        compiler = (
            Path(__file__).resolve().parents[2] / "sfpi/compiler/bin/riscv-tt-elf-g++"
        )
        if not compiler.exists():
            self.skipTest("SFPI compiler is not installed")
        source = """
template<bool B, unsigned N> void probe() {
    static_assert(N == 1 || N == 2 || N == 4);
    if constexpr (B) { static_assert(N == 4); }
    else { static_assert(N % 2 == 0); }
}
"""
        functions, _ = self.constraints(source)
        for enabled in (0, 1):
            for number in (1, 2, 3, 4):
                with self.subTest(enabled=enabled, number=number):
                    accepted = all(
                        llk_ast.check(
                            a, {("template", 0): enabled, ("template", 1): number}
                        )
                        is True
                        for a in functions["probe"].assertions
                    )
                    result = subprocess.run(
                        [
                            str(compiler),
                            "-std=c++17",
                            "-fsyntax-only",
                            "-x",
                            "c++",
                            "-",
                        ],
                        input=source + f"template void probe<{enabled}, {number}>();\n",
                        text=True,
                        capture_output=True,
                    )
                    self.assertEqual(accepted, result.returncode == 0, result.stderr)

    def test_snapshots_survive_merge_without_execution_records(self):
        _, scan = self.scan()
        scan.update(
            arch="quasar",
            trisc="math",
            test="fixture",
            variant="v1",
            build_axes={},
            diagnostics="",
        )
        scan["files"] = [{"path": str(self.header), "parsed": True}]
        for definition in scan["definitions"]:
            definition.update(
                key=definition["id"],
                arch="quasar",
                trisc="math",
                build_axes={},
                header=definition.pop("file"),
                source_digest="snapshot",
                template_parameters=[
                    p["declaration"] for p in definition["parameters"]
                ],
                source_lines=SOURCE.splitlines()[
                    definition["start_line"] - 1 : definition["end_line"]
                ],
            )
            for parameter in (
                definition["parameters"] + definition["runtime_parameters"]
            ):
                for value in parameter["values"]:
                    value["value"] = int(value["value"])
        with Store(self.root / "shard.sqlite") as store:
            store.replace_variant("quasar", "fixture", "v1", [], [scan])
        with Store(self.root / "merged.sqlite") as store:
            store.merge(self.root / "shard.sqlite")
            self.assertEqual(store.scans(), [scan])
            write_report(
                [],
                self.root / "report",
                root=self.root,
                architectures=["quasar"],
                scans=store.scans(),
            )
        report = json.loads((self.root / "report/instantiations.json").read_text())
        self.assertTrue(report["baseline"]["complete"])
        self.assertGreater(report["baseline"]["counts"]["absent"], 0)
        (self.header.parent / "never_included.h").write_text("void never_included() {}")
        baseline = source_baseline([], self.root, ["quasar"], [scan])
        self.assertFalse(baseline["complete"])
        self.assertEqual(
            baseline["unparsed_headers"]["quasar"],
            ["tt_llk_quasar/llk_lib/never_included.h"],
        )
        scan["complete"] = False
        self.assertEqual(
            source_baseline([], self.root, ["quasar"], [scan])["definitions"], []
        )

    def test_gcov_joins_helpers_by_symbol(self):
        symbol = "_ZN4test6helperEi"
        document = {
            "files": [
                {
                    "file": str(self.header),
                    "lines": [],
                    "functions": [
                        {
                            "name": symbol,
                            "demangled_name": "test::helper(int)",
                            "start_line": 19,
                            "blocks": 1,
                            "blocks_executed": 1,
                            "execution_count": 1,
                        }
                    ],
                }
            ]
        }
        records = list(
            function_records(
                document,
                {
                    symbol: {
                        "definition": "ast-key",
                        "function": "helper",
                        "arguments": [],
                    }
                },
            )
        )
        self.assertEqual(records[0]["definition"], "ast-key")
        self.assertEqual(records[0]["function"], "helper")


if __name__ == "__main__":
    unittest.main()
