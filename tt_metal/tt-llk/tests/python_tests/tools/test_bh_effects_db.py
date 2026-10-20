# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Blackhole effect DB, family plugins, findings and CLI (bh-effects/bh-hazards)."""

import importlib
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

from test_bh_model import SYNTHETIC_FILES, write_tree

from tools.llk_state_audit.bh import effects_db
from tools.llk_state_audit.bh.model import Pattern

LLK_ROOT = Path(__file__).resolve().parents[3]

EXTRA_FILES = {
    "tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_demo.h": """
namespace ckernel::sfpu {
inline void demo_init() { l_reg[LRegs::LReg3] = 1; TTI_SFPLOADI(p_sfpu::LREG2, 0, 0x3f80); }
template <bool APPROX>
inline void calculate_demo() { sfpi::vFloat x = sfpi::vConstFloatPrgm0; regfile[p_gpr_pack::TMP0] = 4; }
}
""",
}


class FamilyPluginTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = write_tree(Path(cls._tmp.name), {**SYNTHETIC_FILES, **EXTRA_FILES})
        cls.example = importlib.import_module(
            "tools.llk_state_audit.bh.families._example"
        )
        cls.db = effects_db.build(cls.root, families=[], modules=[cls.example])

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def direct(self, name: str):
        (fn,) = self.db.by_name(name)
        return fn, fn.effects

    def test_example_family_patterns_fire(self) -> None:
        fn, effects = self.direct("demo_init")
        by_pattern = {e.pattern: e for e in effects}
        self.assertEqual(by_pattern["example.l_reg_write"].resource, "SFPU.LREG3")
        self.assertEqual(by_pattern["example.sfploadi"].resource, "SFPU.LREG2")
        self.assertEqual(by_pattern["example.sfploadi"].value, "mod=0;imm=0x3f80")
        self.assertEqual(by_pattern["example.sfploadi"].family, "example")
        self.assertEqual(fn.role, "init")
        self.assertEqual(fn.layer, "sfpu")

    def test_core_reads_and_writes(self) -> None:
        fn, effects = self.direct("calculate_demo")
        self.assertEqual(fn.role, "execute")
        pairs = {(e.resource, e.access, e.pattern) for e in effects}
        self.assertIn(("SFPU.LREG12", "read", "core.sfpi_constant_read"), pairs)
        self.assertIn(("GPR.pack.TMP0", "write", "core.regfile_assign"), pairs)
        self.assertIn("example", [f.name for f in self.db.families])

    def test_validation_rejects_bad_modules(self) -> None:
        bad = types.ModuleType("bad")
        bad.FAMILY = "bad"
        bad.PATTERNS = [Pattern("other.x", "call", "X", "arg:0")]
        with self.assertRaises(effects_db.FamilyError):
            effects_db._family_from_module(bad, "bad")
        bad.PATTERNS = [Pattern("bad.x", "nonsense", "X", "arg:0")]
        with self.assertRaises(effects_db.FamilyError):
            effects_db._family_from_module(bad, "bad")
        missing = types.ModuleType("missing")
        with self.assertRaises(effects_db.FamilyError):
            effects_db._family_from_module(missing, "missing")

    def test_add_sources_kernels(self) -> None:
        kernel = """
#include "api/compute/test_api.h"
namespace NAMESPACE {
void MAIN {
    for (std::uint32_t i = 0; i < n; ++i) {
        test_tile<true>(i);
    }
}
}
"""
        (main,) = self.db.add_sources([("ttnn/kernels/demo.cpp", kernel)])
        self.assertEqual(
            (main.name, main.role, main.layer), ("MAIN", "execute", "kernel")
        )
        self.assertTrue(main.key.endswith("@ttnn/kernels/demo.cpp"))
        effects = self.db.effects(main.key)
        fp32 = [e for e in effects if e.resource == "CFG.ALU_ACC_CTRL_Fp32_enabled"]
        self.assertEqual([e.value for e in fp32], ["1"])
        self.assertEqual(fp32[0].via[0][1], "ttnn/kernels/demo.cpp:6")
        self.assertNotIn(
            main.key, [f["key"] for f in self.db.to_json(transitive=False)["functions"]]
        )

    def test_json_is_deterministic_and_sorted(self) -> None:
        first = effects_db.dumps_effects(self.db.to_json())
        second = effects_db.dumps_effects(
            effects_db.build(self.root, families=[], modules=[self.example]).to_json()
        )
        self.assertEqual(first, second)
        data = json.loads(first)
        keys = [fn["key"] for fn in data["functions"]]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(data["arch"], "blackhole")
        self.assertEqual(
            sum(data["counts"]["functions_per_layer"].values()),
            data["counts"]["functions"],
        )
        self.assertNotIn(str(self.root), first)


class FindingsTest(unittest.TestCase):
    def test_sorting_ids_and_truncation(self) -> None:
        raw = [
            {
                "rank": 2,
                "kind": "ordering",
                "confidence": "low",
                "resource": "ADDR_MOD.3",
                "ops": ["b", "b", "a"],
                "summary": "  two  ",
                "kernels": [f"k{i:02d}.cpp" for i in range(25)][::-1],
                "evidence": ["x.h:10", "x.h:9", "x.h:10"],
                "extra": "dropped",
            },
            {
                "rank": 1,
                "kind": "fold_drop",
                "confidence": "high",
                "resource": "param:pack_tile.relu",
                "ops": ["pack_tile"],
                "summary": "one",
                "kernels": [],
                "evidence": [],
            },
        ]
        findings = effects_db.normalize_findings(raw)
        self.assertEqual([f["id"] for f in findings], ["H0001", "H0002"])
        self.assertEqual(findings[0]["resource"], "param:pack_tile.relu")
        second = findings[1]
        self.assertEqual(second["ops"], ["b", "a"])
        self.assertEqual(second["summary"], "two")
        self.assertEqual(second["kernel_count"], 25)
        self.assertEqual(len(second["kernels"]), 20)
        self.assertEqual(second["kernels"], sorted(second["kernels"]))
        self.assertEqual(second["evidence"], ["x.h:9", "x.h:10"])
        self.assertNotIn("extra", second)
        self.assertEqual(
            sorted(second),
            [
                "confidence",
                "evidence",
                "id",
                "kernel_count",
                "kernels",
                "kind",
                "ops",
                "rank",
                "resource",
                "summary",
            ],
        )

    def test_invalid_findings_raise(self) -> None:
        base = {"rank": 1, "kind": "ordering", "confidence": "high", "resource": "MOP"}
        for key, value in (
            ("kind", "other"),
            ("confidence", "sure"),
            ("rank", "1"),
            ("resource", ""),
        ):
            with self.assertRaises(ValueError):
                effects_db.normalize_findings([{**base, key: value}])


class CommandLineTest(unittest.TestCase):
    def test_commands_are_byte_identical_and_need_no_git(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = write_tree(Path(tmp) / "tree")
            outputs = []
            for run in range(2):
                for command in ("bh-effects", "bh-hazards"):
                    out = Path(tmp) / f"{command}.{run}.json"
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "tools.llk_state_audit",
                            command,
                            "--repo-root",
                            str(root),
                            "--out",
                            str(out),
                        ],
                        cwd=LLK_ROOT,
                        check=True,
                        capture_output=True,
                        env={"PATH": "/nonexistent", "PYTHONHASHSEED": str(run)},
                    )
                    outputs.append(out.read_bytes())
            self.assertEqual(outputs[0], outputs[2])
            self.assertEqual(outputs[1], outputs[3])
            hazards = json.loads(outputs[1])
            self.assertEqual(sorted(hazards), ["findings", "tree_fingerprint"])
            self.assertEqual(
                hazards["tree_fingerprint"], json.loads(outputs[0])["tree_fingerprint"]
            )

    def test_requires_repo_root_and_out(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "tools.llk_state_audit", "bh-effects"],
            cwd=LLK_ROOT,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
