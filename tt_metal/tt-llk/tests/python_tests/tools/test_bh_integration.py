# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Integration fixes of the Blackhole state audit: condition folding and precedence, scanner
masking of unevaluated macro arguments, cfg mask resolution, replay-recording deferral,
supersede rules, and the checker's program-order / restore / duplicate handling."""

import tempfile
import types
import unittest
from pathlib import Path

from test_bh_model import write_tree

from tools.llk_state_audit.bh import effects_db, hazards, kernels, resources, scan
from tools.llk_state_audit.bh.model import Effect, Pattern, combine_conditions

REPO_ROOT = Path(__file__).resolve().parents[5]


def _fn(text: str, name: str) -> scan.FunctionDef:
    source = scan.scan_source(
        text, "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/demo.h", "llk_lib"
    )
    (fn,) = [f for f in source.functions if f.name == name]
    return fn


class ConditionTest(unittest.TestCase):
    def test_low_precedence_parts_are_parenthesized(self) -> None:
        self.assertEqual(scan.join_conditions("a || b", "c", "a || b"), "(a || b) && c")
        self.assertEqual(scan.join_conditions("x ? y : z", "w"), "(x ? y : z) && w")
        self.assertEqual(combine_conditions("p", "q || r"), "p && (q || r)")
        fn = _fn(
            """
inline void f(int x, int a, int b) {
    if (x) { g(); } else if (a || b) { h(); }
}
""",
            "f",
        )
        (call,) = [c for c in fn.calls if c.name == "h"]
        self.assertEqual(call.condition, "!(x) && (a || b)")

    def test_qualified_and_bare_enumerators_compare(self) -> None:
        self.assertIs(scan.truth("ckernel::BinaryOp::ADD == BinaryOp::ADD"), True)
        self.assertIs(scan.truth("BinaryOp::ADD == ckernel::BinaryOp::DIV"), False)
        self.assertIs(scan.truth("StartZero == DstStart::StartZero"), True)
        self.assertIsNone(scan.truth("x == DstStart::StartZero"))
        self.assertIsNone(scan.truth("A::X == B::X"))

    def test_conjuncts_decide_when_the_whole_text_does_not_tokenize(self) -> None:
        self.assertIs(scan.truth("cfg.get() > 1 && A::X == A::Y"), False)
        self.assertIs(scan.truth("!(cfg.get() > 1 || true)"), False)
        self.assertIs(scan.truth("cfg.get() > 1 || A::X == A::X"), True)
        self.assertIsNone(scan.truth("cfg.get() > 1 && A::X == A::X"))
        # a capped text: the incomplete last conjunct is not evaluated
        self.assertIs(scan.truth("A::X == A::Y && A::Z == A::Q..."), False)
        self.assertIsNone(scan.truth("A::X == A::X && A::Z == A::Z..."))
        # a template constexpr helper is opaque, not a comparison chain
        self.assertEqual(
            scan.evaluate("(A::X == A::X && A::Y == A::Z && !f<A::X, B>()) ? 1u : 0u"),
            0,
        )
        self.assertEqual(scan.evaluate("a < b && c > d"), scan.UNKNOWN)

    def test_simplify_drops_true_conjuncts(self) -> None:
        self.assertEqual(
            scan.simplify_condition(
                "!(DstSync::SyncHalf == DstSync::SyncFull) && dest_offset_id"
            ),
            (None, "dest_offset_id"),
        )
        self.assertEqual(
            scan.simplify_condition("A::X == A::Y && v"), (False, "A::X == A::Y && v")
        )
        self.assertEqual(scan.simplify_condition("A::X == A::X"), (True, ""))
        self.assertEqual(
            scan.simplify_condition("(a || b) && c"), (None, "(a || b) && c")
        )


class ScannerTest(unittest.TestCase):
    def test_assert_arguments_are_not_analysed(self) -> None:
        fn = _fn(
            """
inline void f() {
    LLK_ASSERT(is_configured(), "msg");
    LLK_ASSERT_BLOCK(check_all());
    real_call();
}
""",
            "f",
        )
        self.assertEqual(
            [c.name for c in fn.calls], ["LLK_ASSERT", "LLK_ASSERT_BLOCK", "real_call"]
        )

    def test_qualified_call_split_across_lines(self) -> None:
        fn = _fn(
            """
inline void f(int x) {
    detail::
        impl<4>(x);
}
""",
            "f",
        )
        self.assertIn("impl", [c.name for c in fn.calls])


@unittest.skipUnless(
    (REPO_ROOT / resources.CFG_DEFINES).is_file(), "cfg_defines.h not present"
)
class CfgMaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tables = resources.ResourceTables(REPO_ROOT, {})

    def names(self, expr: str) -> list[str]:
        return [n.name for n in self.tables.normalize("cfg_rmw", expr)]

    def test_mask_names_every_covered_field(self) -> None:
        self.assertEqual(
            sorted(
                self.names(
                    "ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32, 0, ALU_FORMAT_SPEC_REG_SrcA_val_MASK | "
                    "ALU_FORMAT_SPEC_REG_SrcB_val_MASK"
                )
            ),
            ["CFG.ALU_FORMAT_SPEC_REG_SrcA_val", "CFG.ALU_FORMAT_SPEC_REG_SrcB_val"],
        )
        # SHAMT of a field in another word does not rename the written field
        self.assertEqual(
            self.names(
                "UNP0_ADDR_CTRL_XY_REG_1_Ystride_ADDR32, UNP0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT, "
                "UNP0_ADDR_CTRL_XY_REG_1_Ystride_MASK"
            ),
            ["CFG.UNP0_ADDR_CTRL_XY_REG_1_Ystride"],
        )

    def test_adc_dimensions_overlap(self) -> None:
        self.assertTrue(self.tables.overlaps("ADC.UNP_A.XY", "ADC.UNP_A.X"))
        self.assertTrue(self.tables.overlaps("ADC.UNP_A.ZW", "ADC.UNP_A.W"))
        self.assertFalse(self.tables.overlaps("ADC.UNP_A.XY", "ADC.UNP_A.Z"))
        self.assertFalse(self.tables.overlaps("ADC.UNP_A.X", "ADC.UNP_B.X"))


EXTRA = {
    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_demo_replay.h": """
namespace ckernel {
inline void _demo_replay_init_() {
    load_replay_buf(0, 4, [] {
        TTI_SETDMAREG(0, 5, 0, LO_16(p_gpr_unpack::TMP0));
    });
    TTI_SETDMAREG(0, 6, 0, LO_16(p_gpr_unpack::TMP1));
    TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP1, p_gpr_unpack::TILE_SIZE_A);
    while (semaphore_read(semaphore::MATH_PACK) > 0) {}
}
inline void _demo_super_() { cfg[upk0_reg] = 3; }
}
""",
}


class EffectDBTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = write_tree(Path(cls._tmp.name))
        for path, text in EXTRA.items():
            target = cls.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        module = types.ModuleType("superdemo")
        module.FAMILY = "superdemo"
        module.PATTERNS = [
            Pattern(
                "superdemo.cfg",
                "assign",
                r"cfg",
                "literal:THCON_SEC0_REG3_Base_address_ADDR32",
                "cfg_addr32",
                value="rhs",
                when=lambda m: m.index == "upk0_reg",
            ),
        ]
        module.SUPERSEDES = {"superdemo.cfg": ("core.cfg_assign",)}
        cls.db = effects_db.build(cls.root, families=[], modules=[module])

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def effects(self, name: str):
        (model,) = self.db.by_name(name)
        return model.effects

    def test_replay_recorded_instructions_are_deferred(self) -> None:
        rows = {
            (e.pattern, e.resource, e.detail)
            for e in self.effects("_demo_replay_init_")
        }
        # the normalizer's own detail (GPR half) is kept; recording appends ``deferred``
        self.assertIn(("core.setdmareg", "GPR.unpack.TMP0", "LO_16;deferred"), rows)
        self.assertIn(("core.setdmareg", "GPR.unpack.TMP1", "LO_16"), rows)
        self.assertTrue(hazards.is_deferred("LO_16;deferred"))

    def test_dmareg_reads_both_operands_and_semaphore_read(self) -> None:
        rows = {
            (e.pattern, e.resource, e.access)
            for e in self.effects("_demo_replay_init_")
        }
        self.assertIn(("core.dmareg_alu.write", "GPR.unpack.TMP0", "write"), rows)
        self.assertIn(("core.dmareg_alu.read", "GPR.unpack.TILE_SIZE_A", "read"), rows)
        self.assertIn(("core.dmareg_alu.read_b", "GPR.unpack.TMP1", "read"), rows)
        self.assertIn(("core.semaphore_read", "SEM.MATH_PACK", "read"), rows)

    def test_supersedes_drops_the_coarse_core_effect(self) -> None:
        patterns = sorted(e.pattern for e in self.effects("_demo_super_"))
        self.assertEqual(patterns, ["superdemo.cfg"])

    def test_direct_values_are_classified(self) -> None:
        kinds = {e.value_kind for e in self.effects("_demo_replay_init_") if e.value}
        self.assertNotIn("unknown", kinds)


def _effect(
    resource: str, value: str, evidence: str, via=(), access="write", detail=""
) -> Effect:
    return Effect(
        resource=resource,
        access=access,
        value_kind="const",
        value=value,
        condition="",
        evidence=evidence,
        thread="pack",
        via=via,
        detail=detail,
    )


class CheckerHelpersTest(unittest.TestCase):
    def test_deferred_spellings(self) -> None:
        for detail in ("deferred", "deferred:replay", "start=0;len=4;deferred"):
            self.assertTrue(hazards.is_deferred(detail), detail)
        for detail in ("", "start=0;len=4", "src=wrcfg"):
            self.assertFalse(hazards.is_deferred(detail), detail)

    def test_program_order_inside_one_execute(self) -> None:
        init = _effect(
            "CFG.X",
            "1",
            "a.h:50",
            via=(("op", ""), ("init", "api.h:10"), ("cfg", "b.h:3")),
        )
        uninit = _effect(
            "CFG.X", "0", "a.h:40", via=(("op", ""), ("uninit", "api.h:12"))
        )
        self.assertLess(hazards.program_key(init), hazards.program_key(uninit))

    def test_restore_of_a_saved_gpr(self) -> None:
        save = _effect(
            "GPR.unpack.SR_STATE_1",
            "CFG(THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1)",
            "a.h:1",
        )
        saved = hazards._saved_gprs([save])
        self.assertEqual(saved, {"SR_STATE_1"})
        self.assertTrue(
            hazards._is_restore(
                _effect("CFG.Y", "GPR(p_gpr_unpack::SR_STATE_1)", "a.h:9"), saved
            )
        )
        self.assertFalse(
            hazards._is_restore(
                _effect("CFG.Y", "GPR(p_gpr_unpack::TMP0)", "a.h:9"), saved
            )
        )

    def test_ternary_values_fold_through_their_condition(self) -> None:
        self.assertEqual(hazards._split_ternary("(c ? a::b : d)"), ("c", "a::b", "d"))
        self.assertIsNone(hazards._split_ternary("a::b"))

    def test_duplicates_merge(self) -> None:
        base = {
            "kind": "ordering",
            "resource": "CFG.X",
            "ops": ["f"],
            "confidence": "medium",
            "summary": "s",
            "kernels": ["k1"],
            "kernel_count": 1,
            "evidence": ["a.h:1"],
            "_score": 1.0,
        }
        other = dict(
            base,
            kernels=["k2"],
            evidence=["a.h:2"],
            _score=2.0,
            summary="t",
            confidence="high",
        )
        (merged,) = hazards.merge_duplicates([base, other])
        self.assertEqual(
            (
                merged["kernels"],
                merged["kernel_count"],
                merged["summary"],
                merged["confidence"],
            ),
            (["k1", "k2"], 2, "t", "high"),
        )

    def test_if_constexpr_arms_are_exclusive(self) -> None:
        self.assertTrue(kernels.exclusive_conditions("r && a && b", "r && !((a && b))"))
        self.assertTrue(kernels.exclusive_conditions("r && !(a && b)", "a && r && b"))
        self.assertFalse(kernels.exclusive_conditions("r && a", "r && !((a && b))"))


if __name__ == "__main__":
    unittest.main()
