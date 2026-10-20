# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the refinement pass of the Blackhole checker: include-closure call resolution,
integer-vs-class overload filtering, compile-time thread attribution of kernel-call effects,
prologue/epilogue fold detection, and the deep / feasibility-aware fold_drop matching.
"""

import tempfile
import unittest
from pathlib import Path

from tools.llk_state_audit.bh import effects_db, hazards, kernels

ADDER = "ttnn/cpp/ttnn/operations/eltwise/refine/device/kernels/compute/adder.cpp"
MULER = "ttnn/cpp/ttnn/operations/eltwise/refine/device/kernels/compute/muler.cpp"
THREADS = "ttnn/cpp/ttnn/operations/eltwise/refine/device/kernels/compute/threads.cpp"
TWINS = "ttnn/cpp/ttnn/operations/eltwise/refine/device/kernels/compute/twins.cpp"

REFINE_FILES = {
    "tt_metal/hw/inc/internal/tt-1xx/blackhole/cfg_defines.h": """
// Registers for ALU
#define ALU_ACC_CTRL_Fp32_enabled_ADDR32 1
#define ALU_ACC_CTRL_Fp32_enabled_SHAMT 29
#define ALU_ACC_CTRL_Fp32_enabled_MASK 0x20000000
#define ALU_ACC_CTRL_Fp32_enabled_RMW ALU_ACC_CTRL_Fp32_enabled_ADDR32, ALU_ACC_CTRL_Fp32_enabled_SHAMT, ALU_ACC_CTRL_Fp32_enabled_MASK
// Registers for PACK0
#define THCON_SEC0_REG1_Pack_L1_Acc_ADDR32 70
#define THCON_SEC0_REG1_Pack_L1_Acc_SHAMT 1
#define THCON_SEC0_REG1_Pack_L1_Acc_MASK 0x2
#define THCON_SEC0_REG1_Pack_L1_Acc_RMW THCON_SEC0_REG1_Pack_L1_Acc_ADDR32, THCON_SEC0_REG1_Pack_L1_Acc_SHAMT, THCON_SEC0_REG1_Pack_L1_Acc_MASK
#define PCK_EDGE_OFFSET_SEC0_mask_ADDR32 80
#define PCK_EDGE_OFFSET_SEC0_mask_SHAMT 0
#define PCK_EDGE_OFFSET_SEC0_mask_MASK 0xffff
#define PCK_EDGE_OFFSET_SEC0_mask_RMW PCK_EDGE_OFFSET_SEC0_mask_ADDR32, PCK_EDGE_OFFSET_SEC0_mask_SHAMT, PCK_EDGE_OFFSET_SEC0_mask_MASK
""",
    "tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel.h": """
namespace ckernel {
template <std::uint32_t CfgAddr32, std::uint32_t Shamt, std::uint32_t Mask>
inline void cfg_reg_rmw_tensix(std::uint32_t val)
{
    TT_RMWCIB0(Mask & 0xff, val << Shamt, CfgAddr32);
}
struct TensorShape { std::uint32_t num_faces; };
}
""",
    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_refine.h": """
namespace ckernel {
template <int op>
inline void _llk_math_binary_demo_init_(const std::uint32_t acc)
{
    if constexpr (op == 0)
    {
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(acc);
    }
    else
    {
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(0);
    }
}
inline void _llk_math_binary_demo_(const std::uint32_t idst)
{
    TTI_NOP;
}
inline void _llk_math_sfpu_demo_(const std::uint32_t v)
{
    cfg_reg_rmw_tensix<PCK_EDGE_OFFSET_SEC0_mask_RMW>(v);
}
inline void _llk_pack_mask_demo_(const std::uint32_t v)
{
    cfg_reg_rmw_tensix<PCK_EDGE_OFFSET_SEC0_mask_RMW>(v);
}
inline void _llk_pack_l1_demo_(const std::uint32_t v)
{
    cfg_reg_rmw_tensix<THCON_SEC0_REG1_Pack_L1_Acc_RMW>(v);
}
inline void _llk_pack_tile_demo_(const std::uint32_t idst)
{
    TTI_NOP;
}
}
""",
    "tt_metal/hw/inc/api/compute/refine.h": """
#pragma once
namespace ckernel {
template <int op>
ALWI void demo_binary_init(std::uint32_t icb, std::uint32_t acc = 0) { MATH((_llk_math_binary_demo_init_<op>(acc))); }
ALWI void demo_add_init(std::uint32_t icb, std::uint32_t acc = 0) { demo_binary_init<0>(icb, acc); }
ALWI void demo_mul_init(std::uint32_t icb, std::uint32_t acc = 1) { demo_binary_init<1>(icb, acc); }
ALWI void demo_add_tiles(std::uint32_t icb, std::uint32_t idst) {
    demo_add_init(icb);
    MATH((_llk_math_binary_demo_(idst)));
}
ALWI void demo_mul_tiles(std::uint32_t icb, std::uint32_t idst) {
    demo_mul_init(icb);
    MATH((_llk_math_binary_demo_(idst)));
}
ALWI void demo_mask_config(std::uint32_t v) { PACK((_llk_pack_mask_demo_(v))); }
ALWI void demo_l1_config(std::uint32_t v) { PACK((_llk_pack_l1_demo_(v))); }
// a multi-phase op: its mid-body config is its own sequencing, not a fold
ALWI void demo_two_phase(std::uint32_t idst) {
    demo_l1_config(1);
    PACK((_llk_pack_tile_demo_(idst)));
    demo_mask_config(3);
    PACK((_llk_pack_tile_demo_(idst)));
}
// an execute whose SFPU helper is named _llk_math_* but runs on the thread of the wrapper
ALWI void demo_pack_sfpu_tile(std::uint32_t v) { PACK((_llk_math_sfpu_demo_(v))); }
ALWI void demo_split_thread(std::uint32_t v) {
    MATH((_llk_math_sfpu_demo_(v)));
    PACK((_llk_pack_mask_demo_(v)));
}
ALWI void shape_op(const ckernel::TensorShape& shape, std::uint32_t a) { MATH((_llk_math_sfpu_demo_(1))); }
ALWI void shape_op(std::uint32_t a, std::uint32_t b) { MATH((_llk_math_sfpu_demo_(2))); }
}
""",
    "tt_metal/hw/inc/api/compute/twin_a.h": """
#pragma once
namespace ckernel {
ALWI void twin_tile(std::uint32_t v) { PACK((_llk_pack_l1_demo_(v))); }
}
""",
    "tt_metal/hw/inc/api/compute/twin_b.h": """
#pragma once
namespace ckernel {
ALWI void twin_tile(std::uint32_t v) { PACK((_llk_pack_mask_demo_(v))); }
}
""",
    ADDER: """
#include "api/compute/refine.h"
void kernel_main() {
    demo_binary_init<0>(0, 1);
    for (std::uint32_t i = 0; i < 4; ++i) {
        demo_add_tiles(0, i);
    }
}
""",
    MULER: """
#include "api/compute/refine.h"
void kernel_main() {
    demo_mul_init(0, 0);
    for (std::uint32_t i = 0; i < 4; ++i) {
        demo_mul_tiles(0, i);
    }
}
""",
    THREADS: """
#include "api/compute/refine.h"
void kernel_main() {
    demo_pack_sfpu_tile(5);
    PACK((demo_split_thread(6)));
    demo_two_phase(0);
    shape_op(get_compile_time_arg_val(1), 2);
}
""",
    TWINS: """
#include "api/compute/twin_b.h"
void kernel_main() {
    twin_tile(7);
}
""",
}


def write_refine_tree(root: Path) -> Path:
    for rel, text in REFINE_FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.lstrip("\n"))
    return root


class RefineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = write_refine_tree(Path(cls._tmp.name))
        cls.db = effects_db.build(cls.root, families=[])
        cls.kernels = {k.path: k for k in kernels.extract(cls.db, cls.root)}
        cls.ctx = hazards.Context(cls.db, list(cls.kernels.values()))
        cls.findings = effects_db.normalize_findings(
            hazards.run(cls.db, cls.root)["findings"]
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def call(self, kernel: str, name: str) -> kernels.KCall:
        matches = [c for c in self.kernels[kernel].calls if c.name == name]
        self.assertTrue(matches, name)
        return matches[0]

    # ---- resolution ------------------------------------------------------- #
    def test_twin_definition_resolves_to_the_included_header(self) -> None:
        call = self.call(TWINS, "twin_tile")
        self.assertEqual(len(call.keys), 1)
        self.assertEqual(
            self.db.function(call.keys[0]).path, "tt_metal/hw/inc/api/compute/twin_b.h"
        )
        self.assertEqual(call.resolution, "exact")

    def test_integer_argument_does_not_bind_a_class_parameter(self) -> None:
        call = self.call(THREADS, "shape_op")
        self.assertEqual(len(call.keys), 1)
        fn = self.db.function(call.keys[0])
        self.assertEqual([p.name for p in fn.params], ["a", "b"])

    # ---- compile-time threads --------------------------------------------- #
    def test_wrapper_thread_overrides_a_name_hint(self) -> None:
        effects = self.ctx.effects(self.call(THREADS, "demo_pack_sfpu_tile"))
        writes = [e for e in effects if e.resource == "CFG.PCK_EDGE_OFFSET_SEC0_mask"]
        self.assertTrue(writes)
        self.assertEqual({e.thread for e in writes}, {"pack"})

    def test_effects_inside_another_thread_wrapper_are_compiled_out(self) -> None:
        call = self.call(THREADS, "demo_split_thread")
        self.assertEqual(call.thread, "pack")
        effects = [e for e in self.ctx.effects(call) if e.access == "write"]
        self.assertTrue(effects)
        self.assertEqual({e.thread for e in effects}, {"pack"})
        # only the PACK((..)) half survives on the pack TRISC
        self.assertEqual(
            {e.evidence for e in effects},
            {"tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_refine.h:24"},
        )

    # ---- prologue / epilogue ---------------------------------------------- #
    def test_mid_body_config_is_not_a_fold(self) -> None:
        call = self.call(THREADS, "demo_two_phase")
        by_resource = {}
        for effect in self.ctx.effects(call):
            if effect.access == "write":
                by_resource[effect.resource] = hazards.fold_path(
                    self.db, hazards._Access(call, effect)
                )
        self.assertTrue(by_resource["CFG.THCON_SEC0_REG1_Pack_L1_Acc"])  # prologue
        self.assertFalse(
            by_resource["CFG.PCK_EDGE_OFFSET_SEC0_mask"]
        )  # between executes

    # ---- fold_drop -------------------------------------------------------- #
    def test_deep_setter_match_names_the_shallow_parameter(self) -> None:
        drops = [f for f in self.findings if f["kind"] == "fold_drop"]
        self.assertEqual([f["resource"] for f in drops], ["param:demo_add_tiles.acc"])
        finding = drops[0]
        self.assertEqual(finding["kernels"], [ADDER])
        self.assertIn("demo_binary_init", finding["ops"])
        self.assertIn(
            f"{ADDER}:3", finding["evidence"]
        )  # the kernel's generic-level setter

    def test_parameter_without_a_feasible_sink_is_not_dropped(self) -> None:
        # demo_mul_tiles pins acc=1 over the kernel's 0, but op 1 never writes acc
        for finding in self.findings:
            self.assertNotIn(MULER, finding["kernels"])

    def test_define_only_difference_is_not_comparable(self) -> None:
        self.assertFalse(
            hazards._comparable(
                self.db,
                "f<EltwiseBinaryType::ELWADD, MathFidelity::LoFi>",
                "f<EltwiseBinaryType::ELWADD, MATH_FIDELITY>",
            )
        )
        self.assertTrue(
            hazards._comparable(self.db, "ReluConfig::none()", "ReluConfig::zero()")
        )
        self.assertFalse(hazards._comparable(self.db, "APPROX", "false"))

    def test_setter_stem(self) -> None:
        self.assertEqual(hazards._setter_stem("pack_untilize_init"), "pack_untilize")
        self.assertEqual(hazards._setter_stem("tilize_init_short_with_dt"), "tilize")
        self.assertEqual(hazards._setter_stem("pack_tile"), "")


if __name__ == "__main__":
    unittest.main()
