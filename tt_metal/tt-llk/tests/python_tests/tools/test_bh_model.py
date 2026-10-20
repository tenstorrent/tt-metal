# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for transitive effect resolution (tools.llk_state_audit.bh.model.CallGraph).

A tiny synthetic Blackhole tree pins parameter substitution, ``if constexpr``
folding, re-normalization, callable parameters with pack expansion, member-call
typing, overload choice and thread propagation.
"""

import tempfile
import unittest
from pathlib import Path

from tools.llk_state_audit.bh import effects_db, model

SYNTHETIC_FILES = {
    "tt_metal/hw/inc/internal/tt-1xx/blackhole/cfg_defines.h": """
// Registers for THREAD
#define DEST_TARGET_REG_CFG_MATH_Offset_ADDR32 1
#define DEST_TARGET_REG_CFG_MATH_Offset_SHAMT 0
#define DEST_TARGET_REG_CFG_MATH_Offset_MASK 0xfff
// Registers for ALU
#define ALU_ACC_CTRL_Fp32_enabled_ADDR32 1
#define ALU_ACC_CTRL_Fp32_enabled_SHAMT 29
#define ALU_ACC_CTRL_Fp32_enabled_MASK 0x20000000
#define ALU_ACC_CTRL_Fp32_enabled_RMW ALU_ACC_CTRL_Fp32_enabled_ADDR32, ALU_ACC_CTRL_Fp32_enabled_SHAMT, ALU_ACC_CTRL_Fp32_enabled_MASK
""",
    "tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_gpr_map.h": """
namespace ckernel {
struct p_gpr_pack { constexpr static std::uint32_t TMP0 = 28; };
}
""",
    "tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel.h": """
namespace ckernel {
template <std::uint32_t CfgAddr32, std::uint32_t Shamt, std::uint32_t Mask>
inline void cfg_reg_rmw_tensix(std::uint32_t val)
{
    TT_RMWCIB0(Mask & 0xff, val << Shamt, CfgAddr32);
}
}
""",
    "tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_addrmod.h": """
namespace ckernel {
constexpr std::uint8_t ADDR_MOD_3 = 3;
constexpr std::uint8_t ADDR_MOD_5 = 5;
struct addr_mod_t
{
    __attribute__((always_inline)) inline void set(const std::uint8_t mod_index) const
    {
        TTI_SETC16(addr_mod_src_reg_addr[mod_index], 1);
    }
};
}
""",
    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_test.h": """
namespace ckernel {
inline void configure_slot(const std::uint8_t slot) { addr_mod_t {}.set(slot); }

template <bool fp32>
inline void _llk_math_test_init_(const std::uint32_t mode)
{
    if constexpr (fp32)
    {
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(1);
    }
    else
    {
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(0);
    }
    addr_mod_t {.srca = {.incr = 1}}.set(ADDR_MOD_3);
    configure_slot(ADDR_MOD_5);
    TTI_SETDMAREG(0, mode, 0, LO_16(p_gpr_pack::TMP0));
}

template <class F, class... ARGS>
inline void run_init(F&& init_func, ARGS&&... args) { init_func(std::forward<ARGS>(args)...); }

inline void my_init(std::uint32_t v) { sfpi::vConstFloatPrgm0 = v; }

inline void _llk_math_test_(std::uint32_t dst) { TTI_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst); }

inline void overloaded(std::uint32_t a) { TTI_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, a); }
inline void overloaded(std::uint32_t a, std::uint32_t b) { TTI_SFPCONFIG(a, b, 0); }
}
""",
    "tt_metal/hw/inc/api/compute/test_api.h": """
namespace ckernel {
template <bool fp32 = false>
ALWI void test_tile(std::uint32_t idst) {
    MATH((_llk_math_test_init_<fp32>(7)));
    MATH((_llk_math_test_(idst)));
    MATH((run_init(my_init, 5)));
}
ALWI void test_tile_init() { MATH((overloaded(3, 0xB))); }
}
""",
}


def write_tree(root: Path, files: dict[str, str] | None = None) -> Path:
    for relative, text in (files or SYNTHETIC_FILES).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


class TransitiveResolutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = write_tree(Path(cls._tmp.name))
        cls.db = effects_db.build(cls.root, families=[])

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def effects_of(self, name: str) -> list[model.Effect]:
        (fn,) = self.db.by_name(name)
        return list(self.db.effects(fn.key))

    def by_resource(self, effects, resource: str) -> list[model.Effect]:
        return [effect for effect in effects if effect.resource == resource]

    def test_constexpr_branch_folds_and_value_substitutes(self) -> None:
        # In its own scope test_tile's template parameter is free: both branches stay, guarded.
        effects = self.effects_of("test_tile")
        own = self.by_resource(effects, "CFG.ALU_ACC_CTRL_Fp32_enabled")
        self.assertEqual(
            sorted((e.value, e.condition) for e in own),
            [("0", "!(fp32)"), ("1", "fp32")],
        )
        self.assertTrue(all(e.params == ("fp32",) for e in own))
        # A call that relies on the default (fp32 = false) folds to the else branch.
        (api,) = self.db.by_name("test_tile")
        called = self.db.call_effects("test_tile", args=("d",), evidence="k.cpp:1")
        fp32 = self.by_resource(called, "CFG.ALU_ACC_CTRL_Fp32_enabled")
        self.assertEqual([e.value for e in fp32], ["0"])
        self.assertEqual(fp32[0].condition, "")
        self.assertEqual(fp32[0].thread, "math")
        self.assertEqual(fp32[0].granularity, "field")
        self.assertTrue(
            fp32[0].evidence.startswith(
                "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_test.h:"
            )
        )
        self.assertEqual(fp32[0].via[0], (api.key, "k.cpp:1"))
        self.assertEqual(
            fp32[0].via[1][0].split("<")[0], "ckernel::_llk_math_test_init_"
        )
        # no descent into cfg_reg_rmw_tensix (stop pattern): no byte-granular duplicate
        self.assertFalse([e for e in effects if e.pattern == "core.rmwcib"])

    def test_resource_renormalizes_after_substitution(self) -> None:
        effects = self.effects_of("test_tile")
        self.assertTrue(self.by_resource(effects, "ADDR_MOD.3"))
        slot5 = self.by_resource(effects, "ADDR_MOD.5")
        self.assertTrue(slot5)
        self.assertTrue(slot5[0].resolved)
        self.assertEqual(
            [step[0] for step in slot5[0].via][-1],
            "ckernel::configure_slot(const std::uint8_t)",
        )
        (helper,) = self.db.by_name("configure_slot")
        own = self.db.effects(helper.key)
        self.assertEqual([e.resource for e in own], ["ADDR_MOD.?slot"])
        self.assertFalse(own[0].resolved)
        self.assertEqual(own[0].params, ("slot",))

    def test_param_flow_and_value_kinds(self) -> None:
        effects = self.effects_of("test_tile")
        offset = self.by_resource(effects, "THDCFG.DEST_TARGET_REG_CFG_MATH_Offset")
        self.assertEqual(
            (offset[0].value, offset[0].value_kind, offset[0].params),
            ("idst", "param", ("idst",)),
        )
        gpr = self.by_resource(effects, "GPR.pack.TMP0")
        self.assertEqual(
            (gpr[0].value, gpr[0].value_kind, gpr[0].detail), ("7", "const", "LO_16")
        )

    def test_callable_parameter_and_pack_expansion(self) -> None:
        effects = self.effects_of("test_tile")
        constant = self.by_resource(effects, "SFPU.LREG12")
        self.assertEqual([e.value for e in constant], ["5"])
        self.assertEqual(
            [step[0] for step in constant[0].via],
            [
                "ckernel::run_init<class,class...>(F&&,ARGS&&...)",
                "ckernel::my_init(std::uint32_t)",
            ],
        )

    def test_overload_by_arity(self) -> None:
        effects = self.effects_of("test_tile_init")
        self.assertEqual([e.resource for e in effects], ["SFPU.LREG11"])
        (init,) = self.db.by_name("test_tile_init")
        self.assertEqual(init.role, "init")
        self.assertEqual(init.calls[0].resolution, "exact")

    def test_roles_threads_and_call_effects(self) -> None:
        (lib_init,) = self.db.by_name("_llk_math_test_init_")
        self.assertEqual(
            (lib_init.role, lib_init.thread, lib_init.layer),
            ("init", "math", "llk_lib"),
        )
        (api,) = self.db.by_name("test_tile")
        self.assertEqual(
            (api.role, api.thread, api.layer), ("execute", "any", "compute_api")
        )
        called = self.db.call_effects(
            "test_tile", template_args=("true",), args=("dst_i",), evidence="k.cpp:3"
        )
        values = {
            e.resource: e.value
            for e in called
            if e.resource.startswith(("CFG.", "THDCFG."))
        }
        self.assertEqual(values["CFG.ALU_ACC_CTRL_Fp32_enabled"], "1")
        self.assertEqual(values["THDCFG.DEST_TARGET_REG_CFG_MATH_Offset"], "dst_i")
        self.assertTrue(all(e.via[0] == (api.key, "k.cpp:3") for e in called))
        reach = [key for key, _, _ in self.db.reachable(api.key)]
        self.assertIn(lib_init.key, reach)

    def test_substitute_and_classify_helpers(self) -> None:
        self.assertEqual(model.substitute("a + b.a", {"a": "x * 2"}), "(x * 2) + b.a")
        self.assertEqual(model.substitute("ns::a(a)", {"a": "q"}), "ns::a(q)")
        self.assertEqual(model.simplify_arg("std::forward<F>(f)"), "f")
        self.assertEqual(model.classify_value("p_setadc::PAC", ["x"]), "const")
        self.assertEqual(model.classify_value("x", ["x"]), "param")
        self.assertEqual(model.classify_value("x + y", ["x"]), "expr")
        self.assertEqual(model.classify_value("", ["x"]), "unknown")


if __name__ == "__main__":
    unittest.main()
