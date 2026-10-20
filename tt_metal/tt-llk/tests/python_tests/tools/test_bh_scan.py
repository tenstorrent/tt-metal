# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Blackhole C++ function extractor (tools.llk_state_audit.bh.scan).

Real-file tests run against the Blackhole headers of this checkout; synthetic
snippets pin the preprocessor, header-parsing and call-site rules.
"""

import unittest
from pathlib import Path

from tools.llk_state_audit.bh import scan

REPO_ROOT = Path(__file__).resolve().parents[5]
BH_COMMON = "tt_metal/tt-llk/tt_llk_blackhole/common/inc"
BH_LIB = "tt_metal/tt-llk/tt_llk_blackhole/llk_lib"
COMPUTE = "tt_metal/hw/inc/api/compute"
SFPU_MACROS = "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"


def _scan_real(*paths: str) -> scan.ScanIndex:
    return scan.scan_tree(
        REPO_ROOT, sources=[(path, scan.layer_of(path)) for path in sorted(paths)]
    )


def _by(index: scan.ScanIndex, name: str, **filters) -> list[scan.FunctionDef]:
    return [
        fn
        for fn in index.by_name(name)
        if all(getattr(fn, key) == value for key, value in filters.items())
    ]


@unittest.skipUnless(
    (REPO_ROOT / BH_COMMON / "ckernel.h").is_file(), "Blackhole sources not present"
)
class RealBlackholeFilesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index = _scan_real(
            f"{BH_COMMON}/ckernel.h",
            f"{BH_COMMON}/ckernel_addrmod.h",
            f"{BH_COMMON}/ckernel_template.h",
            f"{COMPUTE}/eltwise_binary.h",
            f"{COMPUTE}/eltwise_unary/exp.h",
            SFPU_MACROS,
        )

    def test_layers_and_paths_are_repo_relative(self) -> None:
        for fn in self.index.functions:
            self.assertFalse(fn.path.startswith("/"), fn.path)
            self.assertIn(fn.layer, scan.LAYERS)
        self.assertEqual(scan.layer_of(SFPU_MACROS), "llk_api")
        self.assertEqual(
            scan.layer_of(f"{BH_COMMON}/sfpu/ckernel_sfpu_load_config.h"), "sfpu"
        )
        self.assertEqual(scan.layer_of(f"{BH_LIB}/llk_unpack_A.h"), "llk_lib")
        self.assertIsNone(scan.layer_of("ttnn/cpp/foo.h"))

    def test_template_function_params_and_rmw_calls(self) -> None:
        (fn,) = _by(self.index, "cfg_reg_rmw_tensix")
        self.assertEqual(
            [p.name for p in fn.template_params], ["CfgAddr32", "Shamt", "Mask"]
        )
        self.assertEqual([p.name for p in fn.params], ["val"])
        self.assertEqual(fn.namespace, "ckernel")
        rmw = [call for call in fn.calls if call.name == "TT_RMWCIB0"]
        self.assertEqual(len(rmw), 1)
        self.assertTrue(rmw[0].is_macro)
        self.assertEqual(rmw[0].args[2], "CfgAddr32")
        self.assertEqual(rmw[0].condition, "mask_b0 != 0")
        self.assertEqual(
            fn.body_line + fn.masked_body.count("\n", 0, rmw[0].offset), rmw[0].line
        )

    def test_class_methods_and_out_of_class_definitions(self) -> None:
        (setter,) = _by(self.index, "set", owner="addr_mod_t")
        self.assertEqual(setter.qualified_name, "ckernel::addr_mod_t::set")
        self.assertEqual([p.name for p in setter.params], ["mod_index"])
        self.assertIn("always_inline", setter.attributes)
        (program,) = _by(self.index, "program", owner="ckernel_template")
        self.assertEqual(program.qualified_name, "ckernel::ckernel_template::program")
        constructors = _by(self.index, "ckernel_template", is_ctor=True)
        self.assertGreaterEqual(len(constructors), 2)
        self.assertEqual(len({fn.key for fn in constructors}), len(constructors))

    def test_compute_api_thread_wrappers(self) -> None:
        (add,) = [
            fn for fn in self.index.by_name("add_tiles") if fn.namespace == "ckernel"
        ]
        threads = {call.name: call.thread for call in add.calls}
        self.assertEqual(threads.get("llk_unpack_AB"), "unpack")
        self.assertEqual(threads.get("llk_math_eltwise_binary"), "math")
        self.assertIsNone(add.thread_hint)
        self.assertIn("ALWI", add.attributes)

    def test_function_like_macros_are_expanded(self) -> None:
        (exp,) = [
            fn for fn in self.index.by_name("exp_tile") if fn.namespace == "ckernel"
        ]
        params_call = [
            call
            for call in exp.calls
            if call.name == "_llk_math_eltwise_unary_sfpu_params_"
        ]
        self.assertTrue(params_call, [call.name for call in exp.calls])
        call = params_call[0]
        self.assertEqual(call.via_macro, "SFPU_UNARY_CALL")
        self.assertEqual(call.thread, "math")
        self.assertTrue(
            call.args[0].startswith("::ckernel::sfpu::calculate_exponential<"),
            call.args[0],
        )
        self.assertNotIn("_SFPU_EXPAND", call.args[0])

    def test_keys_are_unique_and_order_is_stable(self) -> None:
        keys = [fn.key for fn in self.index.functions]
        self.assertEqual(len(keys), len(set(keys)))
        again = _scan_real(
            f"{BH_COMMON}/ckernel.h",
            f"{BH_COMMON}/ckernel_addrmod.h",
            f"{BH_COMMON}/ckernel_template.h",
            f"{COMPUTE}/eltwise_binary.h",
            f"{COMPUTE}/eltwise_unary/exp.h",
            SFPU_MACROS,
        )
        self.assertEqual(keys, [fn.key for fn in again.functions])
        self.assertEqual(self.index.fingerprint, again.fingerprint)
        self.assertEqual(
            [[c.to_json() for c in fn.calls] for fn in self.index.functions],
            [[c.to_json() for c in fn.calls] for fn in again.functions],
        )

    def test_full_tree_counts_every_layer(self) -> None:
        index = scan.scan_tree(REPO_ROOT)
        counts = index.layer_counts()
        for layer in scan.LAYERS:
            self.assertGreater(counts[layer], 10, (layer, counts))
        names = {fn.name for fn in index.functions}
        for expected in (
            "_llk_unpack_A_",
            "llk_math_eltwise_binary",
            "pack_tile",
            "_init_sfpu_config_reg",
        ):
            self.assertIn(expected, names)


class PreprocessorTest(unittest.TestCase):
    def test_arch_branches_and_thread_regions(self) -> None:
        text = "\n".join(
            [
                "#ifdef ARCH_QUASAR",
                "inline void quasar_only() { }",
                "#else",
                "inline void blackhole_path() { }",
                "#endif",
                "#if defined(ARCH_WORMHOLE) || defined(ARCH_BLACKHOLE)",
                "inline void wh_or_bh() { }",
                "#endif",
                "#ifdef TRISC_MATH",
                "inline void math_side() { }",
                "#endif",
                "#ifdef PERF_DUMP",
                "inline void perf() { }",
                "#endif",
                "#if __has_builtin(__builtin_rvtt_cfg_store)",
                "inline void intrinsic() { }",
                "#else",
                "inline void legacy() { }",
                "#endif",
            ]
        )
        source = scan.scan_text(
            text, "tt_metal/tt-llk/tt_llk_blackhole/common/inc/x.h", "common"
        )
        found = {fn.name: fn for fn in source.functions}
        self.assertEqual(
            sorted(found),
            ["blackhole_path", "intrinsic", "math_side", "perf", "wh_or_bh"],
        )
        self.assertEqual(found["math_side"].thread_hint, "math")
        self.assertIsNone(found["math_side"].pp_condition)
        self.assertEqual(found["perf"].pp_condition, "defined(PERF_DUMP)")

    def test_tristate_evaluation(self) -> None:
        self.assertIs(scan.truth("::SfpuType::exp == SfpuType::abs"), False)
        self.assertIs(scan.truth("(a::X == a::X) && !false"), True)
        self.assertIsNone(scan.truth("x == 1"))
        self.assertEqual(scan.evaluate("(1 << 4) | 0x3"), 19)
        self.assertEqual(scan.evaluate("true ? 3 : 4"), 3)
        self.assertIs(
            scan.truth("defined(ARCH_BLACKHOLE) && !defined(ARCH_QUASAR)"), True
        )


class HeaderAndCallSiteTest(unittest.TestCase):
    def test_attribute_macro_specialization_and_main(self) -> None:
        text = """
namespace ckernel {
template <bool A, bool B>
DEPRECATED_API("use g") ALWI void old_api(const std::uint32_t x) { g(x); }
template <>
inline void spec<true>(int v) { TTI_SETC16(FOO_ADDR32, v); }
template <bool APPROX>
struct Impl<APPROX, Kind::A> {
    static inline void apply(int& v) { v = 1; }
};
}
void MAIN {
    add_tiles(cb0, cb1, 0, 0, 0);
}
"""
        source = scan.scan_text(text, "k.cpp", "kernel")
        names = [fn.name for fn in source.functions]
        self.assertEqual(names, ["old_api", "spec", "apply", "MAIN"])
        old_api, spec, apply, main = source.functions
        self.assertEqual([p.name for p in old_api.template_params], ["A", "B"])
        self.assertEqual(spec.specialization_args, ("true",))
        self.assertEqual(apply.owner, "Impl")
        self.assertEqual(main.params, ())

    def test_member_ctor_and_template_calls(self) -> None:
        body = (
            "ckernel_template tmp(1, 2, TT_OP_X(0));\n"
            "tmp.program();\n"
            "addr_mod_t {.srca = {.incr = 1}}.set(ADDR_MOD_3);\n"
            "if (a < b) { foo<Kind::X, 2>(a, (b, c)); }\n"
            "std::uint32_t local(5);\n"
        )
        calls, _ = scan.find_calls(
            body, base_line=10, class_names={"ckernel_template", "addr_mod_t"}
        )
        by_name = {call.name: call for call in calls}
        self.assertTrue(by_name["ckernel_template"].is_ctor)
        self.assertEqual(by_name["program"].receiver_type, "ckernel_template")
        self.assertEqual(by_name["set"].receiver_type, "addr_mod_t")
        self.assertEqual(by_name["set"].args, ("ADDR_MOD_3",))
        self.assertEqual(by_name["foo"].template_args, ("Kind::X", "2"))
        self.assertEqual(by_name["foo"].args, ("a", "(b, c)"))
        self.assertEqual(by_name["foo"].condition, "a < b")
        self.assertEqual(by_name["foo"].line, 13)
        self.assertNotIn("local", by_name)

    def test_params_defaults_and_variadics(self) -> None:
        params = scan.parse_params(
            "const std::uint32_t x, bool acc = false, ReluConfig r = ReluConfig::none(), Args&&... args",
            "runtime",
        )
        self.assertEqual([p.name for p in params], ["x", "acc", "r", "args"])
        self.assertEqual(params[1].default, "false")
        self.assertEqual(params[2].default, "ReluConfig::none()")
        self.assertTrue(params[3].variadic)
        tparams = scan.parse_params(
            "typename Callable, typename... Args, bool fp32 = DST_ACCUM_MODE",
            "template",
        )
        self.assertEqual([p.name for p in tparams], ["Callable", "Args", "fp32"])
        self.assertEqual(scan.parse_params("std::uint32_t", "runtime")[0].name, "")


if __name__ == "__main__":
    unittest.main()
