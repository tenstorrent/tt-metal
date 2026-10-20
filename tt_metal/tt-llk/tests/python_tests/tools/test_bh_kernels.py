# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Blackhole compute-kernel scanner (``bh/kernels.py``).

The synthetic tree below is shared with ``test_bh_hazards.py``. It holds a tiny
compute API (one folded execute per hazard class), a few LLK primitives, and
kernels under the kernel roots: an SDPA-path kernel with a local helper header,
a fold-drop kernel, a clean kernel that must not fire, an SFPU loop kernel, a
dataflow kernel and a Quasar kernel (both must be skipped).
"""

import tempfile
import unittest
from pathlib import Path

from tools.llk_state_audit.bh import effects_db, kernels

CHECKER_FILES = {
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
#define THCON_SEC0_REG1_Exp_threshold_ADDR32 71
#define THCON_SEC0_REG1_Exp_threshold_SHAMT 0
#define THCON_SEC0_REG1_Exp_threshold_MASK 0xff
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
    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_demo.h": """
namespace ckernel {
inline void _llk_pack_reconfig_l1_acc_(const std::uint32_t enable)
{
    cfg_reg_rmw_tensix<THCON_SEC0_REG1_Pack_L1_Acc_RMW>(enable);
}
inline void _llk_pack_edge_mask_(const std::uint32_t mask)
{
    cfg_reg_rmw_tensix<PCK_EDGE_OFFSET_SEC0_mask_RMW>(mask);
}
inline void _llk_pack_demo_(const std::uint32_t tile)
{
    TTI_RDCFG(p_gpr_pack::TMP0, PCK_EDGE_OFFSET_SEC0_mask_ADDR32);
}
inline void _llk_math_fp32_cfg_(const std::uint32_t enable)
{
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(enable);
}
inline void _llk_unpack_fp32_cfg_(const std::uint32_t enable)
{
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(enable);
}
inline void _llk_pack_gpr_demo_(const std::uint32_t v)
{
    regfile[p_gpr_pack::TMP0] = v;
    TTI_WRCFG(p_gpr_pack::TMP0, 0, THCON_SEC0_REG1_Exp_threshold_ADDR32);
}
inline void _llk_pack_gpr_synced_(const std::uint32_t v)
{
    regfile[p_gpr_pack::TMP0] = v;
    sync_regfile_write(p_gpr_pack::TMP0);
    TTI_WRCFG(p_gpr_pack::TMP0, 0, THCON_SEC0_REG1_Exp_threshold_ADDR32);
}
}
""",
    "tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_demo.h": """
namespace ckernel::sfpu {
inline void sfpu_a_init_() { sfpi::vConstFloatPrgm0 = 2.0f; }
inline void sfpu_b_init_() { sfpi::vConstFloatPrgm0 = 1.5f; }
}
""",
    "tt_metal/hw/inc/api/compute/demo.h": """
#pragma once
namespace ckernel {
ALWI void compute_kernel_hw_startup(std::uint32_t icb, std::uint32_t ocb) { PACK((_llk_pack_edge_mask_(0xffff))); }
ALWI void pack_reconfig_l1_acc(std::uint32_t enable) { PACK((_llk_pack_reconfig_l1_acc_(enable))); }
ALWI void demo_pack_tile(std::uint32_t idst, std::uint32_t ocb, bool l1_acc = false) {
    pack_reconfig_l1_acc(l1_acc ? 1 : 0);
    PACK((_llk_pack_demo_(idst)));
}
ALWI void plain_pack(std::uint32_t idst) { PACK((_llk_pack_demo_(idst))); }
ALWI void reduce_demo_init(std::uint32_t icb) {
    PACK((_llk_pack_edge_mask_(0xf)));
    MATH((_llk_math_fp32_cfg_(1)));
}
ALWI void reduce_demo(std::uint32_t icb, std::uint32_t idst) { reduce_demo_init(icb); }
ALWI void unpack_fp32_init(std::uint32_t icb) { UNPACK((_llk_unpack_fp32_cfg_(0))); }
ALWI void unpack_fp32_tile(std::uint32_t icb) { unpack_fp32_init(icb); }
ALWI void sfpu_a_tile_init() { MATH((ckernel::sfpu::sfpu_a_init_())); }
ALWI void sfpu_a_tile(std::uint32_t idst) { sfpu_a_tile_init(); }
ALWI void sfpu_b_tile_init() { MATH((ckernel::sfpu::sfpu_b_init_())); }
ALWI void sfpu_b_tile(std::uint32_t idst) { sfpu_b_tile_init(); }
ALWI void gpr_threshold_config(std::uint32_t v) { PACK((_llk_pack_gpr_demo_(v))); }
ALWI void gpr_threshold_config_synced(std::uint32_t v) { PACK((_llk_pack_gpr_synced_(v))); }
}
""",
    # --- kernels -------------------------------------------------------------- #
    "ttnn/cpp/ttnn/operations/transformer/sdpa_demo/device/kernels/compute/compute_common.hpp": """
#pragma once
#include "api/compute/demo.h"
template <bool do_pack>
void reduce_block(std::uint32_t cb, std::uint32_t n) {
    for (std::uint32_t j = 0; j < n; ++j) {
        reduce_demo(cb, j);
        if constexpr (do_pack) {
            plain_pack(j);
        } else {
            sfpu_a_tile(j);
        }
        unpack_fp32_tile(cb);
    }
}
""",
    "ttnn/cpp/ttnn/operations/transformer/sdpa_demo/device/kernels/compute/sdpa_demo.cpp": """
#include "api/compute/demo.h"
#include "compute_common.hpp"
void kernel_main() {
    constexpr std::uint32_t cb_in = 0;
    constexpr std::uint32_t cb_out = 16;
    compute_kernel_hw_startup(cb_in, cb_out);
    for (std::uint32_t i = 0; i < 4; ++i) {
        reduce_block<true>(cb_in, i);
    }
}
""",
    "ttnn/cpp/ttnn/operations/eltwise/demo/device/kernels/compute/l1_acc_kernel.cpp": """
#include "api/compute/demo.h"
void kernel_main() {
    compute_kernel_hw_startup(0, 16);
#ifdef PACKER_L1_ACC
    pack_reconfig_l1_acc(1);
#endif
    for (std::uint32_t i = 0; i < 8; ++i) {
        demo_pack_tile(i, 16);
    }
    pack_reconfig_l1_acc(0);
    gpr_threshold_config(3);
}
""",
    "ttnn/cpp/ttnn/operations/eltwise/demo/device/kernels/compute/ok_kernel.cpp": """
#include "api/compute/demo.h"
void kernel_main() {
    compute_kernel_hw_startup(0, 16);
    gpr_threshold_config_synced(3);
    pack_reconfig_l1_acc(1);
    for (std::uint32_t i = 0; i < 8; ++i) {
        demo_pack_tile(i, 16, true);
    }
    pack_reconfig_l1_acc(0);
    for (std::uint32_t i = 0; i < 8; ++i) {
        demo_pack_tile(i, 16);
    }
    for (std::uint32_t i = 0; i < 8; ++i) {
        sfpu_a_tile(0);
    }
    for (std::uint32_t i = 0; i < 8; ++i) {
        sfpu_b_tile(0);
    }
    for (std::uint32_t i = 0; i < 8; ++i) {
        plain_pack(i);
    }
}
""",
    "models/demos/attention_demo/kernels/sfpu_loop.cpp": """
#include "api/compute/demo.h"
void kernel_main() {
    std::uint32_t n = get_arg_val<std::uint32_t>(0);
    while (n > 0) {
        sfpu_a_tile(0);
        sfpu_b_tile(0);
        n--;
    }
}
""",
    # same SFPU pair outside the attention kernels: same-thread thrash is not reported there
    "models/demos/demo/kernels/sfpu_loop_plain.cpp": """
#include "api/compute/demo.h"
void kernel_main() {
    for (std::uint32_t i = 0; i < 4; ++i) {
        sfpu_a_tile(0);
        sfpu_b_tile(0);
    }
}
""",
    "ttnn/cpp/ttnn/operations/eltwise/demo/device/kernels/dataflow/reader.cpp": """
#include "api/dataflow/dataflow_api.h"
void kernel_main() { noc_async_read_barrier(); }
""",
    "ttnn/cpp/ttnn/operations/experimental/quasar/demo/device/kernels/compute/qsr.cpp": """
#include "api/compute/demo.h"
void kernel_main() { plain_pack(0); }
""",
}

SDPA = "ttnn/cpp/ttnn/operations/transformer/sdpa_demo/device/kernels/compute/sdpa_demo.cpp"
HELPER = "ttnn/cpp/ttnn/operations/transformer/sdpa_demo/device/kernels/compute/compute_common.hpp"
L1_ACC = (
    "ttnn/cpp/ttnn/operations/eltwise/demo/device/kernels/compute/l1_acc_kernel.cpp"
)
OK = "ttnn/cpp/ttnn/operations/eltwise/demo/device/kernels/compute/ok_kernel.cpp"
SFPU_LOOP = "models/demos/attention_demo/kernels/sfpu_loop.cpp"
SFPU_PLAIN = "models/demos/demo/kernels/sfpu_loop_plain.cpp"


def write_checker_tree(root: Path) -> Path:
    for relative, text in CHECKER_FILES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


class KernelScanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = write_checker_tree(Path(cls._tmp.name))
        cls.db = effects_db.build(cls.root, families=[])
        cls.kernels = {k.path: k for k in kernels.extract(cls.db, cls.root)}

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def names(self, path: str) -> list[str]:
        return [call.name for call in self.kernels[path].calls]

    def test_discovery_roots_and_filters(self) -> None:
        found = kernels.discover(self.root)
        self.assertIn(SDPA, found)
        self.assertIn(HELPER, found)
        self.assertIn(SFPU_LOOP, found)
        self.assertFalse([p for p in found if "/quasar/" in p])
        self.assertTrue(
            kernels.in_kernel_roots("ttnn/cpp/ttnn/kernel_lib/tilize_helpers.inl")
        )
        self.assertTrue(
            kernels.in_kernel_roots(
                "tests/tt_metal/tt_metal/test_kernels/compute/x.cpp"
            )
        )
        self.assertFalse(
            kernels.in_kernel_roots("ttnn/cpp/ttnn/operations/eltwise/device/op.cpp")
        )
        # compute kernels only: the dataflow reader and the Quasar kernel are not analysed
        self.assertEqual(
            sorted(self.kernels), sorted([L1_ACC, OK, SFPU_LOOP, SFPU_PLAIN, SDPA])
        )

    def test_helpers_are_inlined_with_substituted_arguments(self) -> None:
        kernel = self.kernels[SDPA]
        self.assertEqual(kernel.sources, (HELPER, SDPA))
        self.assertTrue(kernel.attention)
        self.assertEqual(
            self.names(SDPA),
            [
                "compute_kernel_hw_startup",
                "reduce_demo",
                "plain_pack",
                "unpack_fp32_tile",
            ],
        )
        reduce = kernel.calls[1]
        self.assertEqual(reduce.args, ("0", "j"))  # cb -> cb_in -> constexpr 0
        self.assertEqual(reduce.raw_args, ("cb", "j"))
        self.assertEqual([step[0] for step in reduce.stack], ["reduce_block"])
        self.assertEqual(reduce.stack[0][1], f"{SDPA}:9")
        self.assertTrue(reduce.in_attention)
        self.assertEqual(reduce.role, "execute")
        self.assertEqual(reduce.layer, "compute_api")

    def test_loop_nesting_and_constexpr_folding(self) -> None:
        kernel = self.kernels[SDPA]
        reduce, pack = kernel.calls[1], kernel.calls[2]
        self.assertEqual(
            [loop.id for loop in reduce.loops], [f"{SDPA}:8", f"{HELPER}:6"]
        )
        self.assertEqual([loop.kind for loop in reduce.loops], ["for", "for"])
        self.assertEqual(reduce.loops[1].instance, f"{SDPA}:9>{HELPER}:6")
        # if constexpr (do_pack) with do_pack=true: the then-branch survives without a condition
        self.assertEqual(pack.name, "plain_pack")
        self.assertEqual(pack.condition, "")
        self.assertEqual(kernels.same_loop(reduce, pack), f"{SDPA}:9>{HELPER}:6")
        self.assertNotIn("sfpu_a_tile", self.names(SDPA))
        startup = kernel.calls[0]
        self.assertEqual(startup.loops, ())
        self.assertIsNone(kernels.same_loop(startup, reduce))

    def test_ifdef_context_threads_and_config(self) -> None:
        kernel = self.kernels[L1_ACC]
        setter = kernel.calls[1]
        self.assertEqual(setter.name, "pack_reconfig_l1_acc")
        self.assertEqual(setter.ifdefs, ("defined(PACKER_L1_ACC)",))
        self.assertEqual(setter.condition, "")
        self.assertEqual(setter.role, "reconfig")
        self.assertIn("PACKER_L1_ACC", kernel.defines)
        self.assertEqual(kernel.calls[-1].ifdefs, ())
        config = dict(self.kernels[SDPA].config)
        self.assertEqual(config["local cb_out"], "16")
        loop_kernel = self.kernels[SFPU_LOOP]
        self.assertEqual([loop.kind for loop in loop_kernel.calls[-1].loops], ["while"])
        self.assertTrue(loop_kernel.attention)
        self.assertFalse(self.kernels[SFPU_PLAIN].attention)

    def test_loop_spans_and_conditions(self) -> None:
        text = "for (int i = 0; i < 3; ++i) { a(); } while (x) b(); do { c(); } while (y); "
        spans = kernels.loop_spans(text)
        self.assertEqual([s[2] for s in spans], ["for", "while", "do"])
        self.assertTrue(kernels.exclusive_conditions("A && x > 0", "!(x > 0)"))
        self.assertTrue(kernels.exclusive_conditions("defined(D)", "!defined(D) && z"))
        self.assertTrue(kernels.exclusive_conditions("(a || b)", "!(a || b)"))
        self.assertFalse(kernels.exclusive_conditions("a", "b"))
        self.assertFalse(kernels.exclusive_conditions("", "!(a)"))

    def test_extraction_is_deterministic(self) -> None:
        again = kernels.extract(self.db, self.root)
        self.assertEqual(
            [k.to_json() for k in again],
            [self.kernels[p].to_json() for p in sorted(self.kernels)],
        )
        self.assertEqual(
            kernels.fingerprint(again), kernels.fingerprint(list(self.kernels.values()))
        )


if __name__ == "__main__":
    unittest.main()
