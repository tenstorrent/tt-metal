# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Blackhole resource naming (tools.llk_state_audit.bh.resources)."""

import unittest
from pathlib import Path

from tools.llk_state_audit.bh import resources

REPO_ROOT = Path(__file__).resolve().parents[5]


def names(result) -> list[str]:
    return [item.name for item in result]


@unittest.skipUnless(
    (REPO_ROOT / resources.CFG_DEFINES).is_file(), "cfg_defines.h not present"
)
class NormalizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tables = resources.ResourceTables(
            REPO_ROOT, {"SFPCONFIG_TARGET_LREG11": "11", "p_setadc::UNP_AB": "0b011"}
        )

    def norm(self, kind: str, expr: str, thread: str | None = None) -> list[str]:
        return names(self.tables.normalize(kind, expr, thread))

    def test_cfg_fields_words_and_bytes(self) -> None:
        self.assertEqual(
            self.norm("cfg_rmw", "ALU_ACC_CTRL_Fp32_enabled_RMW"),
            ["CFG.ALU_ACC_CTRL_Fp32_enabled"],
        )
        self.assertEqual(
            self.norm(
                "cfg_rmw",
                "STACC_RELU_ApplyRelu_ADDR32, STACC_RELU_ApplyRelu_SHAMT, STACC_RELU_ApplyRelu_MASK",
            ),
            ["CFG.STACC_RELU_ApplyRelu"],
        )
        word = self.tables.normalize(
            "cfg_addr32", "THCON_SEC0_REG2_Out_data_format_ADDR32"
        )[0]
        self.assertEqual(
            (word.name, word.granularity),
            ("CFG.THCON_SEC0_REG2_Out_data_format", "word"),
        )
        loop = self.tables.normalize(
            "cfg_addr32", "THCON_SEC0_REG2_Out_data_format_ADDR32 + i"
        )[0]
        self.assertEqual((loop.granularity, loop.detail), ("word_range", "+i"))
        self.assertEqual(
            self.tables.normalize("cfg_byte", "PCK_EDGE_OFFSET_SEC0_mask_ADDR32")[
                0
            ].granularity,
            "byte",
        )
        self.assertFalse(self.tables.normalize("cfg_rmw", "CfgAddr32")[0].resolved)

    def test_thread_config_and_addr_mod_slots(self) -> None:
        self.assertEqual(
            self.norm("thread_cfg", "DEST_TARGET_REG_CFG_MATH_Offset_ADDR32"),
            ["THDCFG.DEST_TARGET_REG_CFG_MATH_Offset"],
        )
        self.assertEqual(
            self.norm("thread_cfg", "addr_mod_dest_reg_addr[ADDR_MOD_3]"),
            ["ADDR_MOD.3"],
        )
        self.assertEqual(
            self.norm("thread_cfg", "addr_mod_pack_reg_addr[1]"), ["ADDR_MOD_PACK.1"]
        )
        self.assertEqual(self.norm("addr_mod", "ADDR_MOD_7"), ["ADDR_MOD.7"])
        self.assertEqual(self.norm("addr_mod", "mod_index"), ["ADDR_MOD.?mod_index"])

    def test_gprs(self) -> None:
        self.assertEqual(self.norm("gpr", "p_gpr_pack::TMP0"), ["GPR.pack.TMP0"])
        lo = self.tables.normalize("gpr", "LO_16(p_gpr_unpack::TMP0)")[0]
        self.assertEqual((lo.name, lo.detail), ("GPR.unpack.TMP0", "LO_16"))
        self.assertEqual(self.norm("gpr", "p_gpr::ZERO"), ["GPR.any.ZERO"])
        tile_size_a = self.tables.gpr.by_name["unpack"]["TILE_SIZE_A"]
        self.assertEqual(
            self.norm("gpr", f"{tile_size_a}", "unpack"), ["GPR.unpack.TILE_SIZE_A"]
        )

    def test_sfpu(self) -> None:
        self.assertEqual(self.norm("sfpconfig_dest", "0xF"), ["SFPU.CTRL"])
        self.assertEqual(self.norm("sfpconfig_dest", "4 + 1"), ["SFPU.LOADMACRO.SEQ1"])
        self.assertEqual(
            self.norm("sfpconfig_dest", "SFPCONFIG_TARGET_LREG11"), ["SFPU.LREG11"]
        )
        self.assertEqual(
            self.norm("sfpi_const", "sfpi::vConstFloatPrgm0"), ["SFPU.LREG12"]
        )
        self.assertEqual(self.norm("sfpi_const", "vConstIntPrgm2"), ["SFPU.LREG14"])
        self.assertEqual(self.norm("lreg", "p_sfpu::LREG3"), ["SFPU.LREG3"])

    def test_counters_semaphores_mailbox(self) -> None:
        self.assertEqual(
            self.norm("adc", "p_setadc::UNP_AB|XY"), ["ADC.UNP_A.XY", "ADC.UNP_B.XY"]
        )
        self.assertEqual(
            self.norm("adc", "p_setadc::PAC|p_setadc::SET_W"), ["ADC.PAC.W"]
        )
        self.assertEqual(self.norm("rwc", "p_setrwc::SET_AB"), ["RWC.SRCA", "RWC.SRCB"])
        self.assertEqual(
            self.norm("semaphore", "p_stall::SEMAPHORE_1"), ["SEM.MATH_PACK"]
        )
        self.assertEqual(
            self.norm("semaphore", "semaphore::t6_sem(semaphore::UNPACK_SYNC)"),
            ["SEM.UNPACK_SYNC"],
        )
        self.assertEqual(
            self.norm("mailbox", "ckernel::ThreadId::PackThreadId"), ["MAILBOX.pack"]
        )

    def test_overlaps_scope_and_groups(self) -> None:
        self.assertTrue(
            self.tables.overlaps(
                "CFG.STACC_RELU_ApplyRelu",
                "CFG.STACC_RELU_ReluThreshold",
                "word",
                "field",
            )
        )
        self.assertFalse(
            self.tables.overlaps(
                "CFG.STACC_RELU_ApplyRelu", "CFG.STACC_RELU_ReluThreshold"
            )
        )
        self.assertTrue(self.tables.overlaps("ADDR_MOD.?x", "ADDR_MOD.2"))
        self.assertFalse(self.tables.overlaps("ADDR_MOD.1", "ADDR_MOD.2"))
        self.assertTrue(resources.is_per_thread("ADDR_MOD.1"))
        self.assertEqual(resources.scope_of("SFPU.LREG12"), "shared")
        self.assertEqual(resources.scope_of("GPR.pack.TMP0"), "thread_named")
        self.assertTrue(resources.same_thread("math", "any"))
        self.assertIn("pack.relu", self.tables.groups_of("CFG.STACC_RELU_ApplyRelu"))
        self.assertIn("sfpu.constants", self.tables.groups_of("SFPU.LREG12"))

    def test_rmw_template_args_expand(self) -> None:
        self.assertEqual(
            resources.expand_rmw_args(["ALU_ACC_CTRL_Fp32_enabled_RMW", "x"]),
            (
                "ALU_ACC_CTRL_Fp32_enabled_ADDR32",
                "ALU_ACC_CTRL_Fp32_enabled_SHAMT",
                "ALU_ACC_CTRL_Fp32_enabled_MASK",
                "x",
            ),
        )
        classes = {resource.cls for resource in resources.taxonomy()}
        for expected in (
            "CFG",
            "THDCFG",
            "GPR",
            "ADDR_MOD",
            "MOP",
            "REPLAY",
            "ADC",
            "RWC",
            "SFPU",
            "SEM",
            "SW",
            "param",
        ):
            self.assertIn(expected, classes)


if __name__ == "__main__":
    unittest.main()
