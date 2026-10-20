# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Blackhole hazard rules (``bh/hazards.py``) on the synthetic tree of
``test_bh_kernels.py``: a synthetic conflict (dependent reader and cross-thread
writers in an SDPA-path kernel), a synthetic fold drop, an SFPU loop pair, and a
clean kernel that must not fire."""

import json
import tempfile
import unittest
from pathlib import Path

from test_bh_kernels import L1_ACC, OK, SDPA, SFPU_LOOP, SFPU_PLAIN, write_checker_tree

from tools.llk_state_audit.bh import effects_db, hazards
from tools.llk_state_audit.bh.model import Effect


def _effect(**kw) -> Effect:
    base = dict(
        resource="CFG.X",
        access="write",
        value_kind="const",
        value="1",
        condition="",
        evidence="a.h:1",
        thread="math",
    )
    base.update(kw)
    return Effect(**base)


class HazardRulesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = write_checker_tree(Path(cls._tmp.name))
        cls.db = effects_db.build(cls.root, families=[])
        cls.result = hazards.run(cls.db, cls.root)
        cls.findings = effects_db.normalize_findings(cls.result["findings"])

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def find(self, kind: str, resource: str) -> dict:
        matches = [
            f for f in self.findings if f["kind"] == kind and f["resource"] == resource
        ]
        self.assertEqual(
            len(matches), 1, [(f["kind"], f["resource"]) for f in self.findings]
        )
        return matches[0]

    def test_dependent_reader_conflict_in_attention_kernel_ranks_first(self) -> None:
        finding = self.find("resource_conflict", "CFG.PCK_EDGE_OFFSET_SEC0_mask")
        self.assertEqual(finding["rank"], 1)
        self.assertEqual(finding["id"], "H0001")
        self.assertEqual(finding["kernels"], [SDPA])
        self.assertEqual(finding["kernel_count"], 1)
        self.assertEqual(finding["ops"][:2], ["reduce_demo", "plain_pack"])
        self.assertEqual(finding["confidence"], "high")
        # evidence: the folded write, the read, and the kernel/helper call sites
        llk = "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_demo.h"
        self.assertIn(f"{llk}:9", finding["evidence"])  # _llk_pack_edge_mask_ write
        self.assertIn(f"{llk}:13", finding["evidence"])  # _llk_pack_demo_ RDCFG read
        self.assertTrue(
            any(e.startswith(SDPA.rsplit("/", 1)[0]) for e in finding["evidence"])
        )

    def test_cross_thread_conflict(self) -> None:
        finding = self.find("resource_conflict", "CFG.ALU_ACC_CTRL_Fp32_enabled")
        self.assertEqual(finding["ops"][:2], ["reduce_demo", "unpack_fp32_tile"])
        self.assertIn("unpack thread", finding["summary"])
        self.assertEqual(finding["kernels"], [SDPA])

    def test_fold_drop(self) -> None:
        finding = self.find("fold_drop", "param:demo_pack_tile.l1_acc")
        self.assertEqual(finding["kernels"], [L1_ACC])
        self.assertEqual(finding["ops"], ["demo_pack_tile", "pack_reconfig_l1_acc"])
        self.assertIn("PACKER_L1_ACC", finding["summary"])
        self.assertIn(f"{L1_ACC}:6", finding["evidence"])  # the kernel's setter
        self.assertIn(f"{L1_ACC}:9", finding["evidence"])  # the folded execute

    def test_ordering_gpr_store_without_sync(self) -> None:
        # families=[]: the body-local fallback scan (regfile[] store, then WRCFG reading the GPR)
        finding = self.find("ordering", "GPR.pack.TMP0")
        self.assertEqual(finding["ops"], ["_llk_pack_gpr_demo_"])
        self.assertEqual(
            finding["kernels"], [L1_ACC]
        )  # the synced variant in OK does not fire
        llk = "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_demo.h"
        self.assertEqual(finding["evidence"], [f"{llk}:25", f"{llk}:26"])

    def test_sfpu_same_loop_pair_is_not_reported(self) -> None:
        # same-thread SFPU thrash: each folded execute re-runs its own init right before its SFPU
        # body, so the pair is correct at runtime (it only blocks hoisting) -- not a hazard,
        # even inside an attention kernel
        self.assertFalse([f for f in self.findings if f["resource"] == "SFPU.LREG12"])
        for finding in self.findings:
            self.assertNotIn(SFPU_LOOP, finding["kernels"])

    def test_clean_kernel_does_not_fire(self) -> None:
        # explicit l1_acc, a default matching the last setter, SFPU ops in different loops,
        # and an edge-mask reader with only the startup writer: no hazard. Same-thread SFPU
        # thrash outside the attention kernels is not reported either.
        for finding in self.findings:
            self.assertNotIn(OK, finding["kernels"])
            self.assertNotIn(SFPU_PLAIN, finding["kernels"])
        self.assertEqual(len(self.findings), 4)

    def test_output_is_deterministic_and_normalized(self) -> None:
        again = effects_db.normalize_findings(
            hazards.run(self.db, self.root)["findings"]
        )
        self.assertEqual(
            effects_db.dumps({"findings": again}),
            effects_db.dumps({"findings": self.findings}),
        )
        self.assertEqual(
            [f["rank"] for f in self.findings], list(range(1, len(self.findings) + 1))
        )
        for finding in self.findings:
            self.assertEqual(
                sorted(finding),
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
            self.assertFalse([e for e in finding["evidence"] if e.startswith("/")])
        self.assertRegex(self.result["tree_fingerprint"], r"^[0-9a-f]{64}$")

    def test_cli_writer_is_byte_identical(self) -> None:
        out_a = Path(self._tmp.name) / "a.json"
        out_b = Path(self._tmp.name) / "b.json"
        effects_db.write_hazards(self.root, out_a, families=[])
        effects_db.write_hazards(self.root, out_b, families=[])
        self.assertEqual(out_a.read_bytes(), out_b.read_bytes())
        data = json.loads(out_a.read_text())
        self.assertEqual(
            [f["id"] for f in data["findings"]], [f["id"] for f in self.findings]
        )

    def test_value_identity_and_transient_list(self) -> None:
        db = self.db
        self.assertFalse(
            hazards.values_differ(
                db, _effect(value="1"), _effect(value="0x1", evidence="b.h:2")
            )
        )
        self.assertTrue(
            hazards.values_differ(db, _effect(value="1"), _effect(value="2"))
        )
        # SFPCONFIG from LREG0: the immediate is not the value, so two such writes are unknown
        opaque = dict(resource="SFPU.LREG12", value="imm=0;mod=0", value_kind="expr")
        self.assertTrue(
            hazards.values_differ(
                db, _effect(**opaque), _effect(**opaque, evidence="b.h:3")
            )
        )
        self.assertFalse(
            hazards.values_differ(db, _effect(**opaque), _effect(**opaque))
        )
        self.assertTrue(
            hazards.values_differ(
                db,
                _effect(value="?x", value_kind="unknown"),
                _effect(value="?x", value_kind="unknown", evidence="c.h:1"),
            )
        )
        self.assertFalse(hazards.has_unbound("l1_acc ? 1 : 0"))
        self.assertTrue(hazards.has_unbound("?tile_idx"))
        for name in (
            "RWC.DEST",
            "ADC.UNP_A.XY",
            "MOP",
            "SEM.MATH_PACK",
            "CFG.THCON_SEC0_REG3_Base_address",
        ):
            self.assertTrue(hazards.is_transient(name), name)
        for name in (
            "CFG.PCK_EDGE_OFFSET_SEC0_mask",
            "SFPU.LREG12",
            "ADDR_MOD_PACK.0",
            "REPLAY",
        ):
            self.assertFalse(hazards.is_transient(name), name)


if __name__ == "__main__":
    unittest.main()
