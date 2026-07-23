# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the independent candidate-classification gate (Task 3).

These prove that the gate classifies every recognizable persistent-state token
independently of effect extraction, fails closed with architecture/path/line/
token for an unreviewed instruction candidate, and maps the known primitives
(raw RMWCIB, MOP CFG MMIO, mutex, dvalid/source-clear, software shadows) into
the reviewed categories on architecture fixtures and the real source tree.
"""

import tempfile
import unittest
from pathlib import Path

from tools.llk_state_audit import AuditModelError, audit
from tools.llk_state_audit.candidates import (
    CATEGORIES,
    classify_candidates,
    enforce_candidates,
    load_candidate_model,
)
from tools.llk_state_audit.inventory import scan_functions, scan_helpers

REPO_ROOT = Path(__file__).resolve().parents[3]


def write(root: Path, architecture: str, relative: str, source: str) -> Path:
    path = root / f"tt_llk_{architecture}" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def candidates_for(
    source: str, architecture: str = "wormhole_b0", relative: str = "llk_lib/llk_x.h"
):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        write(root, architecture, relative, source)
        records, model = scan_functions(root)
        helpers = scan_helpers(root, _model=model)
        return classify_candidates(records, helpers, load_candidate_model())


class CandidateGateUnitTest(unittest.TestCase):
    def test_unknown_instruction_candidate_fails_with_source_context(self) -> None:
        source = """\
inline void _llk_math_uses_unknown_()
{
    TTI_TOTALLYMADEUPINSN(1, 2, 3);
}
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = write(root, "blackhole", "llk_lib/llk_math.h", source)
            records, model = scan_functions(root)
            helpers = scan_helpers(root, _model=model)
            with self.assertRaises(AuditModelError) as caught:
                enforce_candidates(records, helpers, load_candidate_model())
        message = str(caught.exception)
        self.assertIn("blackhole", message)
        self.assertIn(path.relative_to(root).as_posix(), message)
        self.assertIn(":3", message)
        self.assertIn("TTI_TOTALLYMADEUPINSN", message)

    def test_known_instruction_candidates_map_to_reviewed_categories(self) -> None:
        source = """\
inline void _llk_math_known_()
{
    TTI_WRCFG(0, 0, REG);
    TTI_RMWCIB3(ADDR, mask, data);
    TT_OP_MVMUL(0);
    TTI_MVMUL(0);
    TTI_NOP;
    TTI_ATGETM(index);
    cfg[REG_A] = value;
    mop_cfg[0] = length;
}
"""
        candidates = candidates_for(
            source, architecture="wormhole_b0", relative="llk_lib/llk_math.h"
        )
        by_token = {
            (candidate["token"], candidate["kind"]): candidate["category"]
            for candidate in candidates
        }
        self.assertEqual(by_token[("TTI_WRCFG", "instruction")], "persistent_immediate")
        self.assertEqual(
            by_token[("TTI_RMWCIB3", "instruction")], "persistent_immediate"
        )
        self.assertEqual(by_token[("TT_OP_MVMUL", "instruction")], "transient_datapath")
        self.assertEqual(by_token[("TTI_MVMUL", "instruction")], "transient_datapath")
        self.assertEqual(by_token[("TTI_NOP", "instruction")], "reviewed_non_state")
        self.assertEqual(
            by_token[("TTI_ATGETM", "instruction")], "persistent_immediate"
        )
        self.assertEqual(by_token[("cfg", "mmio")], "persistent_immediate")
        self.assertEqual(by_token[("mop_cfg", "mmio")], "programming")

    def test_tt_op_built_instruction_is_persistent_deferred(self) -> None:
        source = """\
inline void _llk_pack_deferred_()
{
    const auto insn = TT_OP_WRCFG(0, 0, REG);
}
"""
        candidates = candidates_for(source, relative="llk_lib/llk_pack.h")
        deferred = next(c for c in candidates if c["token"] == "TT_OP_WRCFG")
        self.assertEqual(deferred["category"], "persistent_deferred")

    def test_software_shadow_assignment_is_persistent_candidate(self) -> None:
        source = """\
inline void _llk_pack_shadow_(int value)
{
    configured_num_tiles = value;
    src_zero_flag_state = value;
}
"""
        candidates = candidates_for(source, relative="llk_lib/llk_pack.h")
        software = {
            c["base"]: c["category"] for c in candidates if c["kind"] == "software"
        }
        self.assertEqual(software.get("configured_num_tiles"), "persistent_immediate")
        self.assertEqual(software.get("src_zero_flag_state"), "persistent_immediate")

    def test_recognizable_unknown_software_state_fails_closed(self) -> None:
        source = """\
inline void _llk_pack_shadow_(int value)
{
    configured_new_mode = value;
}
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root, "blackhole", "llk_lib/llk_pack.h", source)
            records, model = scan_functions(root)
            helpers = scan_helpers(root, _model=model)
            with self.assertRaisesRegex(
                AuditModelError,
                r"configured_new_mode",
            ):
                enforce_candidates(records, helpers, load_candidate_model())

    def test_local_variables_and_parameters_do_not_become_software_candidates(
        self,
    ) -> None:
        source = """\
inline void _llk_pack_local_(int configured_parameter)
{
    int configured_local = configured_parameter;
    configured_local = 4;
    configured_parameter = configured_local;
}
"""
        candidates = candidates_for(source, relative="llk_lib/llk_pack.h")
        self.assertFalse(any(c["kind"] == "software" for c in candidates))

    def test_topk_replay_init_is_reviewed_software_state(self) -> None:
        source = """\
inline void _llk_math_topk_()
{
    topk_replay_init = -1;
}
"""
        candidates = candidates_for(
            source, relative="common/inc/sfpu/ckernel_sfpu_topk.h"
        )
        topk = next(c for c in candidates if c["base"] == "topk_replay_init")
        self.assertEqual(topk["category"], "persistent_immediate")

    def test_mmio_read_is_not_a_write_candidate(self) -> None:
        source = """\
inline void _llk_math_read_(int value)
{
    int x = cfg[REG_A];
    if (cfg[REG_B] == value) { }
}
"""
        candidates = candidates_for(source, relative="llk_lib/llk_math.h")
        self.assertFalse(any(c["kind"] == "mmio" for c in candidates))

    def test_compound_mmio_assignment_is_a_persistent_candidate(self) -> None:
        source = """\
inline void _llk_math_update_(int value)
{
    cfg[REG_A] |= value;
}
"""
        candidates = candidates_for(source, relative="llk_lib/llk_math.h")
        mmio = [candidate for candidate in candidates if candidate["kind"] == "mmio"]
        self.assertEqual(len(mmio), 1)
        self.assertEqual(mmio[0]["category"], "persistent_immediate")

    def test_generic_helper_call_is_not_a_candidate(self) -> None:
        source = """\
inline void _llk_math_generic_(int value)
{
    some_generic_helper(value);
    compute_offset(value);
}
"""
        candidates = candidates_for(source, relative="llk_lib/llk_math.h")
        self.assertEqual(candidates, [])

    def test_mailbox_helpers_are_independent_persistent_candidates(self) -> None:
        source = """\
inline void _llk_math_mailbox_(int value)
{
    mailbox_write(ThreadId::UnpackThreadId, value);
    value = mailbox_read(ThreadId::MathThreadId);
}
"""
        candidates = candidates_for(source, relative="llk_lib/llk_math.h")
        mailbox = {
            candidate["base"]: candidate["category"]
            for candidate in candidates
            if candidate["kind"] == "helper_call"
        }
        self.assertEqual(
            mailbox,
            {
                "mailbox_read": "persistent_immediate",
                "mailbox_write": "persistent_immediate",
            },
        )

    def test_cfg_read_and_mop_execution_are_not_programming_mutations(self) -> None:
        source = """\
inline void _llk_math_control_()
{
    TTI_RDCFG(0, 0, REG);
    TTI_MOP(0, 0);
    TTI_MOP_CFG(0, 0);
}
"""
        candidates = candidates_for(source, relative="llk_lib/llk_math.h")
        by_base = {candidate["base"]: candidate["category"] for candidate in candidates}
        self.assertEqual(by_base["RDCFG"], "reviewed_non_state")
        self.assertEqual(by_base["MOP"], "persistent_deferred")
        self.assertEqual(by_base["MOP_CFG"], "programming")

    def test_unknown_sfpu_opcode_is_not_hidden_by_transient_family_rule(self) -> None:
        source = """\
inline void _llk_math_unknown_sfpu_()
{
    TTI_SFPFUTURECONFIG(0, 1);
}
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root, "blackhole", "llk_lib/llk_math.h", source)
            records, model = scan_functions(root)
            helpers = scan_helpers(root, _model=model)
            with self.assertRaisesRegex(
                AuditModelError,
                "TTI_SFPFUTURECONFIG",
            ):
                enforce_candidates(records, helpers, load_candidate_model())

    def test_every_category_is_a_reviewed_value(self) -> None:
        source = """\
inline void _llk_math_mixed_()
{
    TTI_WRCFG(0, 0, REG);
    TTI_MVMUL(0);
    TTI_NOP;
    TTI_MOP(0, 0);
}
"""
        candidates = candidates_for(source, relative="llk_lib/llk_math.h")
        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertIn(candidate["category"], CATEGORIES)


class CandidateGateRealSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (REPO_ROOT / "tt_llk_wormhole_b0").is_dir():
            raise unittest.SkipTest("real LLK source tree not present")
        cls.records, cls.model = scan_functions(REPO_ROOT)
        cls.helpers = scan_helpers(REPO_ROOT, _model=cls.model)
        cls.candidates = classify_candidates(
            cls.records, cls.helpers, load_candidate_model()
        )

    def test_real_tree_has_no_unreviewed_candidate(self) -> None:
        unknown = [c for c in self.candidates if c["category"] is None]
        self.assertEqual(unknown, [], msg=unknown[:5])
        # enforcement must therefore succeed
        enforce_candidates(self.records, self.helpers, load_candidate_model())

    def test_real_tree_covers_every_category_per_architecture(self) -> None:
        for architecture in ("wormhole_b0", "blackhole", "quasar"):
            seen = {
                c["category"]
                for c in self.candidates
                if c["architecture"] == architecture
            }
            self.assertTrue(
                {
                    "persistent_immediate",
                    "transient_datapath",
                    "reviewed_non_state",
                }.issubset(seen),
                msg=(architecture, sorted(seen)),
            )

    def test_quasar_rmwcib3_is_classified_persistent(self) -> None:
        quasar_rmwcib = [
            c
            for c in self.candidates
            if c["architecture"] == "quasar" and c["base"] == "RMWCIB3"
        ]
        self.assertTrue(quasar_rmwcib)
        self.assertTrue(
            all(c["category"] == "persistent_immediate" for c in quasar_rmwcib)
        )

    def test_audit_runs_the_candidate_gate(self) -> None:
        # audit() must enforce the gate; a clean real tree returns candidate counts.
        data = audit(REPO_ROOT)
        self.assertIn("candidate_summary", data["summary"])
        totals = data["summary"]["candidate_summary"]["by_category"]
        self.assertEqual(sum(totals.values()), len(self.candidates))


if __name__ == "__main__":
    unittest.main()
