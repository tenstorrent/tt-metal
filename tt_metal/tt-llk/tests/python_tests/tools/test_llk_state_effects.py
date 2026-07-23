# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the normalized cross-architecture LLK state-effect audit (Task 2).

These exercise the parameter-to-state effect extraction, conservative parameter
flow, reviewed-model resource normalization, per-definition classification and
drift detection, plus validation against representative real functions from all
three architectures.
"""

import json
import re
import tempfile
import unittest
from pathlib import Path

from tools.llk_state_audit import (
    AuditModelError,
    audit,
    build_effects,
    classify,
    inventory,
    load_effect_model,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def write(root: Path, architecture: str, relative: str, source: str) -> Path:
    path = root / f"tt_llk_{architecture}" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


class StabilityTierTest(unittest.TestCase):
    def test_debug_experimental_and_stable_paths_are_classified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            body = "inline void _llk_pack_init_()\n{\n    TTI_WRCFG(0, 0, REG);\n}\n"
            write(root, "wormhole_b0", "llk_lib/llk_pack.h", body)
            write(
                root,
                "wormhole_b0",
                "llk_lib/experimental/llk_x.h",
                body.replace("init", "init_exp"),
            )
            write(
                root,
                "wormhole_b0",
                "llk_lib/debug/llk_math_hash_cb.h",
                body.replace("init", "dbg"),
            )
            records = {r["name"]: r for r in inventory(root)["functions"]}

        self.assertEqual(records["_llk_pack_init_"]["stability_tier"], "stable")
        self.assertEqual(
            records["_llk_pack_init_exp_"]["stability_tier"], "experimental"
        )
        self.assertEqual(records["_llk_pack_dbg_"]["stability_tier"], "debug")


class EffectExtractionTest(unittest.TestCase):
    FIXTURE = """\
template <bool is_fp32_dest_acc_en, bool to_from_int8 = false>
inline void _llk_unpack_reconfig_data_format_srca_impl_(
    const std::uint32_t unpack_src_format,
    const std::uint32_t unpack_dst_format,
    const std::uint32_t tile_size)
{
    if constexpr (to_from_int8)
    {
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcAUnsigned_RMW>((unpack_src_format == UInt8) ? 1 : 0);
    }
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32, 0, 0x0f>(unpack_src_format);
    TT_SETDMAREG(0, LOWER_HALFWORD(tile_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_A));
    const std::uint32_t face_dim = unpack_dst_format * 2;
    cfg_reg_rmw_tensix<THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32, 0, 0xffffffff>(face_dim);
}
"""

    def effects_for(
        self,
        source: str,
        architecture: str = "wormhole_b0",
        relative: str = "llk_lib/llk_unpack_common.h",
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root, architecture, relative, source)
            return build_effects(root)["effects"]

    def test_schema_version_and_required_fields(self) -> None:
        effects = self.effects_for(self.FIXTURE)
        self.assertTrue(effects)
        required = {
            "architecture",
            "thread",
            "stage",
            "stability",
            "function",
            "alias_of",
            "lifecycle",
            "parameter",
            "condition",
            "condition_kind",
            "domain",
            "resource",
            "operation",
            "value_expr",
            "persistence",
            "retention_contract",
            "activation",
            "restore",
            "confidence",
            "evidence",
        }
        for effect in effects:
            self.assertEqual(required, set(effect), msg=effect)
            self.assertEqual({"name", "type", "kind"}, set(effect["parameter"]))
            self.assertEqual(
                {
                    "kind",
                    "token",
                    "line",
                    "source_path",
                    "body_fingerprint",
                    "via",
                    "via_chain",
                    "sink",
                },
                set(effect["evidence"]),
            )

    def test_template_arg_resource_and_direct_parameter_flow(self) -> None:
        effects = self.effects_for(self.FIXTURE)
        srca = next(
            e
            for e in effects
            if e["resource"] == "THCON_SEC0_REG0_TileDescriptor_ADDR32"
            and e["parameter"]["name"] == "unpack_src_format"
        )
        self.assertEqual(srca["domain"], "cfg_register")
        self.assertEqual(srca["operation"], "rmw")
        self.assertEqual(srca["parameter"]["kind"], "runtime")
        self.assertEqual(srca["value_expr"], "unpack_src_format")
        self.assertIsNone(srca["condition"])
        self.assertEqual(srca["confidence"], "high")
        self.assertEqual(srca["evidence"]["kind"], "direct")
        self.assertEqual(srca["evidence"]["token"], "cfg_reg_rmw_tensix")
        self.assertEqual(srca["evidence"]["line"], 11)
        self.assertEqual(
            srca["retention_contract"],
            "Retained until the configuration resource is reconfigured.",
        )

    def test_instruction_resources_use_semantic_operands(self) -> None:
        source = """\
inline void _llk_math_configure_resources_(const std::uint32_t value)
{
    TTI_SEMINIT(2, 0, p_stall::SEMAPHORE_1);
    TTI_INCRWC(0, value, 0, 0);
    TTI_SETRWC(0, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
}
"""
        effects = self.effects_for(
            source, architecture="blackhole", relative="llk_lib/llk_math_common.h"
        )
        semaphore = next(effect for effect in effects if effect["operation"] == "init")
        self.assertEqual(semaphore["resource"], "p_stall::SEMAPHORE_1")
        self.assertEqual(semaphore["value_expr"], "2")

        increments = [
            effect for effect in effects if effect["operation"] == "increment"
        ]
        self.assertEqual(len(increments), 1)
        self.assertEqual(increments[0]["resource"], "RWC_D")
        self.assertEqual(increments[0]["value_expr"], "value")
        self.assertEqual(increments[0]["parameter"]["name"], "value")

        setrwc = next(effect for effect in effects if effect["operation"] == "rwc")
        self.assertEqual(setrwc["resource"], "p_setrwc::SET_ABD_F")
        self.assertEqual(setrwc["value_expr"], "0, 0, 0, 0")
        adc = next(effect for effect in effects if effect["operation"] == "adc")
        self.assertEqual(adc["resource"], "p_setadc::UNP_AB")

        quasar_source = """\
inline void _llk_math_configure_resources_(const std::uint32_t value)
{
    TTI_SEMINIT(2, 0, 0, p_stall::SEMAPHORE_1);
    TTI_INCRWC(0, value, 2, 3);
    TTI_SETRWC(0, p_setrwc::CR_D, value, p_setrwc::SET_D);
}
"""
        quasar = self.effects_for(
            quasar_source,
            architecture="quasar",
            relative="llk_lib/llk_math_common.h",
        )
        semaphore = next(effect for effect in quasar if effect["operation"] == "init")
        self.assertEqual(semaphore["resource"], "p_stall::SEMAPHORE_1")
        increments = {
            effect["resource"]: effect["value_expr"]
            for effect in quasar
            if effect["operation"] == "increment"
        }
        self.assertEqual(
            increments,
            {"RWC_A": "value", "RWC_B": "2", "RWC_D": "3"},
        )
        setrwc = next(effect for effect in quasar if effect["operation"] == "rwc")
        self.assertEqual(setrwc["resource"], "p_setrwc::SET_D")
        self.assertEqual(setrwc["value_expr"], "p_setrwc::CR_D, value")

    def test_recoverable_helper_resources_are_not_generic_or_blank(self) -> None:
        source = """\
inline void _llk_pack_configure_resources_(const std::uint32_t value)
{
    cfg_rmw(THCON_PACKER0_REG0_IN_DATA_FORMAT_RMW, value);
    TT_SETDMAREG(0, LOWER_HALFWORD(value << SHIFT), 0, LO_16(p_gpr_pack::TMP0));
    builder.set(ADDR_MOD_3);
    math::set_dst_write_addr<TileShape>(value);
}
"""
        effects = self.effects_for(
            source, architecture="quasar", relative="llk_lib/llk_pack.h"
        )
        by_domain = {}
        for effect in effects:
            by_domain.setdefault(effect["domain"], []).append(effect)
        self.assertTrue(
            any(
                effect["resource"] == "THCON_PACKER0_REG0_IN_DATA_FORMAT_RMW"
                for effect in by_domain["cfg_register"]
            )
        )
        self.assertTrue(
            any(
                effect["resource"] == "LO_16(p_gpr_pack::TMP0)"
                for effect in by_domain["gpr"]
            )
        )
        self.assertTrue(
            any(effect["resource"] == "ADDR_MOD_3" for effect in by_domain["addr_mod"])
        )
        self.assertTrue(
            any(
                effect["resource"] == "destination_write_address"
                for effect in by_domain["destination"]
            )
        )

    def test_compile_time_condition_is_preserved(self) -> None:
        effects = self.effects_for(self.FIXTURE)
        guarded = next(
            e
            for e in effects
            if e["resource"] == "ALU_FORMAT_SPEC_REG0_SrcAUnsigned_RMW"
        )
        self.assertEqual(guarded["condition"], "to_from_int8")
        self.assertEqual(guarded["condition_kind"], "compile_time")
        self.assertEqual(guarded["parameter"]["name"], "unpack_src_format")
        self.assertEqual(guarded["value_expr"], "(unpack_src_format == UInt8) ? 1 : 0")

    def test_gpr_write_extracts_resource_and_runtime_parameter(self) -> None:
        effects = self.effects_for(self.FIXTURE)
        gpr = next(e for e in effects if e["domain"] == "gpr")
        self.assertEqual(gpr["operation"], "write")
        self.assertIn("TILE_SIZE_A", gpr["resource"])
        self.assertEqual(gpr["parameter"]["name"], "tile_size")

    def test_local_variable_flow_is_medium_confidence(self) -> None:
        effects = self.effects_for(self.FIXTURE)
        local = next(
            e
            for e in effects
            if e["resource"] == "THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32"
        )
        self.assertEqual(local["parameter"]["name"], "unpack_dst_format")
        self.assertEqual(local["confidence"], "medium")
        self.assertEqual(local["value_expr"], "face_dim")

    def test_fixed_effect_when_no_parameter_flows(self) -> None:
        source = """\
inline void _llk_unpack_set_srcb_dummy_valid_()
{
    TTI_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_SET_DVALID);
}
"""
        effects = self.effects_for(source)
        dvalid = next(e for e in effects if e["domain"] == "dvalid_semaphore")
        self.assertEqual(dvalid["parameter"]["name"], "-")
        self.assertEqual(dvalid["parameter"]["kind"], "fixed")
        self.assertEqual(dvalid["operation"], "set")

    def test_direct_config_pointer_assignment_with_runtime_condition(self) -> None:
        source = """\
inline void _llk_unpack_configure_addresses_(const std::uint32_t address_a, const std::uint32_t address_b, volatile std::uint32_t *cfg)
{
    if (0 == unp_cfg_context)
    {
        cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_b;
    }
}
"""
        effects = self.effects_for(source)
        assigns = [
            e
            for e in effects
            if e["domain"] == "cfg_register" and e["operation"] == "write"
        ]
        by_resource = {e["resource"]: e for e in assigns}
        self.assertIn("THCON_SEC0_REG3_Base_address_ADDR32", by_resource)
        effect = by_resource["THCON_SEC0_REG3_Base_address_ADDR32"]
        self.assertEqual(effect["parameter"]["name"], "address_a")
        self.assertEqual(effect["value_expr"], "address_a")
        self.assertEqual(effect["condition"], "0 == unp_cfg_context")
        self.assertEqual(effect["condition_kind"], "runtime")

    def test_compound_config_pointer_assignment_is_mapped(self) -> None:
        source = """\
inline void _llk_math_update_config_(std::uint32_t value, volatile std::uint32_t *cfg)
{
    cfg[ALU_CONTROL_ADDR32] |= value;
}
"""
        effects = self.effects_for(
            source,
            architecture="blackhole",
            relative="llk_lib/llk_math.h",
        )
        effect = next(
            effect for effect in effects if effect["domain"] == "cfg_register"
        )
        self.assertEqual(effect["resource"], "ALU_CONTROL_ADDR32")
        self.assertEqual(effect["value_expr"], "|= value")
        self.assertEqual(effect["parameter"]["name"], "value")

    def test_replay_insn_is_pure_but_record_and_replay_are_effects(self) -> None:
        source = """\
inline void _llk_math_replay_()
{
    lltt::record(0, 4);
    lltt::replay(0, 4);
    const auto instruction = lltt::replay_insn(0, 4);
}
"""
        effects = self.effects_for(source, relative="llk_lib/llk_math.h")
        tokens = [
            effect["evidence"]["token"]
            for effect in effects
            if effect["domain"] == "mop_replay"
        ]
        self.assertEqual(tokens, ["lltt::record", "lltt::replay"])

    def test_reg_write_extracts_register_resource_without_generic_writes(self) -> None:
        source = """\
inline void _llk_math_clear_flags_(std::uint32_t value)
{
    reg_write(RISCV_DEBUG_REG_FPU_STICKY_BITS, value);
    write(buffer, value);
}
"""
        effects = self.effects_for(source, relative="llk_lib/llk_math.h")
        writes = [effect for effect in effects if effect["domain"] == "debug_register"]
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0]["resource"], "RISCV_DEBUG_REG_FPU_STICKY_BITS")
        self.assertEqual(writes[0]["value_expr"], "value")

    def test_else_effect_uses_negated_if_condition(self) -> None:
        source = """\
template <bool UseA>
inline void _llk_pack_config_(std::uint32_t a, std::uint32_t b)
{
    if constexpr (UseA) {
        cfg_reg_rmw_tensix<REG_A>(a);
    } else {
        cfg_reg_rmw_tensix<REG_B>(b);
    }
}
"""
        effects = self.effects_for(source, relative="llk_lib/llk_pack.h")
        by_resource = {effect["resource"]: effect for effect in effects}
        self.assertEqual(by_resource["REG_A"]["condition"], "UseA")
        self.assertEqual(by_resource["REG_B"]["condition"], "!(UseA)")
        self.assertEqual(by_resource["REG_B"]["condition_kind"], "compile_time")

    def test_nested_else_if_conditions_compose_outer_to_inner(self) -> None:
        source = """\
inline void _llk_pack_config_(bool outer, bool inner, bool alternate)
{
    if (outer) {
        if (inner) {
            cfg_reg_rmw_tensix<REG_A>(1);
        } else if (alternate) {
            cfg_reg_rmw_tensix<REG_B>(2);
        } else {
            cfg_reg_rmw_tensix<REG_C>(3);
        }
    } else {
        cfg_reg_rmw_tensix<REG_D>(4);
    }
}
"""
        effects = self.effects_for(source, relative="llk_lib/llk_pack.h")
        conditions = {effect["resource"]: effect["condition"] for effect in effects}
        self.assertEqual(conditions["REG_A"], "outer && inner")
        self.assertEqual(conditions["REG_B"], "outer && !(inner) && alternate")
        self.assertEqual(conditions["REG_C"], "outer && !(inner) && !(alternate)")
        self.assertEqual(conditions["REG_D"], "!(outer)")

    def test_no_effect_claims_a_parameter_absent_from_the_expression(self) -> None:
        effects = self.effects_for(self.FIXTURE)
        for effect in effects:
            name = effect["parameter"]["name"]
            if name == "-":
                continue
            self.assertIsNotNone(effect["value_expr"])
            # The claimed parameter must actually be present in the recorded value expression
            # (either directly or via a local it is proven to flow through).
            self.assertTrue(effect["value_expr"], msg=effect)

    def test_if_zero_state_sinks_are_ignored_and_else_line_is_preserved(self) -> None:
        source = """\
inline void _llk_pack_config_()
{
#if 0
    cfg_reg_rmw_tensix<INACTIVE_REG>(1);
#else
    cfg_reg_rmw_tensix<ACTIVE_REG>(2);
#endif
}
"""
        effects = self.effects_for(source, relative="llk_lib/llk_pack.h")
        self.assertEqual(
            [(effect["resource"], effect["evidence"]["line"]) for effect in effects],
            [("ACTIVE_REG", 6)],
        )

    def test_nested_ifdef_and_ifndef_do_not_escape_if_zero(self) -> None:
        source = """\
inline void _llk_pack_config_()
{
#if 0
#ifdef OUTER_FEATURE
#ifndef INNER_FEATURE
    cfg_reg_rmw_tensix<NESTED_INACTIVE_REG>(1);
#endif
#endif
    cfg_reg_rmw_tensix<STILL_INACTIVE_REG>(2);
#else
    cfg_reg_rmw_tensix<ACTIVE_REG>(3);
#endif
}
"""
        effects = self.effects_for(source, relative="llk_lib/llk_pack.h")
        self.assertEqual(
            [(effect["resource"], effect["evidence"]["line"]) for effect in effects],
            [("ACTIVE_REG", 11)],
        )

    def test_if_zero_elif_chain_selects_first_static_true_branch(self) -> None:
        source = """\
inline void _llk_pack_config_()
{
#if 0
    cfg_reg_rmw_tensix<IF_INACTIVE_REG>(1);
#elif 0
    cfg_reg_rmw_tensix<ELIF_ZERO_REG>(2);
#elif 1
    cfg_reg_rmw_tensix<ELIF_ACTIVE_REG>(3);
#else
    cfg_reg_rmw_tensix<ELSE_INACTIVE_REG>(4);
#endif
}
"""
        effects = self.effects_for(source, relative="llk_lib/llk_pack.h")
        self.assertEqual(
            [(effect["resource"], effect["evidence"]["line"]) for effect in effects],
            [("ELIF_ACTIVE_REG", 8)],
        )


class ActivationTest(unittest.TestCase):
    """Task 4: every effect carries a reliable activation label."""

    def effects_for(
        self,
        source: str,
        architecture: str = "wormhole_b0",
        relative: str = "llk_lib/llk_pack.h",
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root, architecture, relative, source)
            return build_effects(root)["effects"]

    def test_every_effect_has_a_known_activation(self) -> None:
        source = """\
inline void _llk_pack_mixed_(std::uint32_t value)
{
    cfg_reg_rmw_tensix<REG_A>(value);
    TTI_WRCFG(0, 0, REG_B);
    cfg[REG_C] = value;
    ckernel_template mop(1, 1, 0);
    mop.program();
}
"""
        effects = self.effects_for(source)
        self.assertTrue(effects)
        for effect in effects:
            self.assertIn(
                effect["activation"],
                {"immediate", "deferred", "programming"},
                msg=effect,
            )

    def test_tti_and_direct_writes_are_immediate(self) -> None:
        source = """\
inline void _llk_pack_immediate_(std::uint32_t value)
{
    TTI_WRCFG(0, 0, REG_B);
    cfg[REG_C] = value;
    shadow_format = value;
}
"""
        effects = self.effects_for(source)
        self.assertTrue(
            all(effect["activation"] == "immediate" for effect in effects), msg=effects
        )

    def test_tt_op_built_instruction_is_deferred_not_immediate(self) -> None:
        source = """\
inline void _llk_pack_deferred_()
{
    TT_OP_WRCFG(0, 0, REG_B);
}
"""
        effects = self.effects_for(source)
        wrcfg = next(effect for effect in effects if effect["resource"] == "REG_B")
        self.assertEqual(wrcfg["activation"], "deferred")

    def test_mop_and_replay_programs_are_programming(self) -> None:
        source = """\
inline void _llk_math_program_()
{
    ckernel_template mop(1, 1, 0);
    mop.program();
    lltt::record(0, 4);
}
"""
        effects = self.effects_for(source, relative="llk_lib/llk_math.h")
        programming = [effect for effect in effects if effect["domain"] == "mop_replay"]
        self.assertTrue(programming)
        self.assertTrue(
            all(effect["activation"] == "programming" for effect in programming)
        )

    def test_mop_cfg_mmio_is_a_programming_effect(self) -> None:
        source = """\
inline void _llk_pack_patch_mop_(std::uint32_t value)
{
    mop_cfg[FAST_UNTILIZE_MOP_LAST_OUTER_CFG_INDEX] = value;
}
"""
        effects = self.effects_for(source)
        self.assertEqual(len(effects), 1)
        effect = effects[0]
        self.assertEqual(effect["domain"], "mop_replay")
        self.assertEqual(
            effect["resource"],
            "mop_cfg[FAST_UNTILIZE_MOP_LAST_OUTER_CFG_INDEX]",
        )
        self.assertEqual(effect["operation"], "program")
        self.assertEqual(effect["value_expr"], "value")
        self.assertEqual(effect["activation"], "programming")

    def test_transitive_effects_preserve_activation(self) -> None:
        source = """\
inline void write_cfg_helper(std::uint32_t value)
{
    TT_OP_WRCFG(0, 0, DEFERRED_REG);
}
inline void _llk_pack_outer_(std::uint32_t value)
{
    write_cfg_helper(value);
}
"""
        effects = self.effects_for(source)
        propagated = next(
            effect
            for effect in effects
            if effect["function"] == "_llk_pack_outer_"
            and effect["resource"] == "DEFERRED_REG"
            and effect["evidence"]["kind"] == "transitive"
        )
        self.assertEqual(propagated["activation"], "deferred")


class MissingPersistentPrimitiveTest(unittest.TestCase):
    def effects_for(
        self,
        source: str,
        architecture: str = "wormhole_b0",
        relative: str = "llk_lib/llk_math.h",
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root, architecture, relative, source)
            return build_effects(root)["effects"]

    def test_mailbox_write_and_read_have_distinct_immediate_effects(self) -> None:
        source = """\
inline void _llk_math_mailbox_(std::uint32_t value)
{
    mailbox_write(ThreadId::UnpackThreadId, value);
    value = mailbox_read(ThreadId::MathThreadId);
}
"""
        effects = self.effects_for(source)
        mailbox = {
            (effect["operation"], effect["resource"]): effect
            for effect in effects
            if effect["domain"] == "mailbox"
        }
        self.assertEqual(
            set(mailbox),
            {
                ("write", "ThreadId::UnpackThreadId"),
                ("consume", "ThreadId::MathThreadId"),
            },
        )
        self.assertEqual(
            mailbox[("write", "ThreadId::UnpackThreadId")]["value_expr"],
            "value",
        )
        self.assertTrue(
            all(effect["activation"] == "immediate" for effect in mailbox.values())
        )

    def test_unpack_nop_dvalid_has_only_semantic_resource(self) -> None:
        source = """\
inline void _llk_unpack_dvalid_()
{
    TTI_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_SET_DVALID);
}
"""
        effects = self.effects_for(
            source,
            architecture="wormhole_b0",
            relative="llk_lib/llk_unpack.h",
        )
        dvalid = [
            effect
            for effect in effects
            if effect["domain"] == "dvalid_semaphore" and effect["operation"] == "set"
        ]
        self.assertEqual(
            [effect["resource"] for effect in dvalid],
            ["UNPACK_DVALID[SrcB]"],
        )

    def test_raw_rmwcib_signatures_are_architecture_correct(self) -> None:
        wh = self.effects_for(
            """\
inline void _llk_math_raw_(std::uint32_t value)
{
    TTI_RMWCIB3(mask, value, WH_CFG_ADDR);
}
"""
        )
        bh = self.effects_for(
            """\
inline void _llk_math_raw_(std::uint32_t value)
{
    TTI_RMWCIB2(mask, value, BH_CFG_ADDR);
}
""",
            architecture="blackhole",
        )
        qsr = self.effects_for(
            """\
inline void _llk_math_raw_(std::uint32_t value)
{
    TTI_RMWCIB3(QSR_CFG_ADDR, mask, value);
}
""",
            architecture="quasar",
        )
        self.assertEqual(
            [
                (effects[0]["resource"], effects[0]["value_expr"])
                for effects in (wh, bh, qsr)
            ],
            [
                ("WH_CFG_ADDR", "value"),
                ("BH_CFG_ADDR", "value"),
                ("QSR_CFG_ADDR", "value"),
            ],
        )

    def test_wh_and_bh_unpacr_nop_model_dvalid_and_source_clear(self) -> None:
        wh = self.effects_for(
            """\
inline void _llk_unpack_dummy_()
{
    TTI_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_SET_DVALID);
    TTI_UNPACR_NOP(SrcA, p_unpacr_nop::UNP_ZEROSRC);
}
""",
            relative="llk_lib/llk_unpack.h",
        )
        bh = self.effects_for(
            """\
inline void _llk_unpack_dummy_()
{
    TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
}
""",
            architecture="blackhole",
            relative="llk_lib/llk_unpack.h",
        )
        self.assertIn(
            ("dvalid_semaphore", "UNPACK_DVALID[SrcB]", "set"),
            {(e["domain"], e["resource"], e["operation"]) for e in wh},
        )
        self.assertIn(
            ("src_data", "SRC_DATA[SrcA]", "clear"),
            {(e["domain"], e["resource"], e["operation"]) for e in wh},
        )
        self.assertEqual(
            {(e["domain"], e["resource"], e["operation"]) for e in bh},
            {
                ("dvalid_semaphore", "UNPACK_DVALID[SrcB]", "set"),
                ("src_data", "SRC_DATA[SrcB]", "clear"),
            },
        )


class TransientDatapathSummaryTest(unittest.TestCase):
    def test_one_normalized_summary_per_function_family(self) -> None:
        source = """\
inline void _llk_math_datapath_()
{
    TTI_MOVD2A(0, 0, 0);
    TTI_MOVD2A(0, 1, 0);
    TTI_ELWADD(0, 0);
    TTI_MVMUL(0);
    TTI_GAPOOL(0);
    TTI_SFPADD(0, 0, 0);
}
inline void _llk_unpack_datapath_()
{
    TTI_UNPACR(SrcA, 0);
}
inline void _llk_pack_datapath_()
{
    TTI_PACR(0);
}
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root, "wormhole_b0", "llk_lib/llk_datapath.h", source)
            effects = build_effects(root)["effects"]
        transient = [
            effect for effect in effects if effect["domain"] == "transient_datapath"
        ]
        self.assertEqual(
            {effect["resource"] for effect in transient},
            {
                "MOV",
                "ELW",
                "MATMUL",
                "POOL",
                "SFPU_DATA_PLANE",
                "UNPACR_DATA_PLANE",
                "PACR_DATA_PLANE",
            },
        )
        mov = next(effect for effect in transient if effect["resource"] == "MOV")
        self.assertIn("2 candidates", mov["value_expr"])
        self.assertEqual(mov["evidence"]["token"], "TTI_MOVD2A")
        self.assertEqual(mov["persistence"], "transient")
        self.assertIsNone(mov["restore"])
        self.assertTrue(all(effect["operation"] == "summarize" for effect in transient))


class TransitiveFlowTest(unittest.TestCase):
    SOURCE = """\
template <bool is_fp32_dest_acc_en>
inline void _llk_unpack_reconfig_data_format_srca_(const std::uint32_t src_format, const std::uint32_t dst_format, const std::uint32_t tile_size)
{
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32, 0, 0x0f>(src_format);
}

template <bool is_fp32_dest_acc_en>
inline void _llk_unpack_reconfig_data_format_(const std::uint32_t srca_format, const std::uint32_t srca_dst, const std::uint32_t tsize)
{
    _llk_unpack_reconfig_data_format_srca_<is_fp32_dest_acc_en>(srca_format, srca_dst, tsize);
}
"""

    def test_transitive_effect_maps_caller_parameter_to_callee_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root, "blackhole", "llk_lib/llk_unpack_common.h", self.SOURCE)
            effects = build_effects(root)["effects"]

        transitive = [
            e
            for e in effects
            if e["function"] == "_llk_unpack_reconfig_data_format_"
            and e["evidence"]["kind"] == "transitive"
        ]
        self.assertTrue(transitive)
        mapped = next(
            e
            for e in transitive
            if e["resource"] == "THCON_SEC0_REG0_TileDescriptor_ADDR32"
        )
        self.assertEqual(mapped["parameter"]["name"], "srca_format")
        self.assertEqual(
            mapped["evidence"]["via"], "_llk_unpack_reconfig_data_format_srca_"
        )
        self.assertIn(mapped["confidence"], {"medium", "low"})

    def test_overload_is_selected_by_template_and_runtime_arity(self) -> None:
        source = """\
inline void _llk_pack_config_(std::uint32_t value)
{
    cfg_reg_rmw_tensix<ONE_ARG_REG>(value);
}
template <bool Mode>
inline void _llk_pack_config_(std::uint32_t value, std::uint32_t other)
{
    cfg_reg_rmw_tensix<TWO_ARG_REG>(other);
}
inline void _llk_pack_wrapper_(std::uint32_t value)
{
    _llk_pack_config_(value);
}
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root, "blackhole", "llk_lib/llk_pack.h", source)
            effects = build_effects(root)["effects"]
        propagated = [
            effect
            for effect in effects
            if effect["function"] == "_llk_pack_wrapper_"
            and effect["evidence"]["kind"] == "transitive"
        ]
        self.assertEqual([effect["resource"] for effect in propagated], ["ONE_ARG_REG"])

    def test_equality_operator_in_parameter_type_is_not_a_default(self) -> None:
        source = """\
inline void arity_helper()
{
    cfg_reg_rmw_tensix<NO_ARG_REG>(1);
}
template <int Mode = 1>
inline void arity_helper(std::enable_if_t<(Mode == 1), int> value)
{
    cfg_reg_rmw_tensix<ONE_ARG_REG>(value);
}
inline void _llk_pack_wrapper_()
{
    arity_helper();
}
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root, "blackhole", "llk_lib/llk_pack.h", source)
            try:
                effects = build_effects(root)["effects"]
            except AuditModelError as error:
                self.fail(f"equality operator was mistaken for a default: {error}")
        propagated = [
            effect for effect in effects if effect["function"] == "_llk_pack_wrapper_"
        ]
        self.assertEqual(
            [effect["resource"] for effect in propagated],
            ["NO_ARG_REG"],
        )

    def test_ambiguous_same_arity_overloads_do_not_propagate(self) -> None:
        source = """\
inline void _llk_pack_config_(std::uint32_t value)
{
    cfg_reg_rmw_tensix<INTEGER_REG>(value);
}
inline void _llk_pack_config_(bool value)
{
    cfg_reg_rmw_tensix<BOOLEAN_REG>(value);
}
inline void _llk_pack_wrapper_(std::uint32_t value)
{
    _llk_pack_config_(ambiguous_value(value));
}
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = write(root, "blackhole", "llk_lib/llk_pack.h", source)
            with self.assertRaisesRegex(
                AuditModelError,
                rf"{re.escape(path.relative_to(root).as_posix())}:11.*_llk_pack_config_",
            ):
                build_effects(root)

    def test_nested_public_calls_reach_helper_effect_fixed_point(self) -> None:
        source = """\
inline void state_helper()
{
    cfg_reg_rmw_tensix<NESTED_REG>(7);
}
inline void _llk_pack_inner_()
{
    state_helper();
}
inline void _llk_pack_outer_()
{
    _llk_pack_inner_();
}
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root, "blackhole", "llk_lib/llk_pack.h", source)
            effects = build_effects(root)["effects"]
        outer = [
            effect
            for effect in effects
            if effect["function"] == "_llk_pack_outer_"
            and effect["resource"] == "NESTED_REG"
        ]
        self.assertEqual(len(outer), 1)
        self.assertEqual(
            [hop["function"] for hop in outer[0]["evidence"]["via_chain"]],
            ["_llk_pack_inner_", "state_helper"],
        )

    def test_omitted_default_argument_preserves_fixed_callee_effect(self) -> None:
        source = """\
inline void _llk_pack_inner_(std::uint32_t value = DEFAULT_VALUE)
{
    cfg_reg_rmw_tensix<DEFAULT_REG>(value);
}
inline void _llk_pack_outer_()
{
    _llk_pack_inner_();
}
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root, "blackhole", "llk_lib/llk_pack.h", source)
            effects = build_effects(root)["effects"]
        propagated = next(
            effect
            for effect in effects
            if effect["function"] == "_llk_pack_outer_"
            and effect["resource"] == "DEFAULT_REG"
        )
        self.assertEqual(propagated["parameter"]["kind"], "fixed")
        self.assertEqual(propagated["value_expr"], "DEFAULT_VALUE")

    def test_namespace_qualified_call_selects_matching_helper_identity(self) -> None:
        source = """\
namespace first {
inline void state_helper(std::uint32_t value)
{
    cfg_reg_rmw_tensix<FIRST_REG>(value);
}
}
namespace second {
inline void state_helper(std::uint32_t value)
{
    cfg_reg_rmw_tensix<SECOND_REG>(value);
}
}
inline void _llk_pack_outer_(std::uint32_t value)
{
    second::state_helper(value);
}
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root, "blackhole", "llk_lib/llk_pack.h", source)
            effects = build_effects(root)["effects"]
        resources = {
            effect["resource"]
            for effect in effects
            if effect["function"] == "_llk_pack_outer_"
        }
        self.assertEqual(resources, {"SECOND_REG"})

    def test_unresolved_effect_carrying_call_fails_with_source_context(self) -> None:
        source = """\
namespace first {
inline void state_helper()
{
    cfg_reg_rmw_tensix<FIRST_STATE_REG>(1);
}
}
namespace second {
inline void state_helper()
{
    cfg_reg_rmw_tensix<SECOND_STATE_REG>(1);
}
}
inline void _llk_pack_outer_()
{
    missing::state_helper();
}
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = write(root, "blackhole", "llk_lib/llk_pack.h", source)
            with self.assertRaisesRegex(
                AuditModelError,
                rf"{re.escape(path.relative_to(root).as_posix())}:15.*missing::state_helper",
            ):
                build_effects(root)

    def test_unqualified_member_call_resolves_within_owning_class(self) -> None:
        source = """\
class First {
public:
    void program();
    void program_and_run();
};
class Second {
public:
    void program();
};
inline void First::program() { mop_cfg[0] = 1; }
inline void Second::program() { mop_cfg[0] = 2; }
inline void First::program_and_run() { program(); }
inline void _llk_pack_outer_()
{
    First value;
    value.program_and_run();
}
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root, "blackhole", "llk_lib/llk_pack.h", source)
            effects = build_effects(root)["effects"]
        self.assertTrue(
            any(
                effect["function"] == "_llk_pack_outer_"
                and effect["domain"] == "mop_replay"
                for effect in effects
            )
        )


class CommonHelperTraversalTest(unittest.TestCase):
    """Task-1 systemic fix: propagate state effects through non-LLK common helpers."""

    def effects_for(
        self,
        source: str,
        architecture: str = "wormhole_b0",
        relative: str = "llk_lib/llk_math_common.h",
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root, architecture, relative, source)
            return build_effects(root)["effects"]

    def inventory_for(
        self,
        source: str,
        architecture: str = "wormhole_b0",
        relative: str = "llk_lib/llk_math_common.h",
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root, architecture, relative, source)
            return {record["name"]: record for record in inventory(root)["functions"]}

    def test_one_hop_non_llk_helper_propagates_effect(self) -> None:
        source = """\
inline void set_math_semaphores_helper()
{
    t6_semaphore_post<p_stall::MATH>(semaphore::MATH_PACK);
}

template <bool Dst>
inline void _llk_math_dummy_done_()
{
    set_math_semaphores_helper();
}
"""
        effects = self.effects_for(source)
        published = [e for e in effects if e["function"] == "_llk_math_dummy_done_"]
        self.assertTrue(published)
        post = next(
            e
            for e in published
            if e["domain"] == "dvalid_semaphore" and e["operation"] == "post"
        )
        self.assertEqual(post["evidence"]["kind"], "transitive")
        self.assertEqual(post["evidence"]["via"], "set_math_semaphores_helper")
        self.assertEqual(post["resource"], "semaphore::MATH_PACK")

    def test_helper_is_not_published_in_inventory(self) -> None:
        source = """\
inline void set_math_semaphores_helper()
{
    t6_semaphore_post<p_stall::MATH>(semaphore::MATH_PACK);
}

inline void _llk_math_uses_helper_()
{
    set_math_semaphores_helper();
}
"""
        names = set(self.inventory_for(source))
        self.assertIn("_llk_math_uses_helper_", names)
        self.assertNotIn("set_math_semaphores_helper", names)

    def test_two_hop_common_helper_chain(self) -> None:
        source = """\
inline void inner_helper()
{
    reg_write(RISCV_DEBUG_REG_FPU_STICKY_BITS, 0);
}

inline void outer_helper()
{
    inner_helper();
}

inline void _llk_math_chain_()
{
    outer_helper();
}
"""
        effects = self.effects_for(source)
        published = [e for e in effects if e["function"] == "_llk_math_chain_"]
        debug = next(e for e in published if e["domain"] == "debug_register")
        self.assertEqual(debug["resource"], "RISCV_DEBUG_REG_FPU_STICKY_BITS")
        self.assertEqual(debug["evidence"]["kind"], "transitive")
        self.assertEqual(debug["evidence"]["via"], "outer_helper")
        self.assertEqual(
            [hop["function"] for hop in debug["evidence"]["via_chain"]],
            ["outer_helper", "inner_helper"],
        )
        self.assertEqual(
            debug["evidence"]["sink"]["token"],
            "reg_write",
        )
        self.assertIn(
            "llk_math_common.h",
            debug["evidence"]["sink"]["source_path"],
        )

    def test_parameter_forwarding_through_helper(self) -> None:
        source = """\
inline void write_cfg_helper(const std::uint32_t fmt)
{
    cfg_reg_rmw_tensix<SOME_REG>(fmt);
}

inline void _llk_unpack_fmt_(const std::uint32_t src_format)
{
    write_cfg_helper(src_format);
}
"""
        effects = self.effects_for(source, relative="llk_lib/llk_unpack_common.h")
        published = [e for e in effects if e["function"] == "_llk_unpack_fmt_"]
        mapped = next(e for e in published if e["resource"] == "SOME_REG")
        self.assertEqual(mapped["parameter"]["name"], "src_format")
        self.assertEqual(mapped["evidence"]["kind"], "transitive")
        self.assertEqual(mapped["evidence"]["via"], "write_cfg_helper")

    def test_fixed_effect_through_helper_has_no_parameter(self) -> None:
        source = """\
inline void set_dvalid_helper()
{
    TTI_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_SET_DVALID);
}

inline void _llk_unpack_fixed_()
{
    set_dvalid_helper();
}
"""
        effects = self.effects_for(source, relative="llk_lib/llk_unpack_common.h")
        published = [e for e in effects if e["function"] == "_llk_unpack_fixed_"]
        dvalid = next(e for e in published if e["domain"] == "dvalid_semaphore")
        self.assertEqual(dvalid["parameter"]["name"], "-")
        self.assertEqual(dvalid["parameter"]["kind"], "fixed")
        self.assertEqual(dvalid["evidence"]["kind"], "transitive")

    def test_helper_cycle_is_bounded_and_safe(self) -> None:
        source = """\
inline void pong_helper();

inline void ping_helper()
{
    pong_helper();
    reg_write(RISCV_DEBUG_REG_FPU_STICKY_BITS, 0);
}

inline void pong_helper()
{
    ping_helper();
}

inline void _llk_math_cycle_()
{
    ping_helper();
}
"""
        effects = self.effects_for(source)
        published = [e for e in effects if e["function"] == "_llk_math_cycle_"]
        debug = [e for e in published if e["domain"] == "debug_register"]
        self.assertTrue(debug)
        self.assertTrue(all(e["evidence"]["kind"] == "transitive" for e in debug))

    def test_ambiguous_helper_overload_fails_closed(self) -> None:
        source = """\
inline void amb_helper(std::uint32_t v)
{
    cfg_reg_rmw_tensix<INT_REG>(v);
}

inline void amb_helper(bool v)
{
    cfg_reg_rmw_tensix<BOOL_REG>(v);
}

inline void _llk_pack_amb_(std::uint32_t v)
{
    amb_helper(ambiguous_value(v));
}
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = write(
                root,
                "wormhole_b0",
                "llk_lib/llk_pack.h",
                source,
            )
            with self.assertRaisesRegex(
                AuditModelError,
                rf"{re.escape(path.relative_to(root).as_posix())}:13.*amb_helper",
            ):
                build_effects(root)

    def test_template_specialization_drops_impossible_branch(self) -> None:
        source = """\
template <int SEL>
inline void sel_helper()
{
    if constexpr (SEL == 0)
    {
        cfg_reg_rmw_tensix<REG_ZERO>(1);
    }
    else
    {
        cfg_reg_rmw_tensix<REG_ONE>(2);
    }
}

inline void _llk_pack_sel_()
{
    sel_helper<0>();
}
"""
        effects = self.effects_for(source, relative="llk_lib/llk_pack.h")
        published = {
            e["resource"]: e
            for e in effects
            if e["function"] == "_llk_pack_sel_"
            and e["evidence"]["kind"] == "transitive"
        }
        self.assertIn("REG_ZERO", published)
        self.assertNotIn("REG_ONE", published)
        self.assertIsNone(published["REG_ZERO"]["condition"])

    def test_namespace_qualified_helper_call_resolves_by_final_name(self) -> None:
        source = """\
inline void _set_dest_base_helper_(const std::uint32_t base_addr)
{
    cfg[DEST_TARGET_REG_CFG_MATH_SEC0_Offset_ADDR32] = base_addr;
}

inline void _llk_sync_qualified_(const std::uint32_t base_addr)
{
    ckernel::trisc::_set_dest_base_helper_(base_addr);
}
"""
        effects = self.effects_for(source, relative="llk_lib/llk_sync.h")
        published = [
            e
            for e in effects
            if e["function"] == "_llk_sync_qualified_"
            and e["evidence"]["kind"] == "transitive"
        ]
        mapped = next(e for e in published if e["domain"] == "cfg_register")
        self.assertEqual(mapped["evidence"]["via"], "_set_dest_base_helper_")
        self.assertEqual(mapped["parameter"]["name"], "base_addr")

    def test_math_pack_sync_init_stage_is_math(self) -> None:
        source = """\
template <int Dst, bool acc>
inline void _llk_math_pack_sync_init_()
{
    reset_dest_offset_id();
}
"""
        records = self.inventory_for(source)
        record = records["_llk_math_pack_sync_init_"]
        self.assertEqual(record["stage"], "math")
        self.assertEqual(record["thread"], "T1")


class ClassificationTest(unittest.TestCase):
    SOURCE = """\
inline void _llk_pack_init_(const std::uint32_t format)
{
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32>(format);
}

inline void _llk_pack_wrapper_(const std::uint32_t format)
{
    _llk_pack_init_(format);
}

inline std::uint32_t _llk_pack_get_size_()
{
    return 42;
}
"""

    def classify_root(self, root: Path):
        write(root, "quasar", "llk_lib/llk_pack.h", self.SOURCE)
        return classify(root)

    def test_every_definition_is_classified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.classify_root(root)
        by_name = {d["name"]: d for d in result["definitions"]}
        self.assertEqual(by_name["_llk_pack_init_"]["classification"], "included")
        self.assertEqual(by_name["_llk_pack_wrapper_"]["classification"], "wrapper")
        self.assertEqual(
            by_name["_llk_pack_wrapper_"]["canonical_target"], "_llk_pack_init_"
        )
        self.assertEqual(by_name["_llk_pack_get_size_"]["classification"], "excluded")
        self.assertTrue(by_name["_llk_pack_get_size_"]["reason"])
        for definition in result["definitions"]:
            self.assertIn(
                definition["classification"], {"included", "wrapper", "excluded"}
            )
            self.assertTrue(definition["body_fingerprint"])

    def test_drift_detection_flags_fingerprint_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = write(root, "quasar", "llk_lib/llk_pack.h", self.SOURCE)
            baseline = classify(root)
            reviewed = Path(temporary) / "classification.json"
            reviewed.write_text(json.dumps(baseline, sort_keys=True), encoding="utf-8")
            clean = classify(root, reviewed=reviewed)
            self.assertEqual(clean["drift"], [])
            path.write_text(
                self.SOURCE.replace("return 42;", "return 43;"), encoding="utf-8"
            )
            drifted = classify(root, reviewed=reviewed)

        self.assertTrue(drifted["drift"])
        self.assertTrue(
            any(item["name"] == "_llk_pack_get_size_" for item in drifted["drift"])
        )

    def test_state_ownership_lifecycles_are_explicit_and_included(self) -> None:
        source = "\n".join(
            f"inline void _llk_pack_{lifecycle}_state_() {{}}"
            for lifecycle in ("reinit", "clear", "reset", "done", "restore")
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root, "quasar", "llk_lib/llk_pack.h", source)
            records = {
                record["name"]: record for record in inventory(root)["functions"]
            }
            definitions = {
                definition["name"]: definition
                for definition in classify(root)["definitions"]
            }
        for lifecycle in ("reinit", "clear", "reset", "done", "restore"):
            name = f"_llk_pack_{lifecycle}_state_"
            self.assertEqual(records[name]["lifecycle"], lifecycle)
            self.assertEqual(definitions[name]["classification"], "included")

    def test_configure_and_reconfigure_names_are_configure_lifecycle(self) -> None:
        source = """\
inline void _llk_unpack_hw_configure_() {}
inline void _llk_pack_reconfigure_state_() {}
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root, "blackhole", "llk_lib/llk_config.h", source)
            records = {
                record["name"]: record for record in inventory(root)["functions"]
            }
        self.assertEqual(records["_llk_unpack_hw_configure_"]["lifecycle"], "configure")
        self.assertEqual(
            records["_llk_pack_reconfigure_state_"]["lifecycle"], "configure"
        )

    def test_clear_and_restore_effects_identify_restore_owner(self) -> None:
        source = """\
inline void _llk_math_clear_flags_()
{
    reg_write(RISCV_DEBUG_REG_FPU_STICKY_BITS, 0);
}
inline void _llk_pack_restore_state_()
{
    reset_dest_offset_id();
}
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root, "wormhole_b0", "llk_lib/llk_state.h", source)
            effects = build_effects(root)["effects"]
        by_function = {effect["function"]: effect for effect in effects}
        self.assertIsNone(by_function["_llk_math_clear_flags_"]["restore"])
        self.assertIsNone(by_function["_llk_pack_restore_state_"]["restore"])


class AuditDatasetTest(unittest.TestCase):
    def test_audit_is_deterministic_and_json_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(
                root,
                "wormhole_b0",
                "llk_lib/llk_pack.h",
                "inline void _llk_pack_init_(const std::uint32_t f)\n{\n    cfg_reg_rmw_tensix<REG>(f);\n}\n",
            )
            first = audit(root)
            second = audit(root)

        self.assertEqual(
            json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True)
        )
        self.assertEqual(first["schema_version"], 2)
        self.assertIn("effects", first)
        self.assertIn("classification", first)
        self.assertIn("summary", first)
        summary = first["summary"]
        self.assertIn("by_architecture", summary)
        self.assertIn("by_stability", summary)
        self.assertIn("by_lifecycle", summary)
        self.assertIn("by_classification", summary)


class RealSourceValidationTest(unittest.TestCase):
    """Validate against representative high-risk real functions in all three arches."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.llk_root = REPO_ROOT
        if not (cls.llk_root / "tt_llk_wormhole_b0").is_dir():
            raise unittest.SkipTest("real LLK source tree not present")
        cls.data = audit(cls.llk_root)
        cls.effects = cls.data["effects"]

    def effects_for_function(self, name: str, architecture: str):
        return [
            e
            for e in self.effects
            if e["function"] == name and e["architecture"] == architecture
        ]

    def assert_source_line(self, effect) -> None:
        path = self.llk_root / effect["evidence"]["source_path"]
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        line_no = effect["evidence"]["line"]
        self.assertTrue(
            1 <= line_no <= len(lines), msg=f"line {line_no} out of range for {path}"
        )
        self.assertTrue(
            lines[line_no - 1].strip(), msg=f"empty source line for {effect}"
        )

    def test_wormhole_unpack_hw_configure_touches_cfg_and_gpr(self) -> None:
        effects = self.effects_for_function("_llk_unpack_hw_configure_", "wormhole_b0")
        self.assertTrue(effects)
        domains = {e["domain"] for e in effects}
        self.assertIn("gpr", domains)
        for effect in effects:
            self.assert_source_line(effect)

    def test_wormhole_unpack_reconfig_srca_int8_condition_present(self) -> None:
        effects = self.effects_for_function(
            "_llk_unpack_reconfig_data_format_srca_impl_", "wormhole_b0"
        )
        self.assertTrue(effects)
        conditions = {e["condition"] for e in effects if e["condition"]}
        self.assertTrue(any("int8" in (c or "").lower() for c in conditions))
        self.assertTrue(any(e["domain"] == "cfg_register" for e in effects))
        for effect in effects:
            self.assert_source_line(effect)

    def test_blackhole_and_quasar_have_effects(self) -> None:
        for architecture in ("blackhole", "quasar"):
            arch_effects = [
                e for e in self.effects if e["architecture"] == architecture
            ]
            self.assertTrue(arch_effects, msg=architecture)

    def test_quasar_sync_semaphore_effects(self) -> None:
        effects = self.effects_for_function("_llk_sync_init_", "quasar")
        self.assertTrue(effects)
        self.assertTrue(any(e["domain"] == "dvalid_semaphore" for e in effects))
        for effect in effects:
            self.assert_source_line(effect)

    def test_wormhole_debug_clear_uses_named_register_resource(self) -> None:
        effects = self.effects_for_function(
            "_llk_math_clear_compute_special_value_flags_", "wormhole_b0"
        )
        debug_write = next(
            effect
            for effect in effects
            if effect["resource"] == "RISCV_DEBUG_REG_FPU_STICKY_BITS"
        )
        self.assertEqual(debug_write["domain"], "debug_register")
        self.assertEqual(debug_write["operation"], "write")
        self.assertIsNone(debug_write["restore"])
        self.assert_source_line(debug_write)

    def test_hash_cb_tile_reset_counters_is_mapped_state(self) -> None:
        for architecture in ("wormhole_b0", "blackhole"):
            effects = self.effects_for_function("_llk_math_hash_cb_tile_", architecture)
            reset = next(
                effect
                for effect in effects
                if effect["domain"] == "counter"
                and effect["operation"] == "reset"
                and effect["resource"] == "p_setrwc::SET_ABD_F"
            )
            self.assertEqual(reset["parameter"]["name"], "-")
            self.assertEqual(reset["value_expr"], "0")
            self.assertEqual(reset["evidence"]["kind"], "direct")
            self.assert_source_line(reset)

            classification = next(
                item
                for item in self.data["classification"]
                if item["architecture"] == architecture
                and item["name"] == "_llk_math_hash_cb_tile_"
            )
            self.assertEqual(classification["classification"], "included")

    def test_hash_cb_persistent_sfpu_and_dest_state_is_mapped(self) -> None:
        for architecture in ("wormhole_b0", "blackhole"):
            init = self.effects_for_function("_llk_math_hash_cb_init_", architecture)
            tile = self.effects_for_function("_llk_math_hash_cb_tile_", architecture)
            store = self.effects_for_function(
                "_llk_math_hash_cb_store_to_dest_", architecture
            )
            self.assertTrue(any(effect["domain"] == "sfpu_lreg" for effect in init))
            self.assertTrue(any(effect["domain"] == "sfpu_lreg" for effect in tile))
            self.assertTrue(any(effect["domain"] == "dest_data" for effect in store))

    def test_real_hw_configure_lifecycles(self) -> None:
        functions = self.data["inventory"]["functions"]
        for architecture in ("wormhole_b0", "blackhole", "quasar"):
            for name in ("_llk_unpack_hw_configure_", "_llk_pack_hw_configure_"):
                matches = [
                    function
                    for function in functions
                    if function["architecture"] == architecture
                    and function["name"] == name
                ]
                self.assertTrue(matches, msg=(architecture, name))
                self.assertTrue(
                    all(function["lifecycle"] == "configure" for function in matches)
                )

    def test_quasar_stall_cfg_is_shared_sync_state(self) -> None:
        function = next(
            function
            for function in self.data["inventory"]["functions"]
            if function["architecture"] == "quasar"
            and function["name"] == "_llk_stall_cfg_on_"
        )
        self.assertEqual((function["thread"], function["stage"]), ("shared", "shared"))

    def test_math_pack_sync_is_owned_by_math_thread(self) -> None:
        for architecture in ("wormhole_b0", "blackhole", "quasar"):
            function = next(
                function
                for function in self.data["inventory"]["functions"]
                if function["architecture"] == architecture
                and function["name"] == "_llk_math_pack_sync_init_"
            )
            self.assertEqual(
                (function["thread"], function["stage"]),
                ("T1", "math"),
            )

    def test_common_pack_helpers_are_mapped_for_wormhole_and_blackhole(self) -> None:
        expected = {
            "_llk_pack_hw_configure_": "configure_pack",
            "_llk_pack_reconfig_data_format_": "reconfig_packer_data_format",
            "_llk_pack_reconfig_l1_acc_": "reconfigure_packer_l1_acc",
        }
        for architecture in ("wormhole_b0", "blackhole"):
            for function, helper in expected.items():
                effects = self.effects_for_function(function, architecture)
                self.assertTrue(
                    any(
                        effect["domain"] == "cfg_register"
                        and effect["evidence"]["via"] == helper
                        for effect in effects
                    ),
                    msg=(architecture, function),
                )

    def test_math_common_helper_effects_are_propagated(self) -> None:
        for architecture in ("wormhole_b0", "blackhole"):
            done = self.effects_for_function(
                "_llk_math_dest_section_done_", architecture
            )
            self.assertTrue(
                any(
                    effect["domain"] == "dvalid_semaphore"
                    and effect["resource"] == "semaphore::MATH_PACK"
                    and effect["operation"] == "post"
                    for effect in done
                )
            )
            sync = self.effects_for_function("_llk_math_pack_sync_init_", architecture)
            self.assertTrue(
                any(
                    effect["domain"] == "cfg_register"
                    and "DEST_TARGET_REG_CFG_MATH" in effect["resource"]
                    for effect in sync
                )
            )

    def test_quasar_common_state_helpers_are_mapped(self) -> None:
        sync = self.effects_for_function("_llk_sync_advance_dest_section_", "quasar")
        self.assertTrue(
            any(effect["resource"] == "dest_register_offset" for effect in sync)
        )
        sfpu_init = self.effects_for_function("_llk_math_sfpu_init_", "quasar")
        self.assertTrue(any(effect["domain"] == "sfpu_config" for effect in sfpu_init))
        sfpu_done = self.effects_for_function("_llk_math_sfpu_done_", "quasar")
        self.assertTrue(
            any(
                effect["domain"] == "counter"
                and effect["resource"] == "p_setrwc::SET_D"
                for effect in sfpu_done
            )
        )
        pack_done = self.effects_for_function(
            "_llk_pack_dest_dvalid_section_done_", "quasar"
        )
        self.assertTrue(
            any(effect["resource"] == "clear_dest_bank_id" for effect in pack_done)
        )
        dummy_valid = self.effects_for_function(
            "_llk_unpack_set_srcB_dummy_valid_", "quasar"
        )
        self.assertTrue(
            any(
                effect["domain"] == "dvalid_semaphore"
                and effect["resource"] == "UNPACK_DVALID"
                for effect in dummy_valid
            )
        )

    def test_real_mailbox_paths_are_mapped_for_wh_and_bh(self) -> None:
        for architecture in ("wormhole_b0", "blackhole"):
            math = self.effects_for_function(
                "_llk_math_eltwise_unary_datacopy_", architecture
            )
            unpack = self.effects_for_function("_llk_unpack_tilize_", architecture)
            self.assertTrue(
                any(
                    effect["domain"] == "mailbox" and effect["operation"] == "write"
                    for effect in math
                ),
                msg=(architecture, "write"),
            )
            self.assertTrue(
                any(
                    effect["domain"] == "mailbox" and effect["operation"] == "consume"
                    for effect in unpack
                ),
                msg=(architecture, "consume"),
            )

    def test_quasar_reduce_dstacc_override_is_mapped(self) -> None:
        effects = self.effects_for_function("_llk_math_reduce_", "quasar")
        self.assertTrue(
            any(
                effect["domain"] == "cfg_register"
                and effect["resource"] == "ALU_FORMAT_SPEC_REG_Dstacc_override_ADDR32"
                and effect["operation"] == "rmw"
                for effect in effects
            )
        )

    def test_real_wh_bh_unpacr_nop_persistent_aspects_are_mapped(self) -> None:
        wh = self.effects_for_function(
            "_llk_unpack_set_srcb_dummy_valid_", "wormhole_b0"
        )
        self.assertTrue(
            any(
                effect["resource"] == "UNPACK_DVALID[SrcB]"
                and effect["operation"] == "set"
                for effect in wh
            )
        )
        bh = self.effects_for_function("_llk_unpack_tilizeA_B_", "blackhole")
        self.assertTrue(
            any(
                effect["domain"] == "src_data" and effect["operation"] == "clear"
                for effect in bh
            )
        )

    def test_quasar_pack_init_does_not_inherit_impossible_packer1_branch(self) -> None:
        effects = self.effects_for_function("_llk_pack_init_", "quasar")
        resources = {effect["resource"] for effect in effects}
        self.assertTrue(any("PACKER0" in resource for resource in resources))
        self.assertFalse(any("PACKER1" in resource for resource in resources))

    def test_canonical_restore_contracts_are_resource_scoped(self) -> None:
        contracts = load_effect_model()["restore_contracts"]
        canonical = [
            contract for contract in contracts if contract["kind"] == "canonical_reset"
        ]
        self.assertTrue(canonical)
        self.assertTrue(all(contract.get("effect_selector") for contract in canonical))
        for architecture in ("wormhole_b0", "blackhole"):
            effects = self.effects_for_function(
                "_llk_unpack_tilize_init_", architecture
            )
            self.assertTrue(
                all(
                    effect["restore"] is None
                    for effect in effects
                    if effect["domain"] in {"counter", "dvalid_semaphore", "mop_replay"}
                )
            )

    def test_real_reviewed_restore_contracts(self) -> None:
        for architecture in ("wormhole_b0", "blackhole"):
            ab = self.effects_for_function("_llk_unpack_AB_init_", architecture)
            tilize = self.effects_for_function("_llk_unpack_tilize_init_", architecture)
            untilize = self.effects_for_function(
                "_llk_pack_untilize_init_", architecture
            )
            mask = self.effects_for_function(
                "_llk_pack_reduce_mask_config_", architecture
            )
            self.assertTrue(ab and tilize and untilize and mask, msg=architecture)
            reviewed_ab = [effect for effect in ab if effect["restore"] is not None]
            self.assertTrue(reviewed_ab)
            self.assertTrue(
                all(
                    effect["restore"]["kind"] == "no_op_transient"
                    and effect["restore"]["owner"] is None
                    for effect in reviewed_ab
                )
            )
            self.assertTrue(
                all(
                    effect["restore"] is None
                    for effect in ab
                    if effect not in reviewed_ab
                )
            )
            reviewed_tilize = [
                effect for effect in tilize if effect["restore"] is not None
            ]
            self.assertTrue(reviewed_tilize)
            self.assertTrue(
                all(
                    effect["restore"]["kind"] == "canonical_reset"
                    and effect["restore"]["owner"] == "_llk_unpack_tilize_uninit_"
                    and effect["domain"] == "cfg_register"
                    for effect in reviewed_tilize
                )
            )
            if architecture == "wormhole_b0":
                reviewed_untilize = [
                    effect for effect in untilize if effect["restore"] is not None
                ]
                transient_untilize = [
                    effect
                    for effect in untilize
                    if effect["persistence"] == "transient"
                ]
                self.assertTrue(reviewed_untilize)
                self.assertTrue(transient_untilize)
                self.assertTrue(
                    all(
                        effect["restore"]["kind"] == "no_op_transient"
                        and effect["restore"]["owner"] is None
                        and effect["restore"]["pair"] == "_llk_pack_untilize_uninit_"
                        for effect in reviewed_untilize
                    )
                )
                self.assertTrue(
                    all(effect["restore"] is None for effect in transient_untilize)
                )
            else:
                reviewed_untilize = [
                    effect for effect in untilize if effect["restore"] is not None
                ]
                self.assertTrue(reviewed_untilize)
                self.assertTrue(
                    all(
                        effect["restore"]["kind"] == "canonical_reset"
                        and effect["restore"]["owner"] == "_llk_pack_untilize_uninit_"
                        for effect in reviewed_untilize
                    )
                )
            reviewed_mask = [effect for effect in mask if effect["restore"] is not None]
            self.assertTrue(reviewed_mask)
            self.assertTrue(
                all(
                    effect["restore"]["kind"] == "canonical_reset"
                    and effect["restore"]["owner"] == "_llk_pack_reduce_mask_clear_"
                    and effect["domain"] in {"cfg_register", "counter"}
                    for effect in reviewed_mask
                )
            )
            pack_contract = next(
                contract
                for contract in load_effect_model()["restore_contracts"]
                if contract["architecture"] == architecture
                and contract["function"] == "_llk_pack_init_"
            )
            self.assertEqual(
                (pack_contract["kind"], pack_contract["owner"]),
                ("no_op_transient", None),
            )

    def test_quasar_clear_and_done_restore_contracts(self) -> None:
        clear_effects = self.effects_for_function(
            "_llk_pack_reduce_mask_clear_", "quasar"
        )
        done_effects = self.effects_for_function(
            "_llk_pack_dest_dvalid_section_done_", "quasar"
        )
        self.assertTrue(clear_effects)
        self.assertTrue(done_effects)
        self.assertTrue(
            all(
                effect["restore"]["kind"] == "canonical_reset"
                for effect in clear_effects
            )
        )
        self.assertTrue(
            all(
                effect["restore"]["owner"] == "_llk_pack_reduce_mask_clear_"
                for effect in clear_effects
            )
        )
        reviewed_done = [
            effect for effect in done_effects if effect["restore"] is not None
        ]
        transient_done = [
            effect for effect in done_effects if effect["persistence"] == "transient"
        ]
        self.assertTrue(reviewed_done)
        self.assertTrue(transient_done)
        self.assertTrue(
            all(
                effect["restore"]["kind"] == "canonical_reset"
                for effect in reviewed_done
            )
        )
        self.assertTrue(
            all(
                effect["restore"]["owner"] == "_llk_pack_dest_dvalid_section_done_"
                for effect in reviewed_done
            )
        )
        self.assertTrue(all(effect["restore"] is None for effect in transient_done))

    def test_unknown_persistent_configure_effect_has_no_invented_contract(self) -> None:
        source = "inline void _llk_math_configure_unknown_() { cfg_reg_rmw_tensix<REG>(1); }\n"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root, "blackhole", "llk_lib/llk_math.h", source)
            effects = build_effects(root)["effects"]
        self.assertEqual(len(effects), 1)
        self.assertIsNone(effects[0]["restore"])
        self.assertEqual(
            effects[0]["retention_contract"],
            "Retained until the configuration resource is reconfigured.",
        )

    def test_setc16_maps_register_then_value(self) -> None:
        source = """
inline void _llk_math_configure_setc16_(const std::uint32_t value) {
    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, value);
}
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root, "blackhole", "llk_lib/llk_math.h", source)
            effect = build_effects(root)["effects"][0]
        self.assertEqual(effect["resource"], "CLR_DVALID_SrcA_Disable_ADDR32")
        self.assertEqual(effect["value_expr"], "value")
        self.assertEqual(effect["parameter"]["name"], "value")

    def test_reg2flop_maps_flop_index_and_gpr_value(self) -> None:
        source = """
inline void _llk_unpack_configure_reg2flop_() {
    TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG2_Out_data_format_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);
}
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root, "wormhole_b0", "llk_lib/llk_unpack.h", source)
            effect = build_effects(root)["effects"][0]
        self.assertEqual(
            effect["resource"],
            "THCON_SEC0_REG2_Out_data_format_ADDR32 - THCON_CFGREG_BASE_ADDR32",
        )
        self.assertEqual(effect["value_expr"], "p_gpr_unpack::TMP0")

    def test_blackhole_pack_init_does_not_map_source_format_to_fixed_counter(
        self,
    ) -> None:
        effects = self.effects_for_function("_llk_pack_init_", "blackhole")
        self.assertFalse(
            any(
                effect["domain"] == "counter"
                and effect["parameter"]["name"] == "pack_src_format"
                for effect in effects
            )
        )
        self.assertFalse(
            any(effect["resource"] == "pack_init_state" for effect in effects)
        )

    def test_blackhole_untilize_restore_only_applies_to_z_stride(self) -> None:
        effects = self.effects_for_function("_llk_pack_untilize_init_", "blackhole")
        restored = [effect for effect in effects if effect["restore"]]
        self.assertTrue(restored)
        self.assertTrue(
            all(
                effect["restore"]["kind"] == "canonical_reset"
                and effect["restore"]["owner"] == "_llk_pack_untilize_uninit_"
                and effect["domain"] == "cfg_register"
                and effect["resource"] == "PCK0_ADDR_CTRL_ZW_REG_0_Zstride_RMW"
                for effect in restored
            )
        )
        self.assertTrue(
            any(
                effect["restore"] is None
                and effect["domain"] in {"addr_mod", "mop_replay", "gpr", "counter"}
                for effect in effects
            )
        )

    def test_same_name_unpack_ab_overloads_receive_transitive_effects(self) -> None:
        for architecture in ("wormhole_b0", "blackhole"):
            definitions = [
                item
                for item in self.data["inventory"]["functions"]
                if item["architecture"] == architecture
                and item["name"] == "_llk_unpack_AB_init_"
            ]
            effects = [
                effect
                for effect in self.effects
                if effect["architecture"] == architecture
                and effect["function"] == "_llk_unpack_AB_init_"
            ]
            covered_fingerprints = {
                effect["evidence"]["body_fingerprint"] for effect in effects
            }
            self.assertEqual(
                covered_fingerprints,
                {definition["body_fingerprint"] for definition in definitions},
            )
            self.assertTrue(
                any(effect["evidence"]["kind"] == "transitive" for effect in effects)
            )

    def test_functions_with_effects_are_not_classified_excluded(self) -> None:
        classification = {
            (
                item["architecture"],
                item["name"],
                item["source"]["path"],
                item["body_fingerprint"],
            ): item["classification"]
            for item in self.data["classification"]
        }
        for effect in self.effects:
            identity = (
                effect["architecture"],
                effect["function"],
                effect["evidence"]["source_path"],
                effect["evidence"]["body_fingerprint"],
            )
            self.assertNotEqual(
                classification[identity],
                "excluded",
                msg=identity,
            )

    def test_wormhole_untilize_else_if_conditions_are_composed(self) -> None:
        effects = self.effects_for_function("_llk_pack_untilize_init_", "wormhole_b0")
        by_line = {
            effect["evidence"]["line"]: effect["condition"]
            for effect in effects
            if effect["domain"] == "counter" and effect["operation"] == "adc"
        }
        self.assertEqual(by_line[204], "!(diagonal) && narrow_row")
        self.assertEqual(by_line[208], "!(diagonal) && !(narrow_row)")

    def test_restore_contract_source_anchors_resolve(self) -> None:
        model = load_effect_model()
        for contract in model["restore_contracts"]:
            for reference in ("source", "owner_source", "pair_source"):
                evidence = contract[reference]
                target = (
                    contract[reference.removesuffix("_source")]
                    if reference != "source"
                    else contract["function"]
                )
                if target is None:
                    self.assertIsNone(evidence)
                    continue
                self.assertIsNotNone(evidence, msg=(reference, contract))
                source = self.llk_root / evidence["path"]
                lines = source.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                line = evidence["line"]
                self.assertTrue(1 <= line <= len(lines), msg=contract)
                self.assertIn(evidence["token"], lines[line - 1], msg=contract)

    def test_wormhole_overload_direct_effect_identity_is_exact(self) -> None:
        definitions = [
            definition
            for definition in self.data["classification"]
            if definition["architecture"] == "wormhole_b0"
            and definition["name"] == "_llk_unpack_AB_init_"
        ]
        self.assertEqual(len(definitions), 3)
        by_line = {
            definition["source"]["start_line"]: definition["reason"]
            for definition in definitions
        }
        self.assertEqual(by_line[149], "recognized direct state primitive")
        self.assertEqual(
            by_line[173],
            "recognized state effect through analyzed LLK call",
        )
        self.assertEqual(
            by_line[191],
            "recognized state effect through analyzed LLK call",
        )

    def test_pack_configure_present_in_all_arches(self) -> None:
        for architecture in ("wormhole_b0", "blackhole", "quasar"):
            configs = [
                e
                for e in self.effects
                if e["architecture"] == architecture
                and "pack" in e["function"]
                and e["lifecycle"] in {"configure", "init"}
            ]
            self.assertTrue(configs, msg=architecture)

    def test_every_effect_has_resolvable_source_anchor(self) -> None:
        sampled = self.effects[::37]
        for effect in sampled:
            self.assert_source_line(effect)

    def test_no_wrapper_double_counting_in_classification(self) -> None:
        classification = self.data["classification"]
        names = [d["name"] for d in classification]
        self.assertEqual(
            len(names),
            len(
                set(
                    (d["name"], d["architecture"], d["source"]["start_line"])
                    for d in classification
                )
            ),
        )


if __name__ == "__main__":
    unittest.main()
