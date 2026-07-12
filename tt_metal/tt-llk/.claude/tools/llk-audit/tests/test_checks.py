# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Hermetic unit tests for the checker logic — no clang / no repo needed. Each test
builds a tiny synthetic fact base (the exact shape the C++ extractor emits) and
asserts the checker's behavior, including the ground-truth cases and the
false-positive fixes found during validation.

Run:  python3 tests/test_checks.py     (from the llk-audit dir; prints PASS/FAIL)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llkaudit.checks.cb_sync import CbSync
from llkaudit.checks.cfg_word_overlap import CfgWordOverlap
from llkaudit.checks.mailbox_sync import MailboxSync
from llkaudit.checks.mmio_race import MmioRace
from llkaudit.checks.noc_sync import NocSync
from llkaudit.checks.reconfig_stall import ReconfigStall
from llkaudit.checks.semaphore_handshake import SemaphoreHandshake
from llkaudit.checks.srcreg_bank import SrcRegBank
from llkaudit.factbase import FactBase


def fn(name, file, b, e):
    return {
        "family": "function",
        "name": name,
        "file": file,
        "off": b,
        "end_off": e,
        "line": b,
    }


def pw(file, off, producer, index, prov="call", func=""):
    return {
        "family": "pointer_write",
        "file": file,
        "off": off,
        "line": off,
        "function": func,
        "op": "=",
        "provenance_kind": prov,
        "producer": producer,
        "index_text": index,
    }


def macro(file, off, name, text, func=""):
    return {
        "family": "macro",
        "file": file,
        "off": off,
        "line": off,
        "function": func,
        "name": name,
        "text": text,
    }


def call(file, off, name, text=None, func="", arg0="", recv="", recv_type="", argc=-1):
    return {
        "family": "call",
        "file": file,
        "off": off,
        "line": off,
        "function": func,
        "name": name,
        "text": text or (name + "()"),
        "arg0": arg0,
        "recv": recv,
        "recv_type": recv_type,
        "argc": argc,
    }


CASES = []


def case(fn_):
    CASES.append(fn_)
    return fn_


@case
def test_mmio_locally_ordered_via_trisc_cfg():
    F = "tt_llk_wormhole_b0/llk_lib/llk_unpack_A.h"
    facts = [
        fn("_llk_unpack_A_", F, 100, 200),
        pw(F, 110, "get_cfg_pointer", "upk0_reg", func="_llk_unpack_A_"),
        macro(
            F,
            120,
            "TTI_STALLWAIT",
            "TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG)",
            func="_llk_unpack_A_",
        ),
    ]
    fb = FactBase("wormhole", facts)
    out = MmioRace().run(fb)
    assert len(out) == 1 and out[0].hint == "LOCALLY_ORDERED", out
    assert "TTI_STALLWAIT" in out[0].evidence[0]


@case
def test_mmio_no_local_ordering():
    F = "tt_llk_wormhole_b0/llk_lib/llk_unpack_common.h"
    facts = [
        fn("_llk_unpack_configure_addresses_", F, 300, 350),
        pw(
            F,
            313,
            "cfg",
            "THCON_SEC0_REG3_Base_address_ADDR32",
            prov="var",
            func="_llk_unpack_configure_addresses_",
        ),
    ]
    out = MmioRace().run(FactBase("wormhole", facts))
    assert len(out) == 1 and out[0].hint == "NO_LOCAL_ORDERING", out


@case
def test_mmio_sync_regfile_only_orders_gpr():
    F = "tt_llk_wormhole_b0/common/inc/cunpack_common.h"
    # a GPR (regfile) write followed by sync_regfile_write -> ordered
    facts = [
        fn("configure_unpack_AB", F, 800, 950),
        pw(F, 920, "get_regfile_pointer", "p_gpr::X", func="configure_unpack_AB"),
        call(F, 925, "sync_regfile_write", func="configure_unpack_AB"),
    ]
    out = MmioRace().run(FactBase("wormhole", facts))
    assert out[0].hint == "LOCALLY_ORDERED", out
    # but sync_regfile_write must NOT order a cfg (non-GPR) write
    facts2 = [
        fn("f", F, 800, 950),
        pw(F, 920, "get_cfg_pointer", "SOME_ADDR32", func="f"),
        call(F, 925, "sync_regfile_write", func="f"),
    ]
    out2 = MmioRace().run(FactBase("wormhole", facts2))
    assert out2[0].hint == "NO_LOCAL_ORDERING", out2


@case
def test_cfg_word_overlap_config_vs_threadconfig_not_aliased():
    # A Config write (cfg[] / here THCON via get_cfg_pointer) and a ThreadConfig
    # write (SETC16) at the SAME index are in different hardware arrays and must
    # NOT be reported as a shared word (BackendConfiguration.md).
    Fp = "tt_llk_wormhole_b0/common/inc/cpack_common.h"
    Fm = "tt_llk_wormhole_b0/llk_lib/llk_math_reduce.h"
    facts = [
        fn("set_packer_config", Fp, 400, 600),
        fn("mathfn", Fm, 40, 80),
        pw(
            Fp,
            493,
            "get_cfg_pointer",
            "THCON_SEC0_REG1_Row_start_section_size_ADDR32",
            func="set_packer_config",
        ),
        macro(
            Fm,
            53,
            "TTI_SETC16",
            "TTI_SETC16(FP16A_FORCE_Enable_ADDR32, x)",
            func="mathfn",
        ),
    ]
    fb = FactBase("wormhole", facts)
    fb.addr32 = {
        "THCON_SEC0_REG1_Row_start_section_size_ADDR32": 56,
        "FP16A_FORCE_Enable_ADDR32": 56,
    }
    shared = [
        f for f in CfgWordOverlap().run(fb) if f.hint == "CROSS_THREAD_SHARED_WORD"
    ]
    assert (
        shared == []
    ), f"Config/ThreadConfig index-56 alias must not be flagged: {shared}"


@case
def test_cfg_word_overlap_setc16_does_not_contaminate_config_word():
    # The word-0 regression: an ALU field (Config, cfg_reg_rmw) shared by two
    # threads is a real finding; a SETC16 (ThreadConfig) write at the SAME index
    # must neither create a false pairing nor pollute the Config finding.
    Fu = "tt_llk_wormhole_b0/common/inc/cunpack_common.h"
    Fm = "tt_llk_wormhole_b0/llk_lib/llk_math_common.h"
    Fc = "tt_llk_wormhole_b0/common/inc/cmath_common.h"
    facts = [
        fn("u", Fu, 810, 830),
        fn("m", Fm, 50, 70),
        fn("c", Fc, 250, 300),
        call(
            Fu,
            820,
            "cfg_reg_rmw_tensix",
            "cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32>",
            func="u",
        ),
        call(
            Fm,
            55,
            "cfg_reg_rmw_tensix",
            "cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32>",
            func="m",
        ),
        macro(
            Fc,
            256,
            "TTI_SETC16",
            "TTI_SETC16(CFG_STATE_ID_StateID_ADDR32, x)",
            func="c",
        ),
    ]
    fb = FactBase("wormhole", facts)
    fb.addr32 = {
        "ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32": 0,
        "CFG_STATE_ID_StateID_ADDR32": 0,
    }
    shared = [
        f for f in CfgWordOverlap().run(fb) if f.hint == "CROSS_THREAD_SHARED_WORD"
    ]
    assert len(shared) == 1 and shared[0].kind == "shared_word@CONFIG:0", shared
    assert all("CFG_STATE_ID" not in e for e in shared[0].evidence), shared[0].evidence


@case
def test_cfg_word_overlap_real_shared_word():
    # Two threads writing MAIN word 1 -> flagged.
    Fu = "tt_llk_wormhole_b0/common/inc/cunpack_common.h"
    Fm = "tt_llk_wormhole_b0/llk_lib/llk_math_common.h"
    facts = [
        fn("cfgu", Fu, 840, 850),
        fn("cfgm", Fm, 50, 60),
        call(
            Fu,
            847,
            "cfg_reg_rmw_tensix",
            "cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32>",
            func="cfgu",
        ),
        call(
            Fm,
            52,
            "cfg_reg_rmw_tensix",
            "cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32>",
            func="cfgm",
        ),
    ]
    fb = FactBase("wormhole", facts)
    fb.addr32 = {"ALU_FORMAT_SPEC_REG0_SrcA_ADDR32": 1}
    shared = [
        f for f in CfgWordOverlap().run(fb) if f.hint == "CROSS_THREAD_SHARED_WORD"
    ]
    assert len(shared) == 1 and ":1" in shared[0].kind, shared


@case
def test_semaphore_excludes_wrapper_and_raii():
    Fk = "tt_llk_wormhole_b0/common/inc/ckernel.h"
    Fg = "tt_llk_wormhole_b0/common/inc/ckernel_mutex_guard.h"
    facts = [
        fn("t6_mutex_acquire", Fk, 320, 332),
        fn("T6MutexLockGuard", Fg, 15, 20),
        fn("~T6MutexLockGuard", Fg, 22, 26),
        call(Fk, 330, "t6_mutex_acquire", func="t6_mutex_acquire"),  # wrapper def
        call(Fg, 19, "t6_mutex_acquire", func="T6MutexLockGuard"),  # RAII ctor
        call(Fg, 24, "t6_mutex_release", func="~T6MutexLockGuard"),  # RAII dtor
        macro(Fk, 200, "TTI_SEMINIT", "TTI_SEMINIT(1,0,S)", func="init_fn"),
        fn("init_fn", Fk, 195, 205),
    ]
    out = SemaphoreHandshake().run(FactBase("wormhole", facts))
    assert out == [], f"wrapper-def/RAII must not be flagged: {[f.detail for f in out]}"


@case
def test_semaphore_real_imbalance_flagged():
    F = "tt_llk_wormhole_b0/llk_lib/llk_pack_common.h"
    facts = [
        fn("use", F, 200, 260),
        call(F, 210, "t6_mutex_acquire", func="use"),  # 1 acquire, 0 release
        macro(F, 205, "TTI_SEMINIT", "TTI_SEMINIT(1,0,S)", func="use"),
    ]
    out = SemaphoreHandshake().run(FactBase("wormhole", facts))
    assert any(f.hint == "MUTEX_IMBALANCE" for f in out), out


@case
def test_reconfig_setdmareg_not_the_write_and_latched_allowlist():
    # program_packer_destination: SETDMAREG (GPR) first, then latched L1_Dest_addr
    # REG2FLOP -> must NOT be flagged.
    F = "tt_llk_wormhole_b0/common/inc/cpack_common.h"
    facts = [
        fn("program_packer_destination", F, 860, 885),
        macro(
            F,
            866,
            "TT_SETDMAREG",
            "TT_SETDMAREG(0, x, 0, OUTPUT_ADDR)",
            func="program_packer_destination",
        ),
        macro(
            F,
            877,
            "TTI_REG2FLOP",
            "TTI_REG2FLOP(1,0,0,0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, OUTPUT_ADDR)",
            func="program_packer_destination",
        ),
    ]
    out = ReconfigStall().run(FactBase("wormhole", facts))
    assert (
        out == []
    ), f"latched L1_Dest_addr must not be flagged: {[f.detail for f in out]}"


@case
def test_reconfig_no_unit_drain_flagged():
    F = "tt_llk_wormhole_b0/common/inc/cpack_common.h"
    facts = [
        fn("set_packer_strides", F, 400, 440),
        macro(
            F,
            418,
            "TTI_WRCFG",
            "TTI_WRCFG(TMP0, WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32)",
            func="set_packer_strides",
        ),
    ]
    out = ReconfigStall().run(FactBase("wormhole", facts))
    assert len(out) == 1 and out[0].hint == "NO_UNIT_DRAIN", out


# --- Copilot review regressions -------------------------------------------


@case
def test_reconfig_stallwait_wait_operand_only():
    # #3: STALLWAIT(STALL_UNPACK, TRISC_CFG) — the UNPACK is the stall_res (1st
    # operand), the wait_res is TRISC_CFG. It must NOT count as a unit drain.
    F = "tt_llk_wormhole_b0/llk_lib/llk_unpack_reconfig.h"
    facts = [
        fn("_llk_unpack_reconfig_data_format_", F, 100, 200),
        macro(
            F,
            110,
            "TTI_STALLWAIT",
            "TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG)",
            func="_llk_unpack_reconfig_data_format_",
        ),
        macro(
            F,
            120,
            "TTI_WRCFG",
            "TTI_WRCFG(TMP, WRCFG_32b, SOME_ADDR32)",
            func="_llk_unpack_reconfig_data_format_",
        ),
    ]
    out = ReconfigStall().run(FactBase("wormhole", facts))
    assert len(out) == 1 and out[0].hint == "THCON_ONLY", out
    # positive control: a real unit drain in the wait_res -> not flagged
    facts[1] = macro(
        F,
        110,
        "TTI_STALLWAIT",
        "TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK)",
        func="_llk_unpack_reconfig_data_format_",
    )
    assert ReconfigStall().run(FactBase("wormhole", facts)) == []


@case
def test_mmio_guard_after_consumer_is_not_ordering():
    # #4: write, then a consumer, then a TRISC_CFG stall. The stall follows the
    # consumer, so it cannot order that consumer -> NO_LOCAL_ORDERING.
    F = "tt_llk_blackhole/llk_lib/experimental/llk_unpack_A_topk_xl_copy.h"
    facts = [
        fn("_llk_unpack_A_topk_", F, 100, 200),
        pw(F, 110, "get_cfg_pointer", "SOME_ADDR32", func="_llk_unpack_A_topk_"),
        macro(
            F,
            120,
            "TTI_UNPACR_NOP",
            "TTI_UNPACR_NOP(SrcA, x)",
            func="_llk_unpack_A_topk_",
        ),
        macro(
            F,
            130,
            "TTI_STALLWAIT",
            "TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG)",
            func="_llk_unpack_A_topk_",
        ),
    ]
    out = MmioRace().run(FactBase("blackhole", facts))
    assert len(out) == 1 and out[0].hint == "NO_LOCAL_ORDERING", out
    # control: guard BEFORE the consumer -> ordered (reassign both slots)
    facts[2] = macro(
        F,
        120,
        "TTI_STALLWAIT",
        "TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG)",
        func="_llk_unpack_A_topk_",
    )
    facts[3] = macro(
        F, 130, "TTI_UNPACR_NOP", "TTI_UNPACR_NOP(SrcA, x)", func="_llk_unpack_A_topk_"
    )
    assert MmioRace().run(FactBase("blackhole", facts))[0].hint == "LOCALLY_ORDERED"


@case
def test_cfg_word_overlap_includes_cfg_rmw_calls():
    # #5: cfg_rmw / cfg_rmw_gpr must enter the shared-word map.
    Fu = "tt_llk_wormhole_b0/common/inc/cunpack_common.h"
    Fp = "tt_llk_wormhole_b0/common/inc/cpack_common.h"
    facts = [
        fn("u", Fu, 10, 20),
        fn("p", Fp, 30, 40),
        call(Fu, 12, "cfg_rmw", func="u", arg0="SHARED_FIELD_RMW"),
        call(Fp, 32, "cfg_rmw_gpr", func="p", arg0="SHARED_FIELD_RMW"),
    ]
    fb = FactBase("wormhole", facts)
    fb.addr32 = {"SHARED_FIELD_ADDR32": 7}  # _RMW alias -> _ADDR32
    shared = [
        f for f in CfgWordOverlap().run(fb) if f.hint == "CROSS_THREAD_SHARED_WORD"
    ]
    assert len(shared) == 1 and ":7" in shared[0].kind, shared


@case
def test_cfg_word_overlap_cfg_rmw_runtime_arg_is_unresolved():
    # #5b: cfg_rmw with a runtime address variable -> surfaced as UNRESOLVED.
    F = "tt_llk_wormhole_b0/common/inc/cpack_common.h"
    facts = [fn("f", F, 10, 20), call(F, 12, "cfg_rmw", func="f", arg0="cfg_addr32")]
    out = CfgWordOverlap().run(FactBase("wormhole", facts))
    assert any(f.hint == "UNRESOLVED" for f in out), out


@case
def test_cfg_word_overlap_unresolved_ordered_write():
    # #6: an ordered-write macro whose ADDR32 isn't in cfg_defines -> UNRESOLVED
    # (not silently dropped).
    F = "tt_llk_wormhole_b0/llk_lib/llk_math_common.h"
    facts = [
        fn("m", F, 10, 20),
        macro(F, 12, "TTI_SETC16", "TTI_SETC16(NEWLY_ADDED_FIELD_ADDR32, x)", func="m"),
    ]
    fb = FactBase("wormhole", facts)
    fb.addr32 = {}  # field not defined -> drift
    out = CfgWordOverlap().run(fb)
    assert any(
        f.hint == "UNRESOLVED" and "NEWLY_ADDED_FIELD" in f.detail for f in out
    ), out


@case
def test_semaphore_wait_without_init_per_identity():
    # #2: a concrete wait on SEM_X with only a concrete init of SEM_Y (no generic
    # init) -> flagged; matching init -> not flagged; generic init -> wildcard.
    F = "tt_llk_wormhole_b0/common/inc/cmath_common.h"

    def base(waited, inited, generic=False):
        f = [fn("use", F, 100, 200), fn("init_fn", F, 50, 80)]
        f.append(
            call(
                F,
                110,
                "t6_semaphore_wait_on_zero",
                func="use",
                arg0=f"semaphore::{waited}",
            )
        )
        if generic:
            f.append(
                macro(
                    F,
                    60,
                    "TTI_SEMINIT",
                    "TTI_SEMINIT(max_value, 0, semaphore::t6_sem(index))",
                    func="init_fn",
                )
            )
        else:
            f.append(
                macro(
                    F,
                    60,
                    "TTI_SEMINIT",
                    f"TTI_SEMINIT(1, 0, p_stall::{inited})",
                    func="init_fn",
                )
            )
        return FactBase("wormhole", f)

    # mismatched concrete init, no generic -> flagged, higher confidence (no tag)
    out = SemaphoreHandshake().run(base("FPU_SFPU", "SEMAPHORE_1"))
    ww = [f for f in out if f.hint == "WAIT_WITHOUT_INIT"]
    assert len(ww) == 1 and ww[0].safety == "", ww
    # matching concrete init -> not flagged
    out = SemaphoreHandshake().run(base("SEMAPHORE_1", "SEMAPHORE_1"))
    assert not any(f.hint == "WAIT_WITHOUT_INIT" for f in out), out
    # generic (parameterized) init present -> EMITTED as LOW_CONFIDENCE, not
    # globally suppressed (the generic t6_sem(index) init may cover it, but a
    # recall augmentor must not drop the candidate).
    out = SemaphoreHandshake().run(base("FPU_SFPU", "SEMAPHORE_1", generic=True))
    ww = [f for f in out if f.hint == "WAIT_WITHOUT_INIT"]
    assert len(ww) == 1 and ww[0].safety == "LOW_CONFIDENCE", ww


@case
def test_reconfig_pack_not_drained_by_unpack_condition():
    # 'PACK' is a substring of 'UNPACK'. A PACK-thread reconfig whose STALLWAIT
    # condition drains only the UNPACKER must NOT be treated as PACK-drained.
    F = "tt_llk_wormhole_b0/common/inc/cpack_common.h"
    facts = [
        fn("set_packer_config", F, 100, 200),
        macro(
            F,
            110,
            "TTI_STALLWAIT",
            "TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK)",
            func="set_packer_config",
        ),
        macro(
            F,
            120,
            "TTI_WRCFG",
            "TTI_WRCFG(TMP, WRCFG_32b, SOME_ADDR32)",
            func="set_packer_config",
        ),
    ]
    out = ReconfigStall().run(FactBase("wormhole", facts))
    assert len(out) == 1 and out[0].hint == "THCON_ONLY", out  # NOT silently drained
    # a genuine PACK drain in the condition -> not flagged
    facts[1] = macro(
        F,
        110,
        "TTI_STALLWAIT",
        "TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK)",
        func="set_packer_config",
    )
    assert ReconfigStall().run(FactBase("wormhole", facts)) == []


# --- Quasar validation regressions ----------------------------------------


@case
def test_cfg_word_cfg_rmw_via_rmw_macro_and_no_empty_noise():
    # Quasar writes config via cfg_rmw(FIELD_RMW,...); FIELD_RMW is captured as an
    # object-macro fact. Two threads on the same word -> CROSS_THREAD; the empty
    # arg0 cfg_rmw call must NOT emit a '?' UNRESOLVED.
    Fp = "tt_llk_quasar/common/inc/cpack_common.h"
    Fm = "tt_llk_quasar/common/inc/cmath_common.h"
    facts = [
        fn("p", Fp, 10, 20),
        fn("m", Fm, 30, 40),
        macro(Fp, 12, "SHARED_FIELD_RMW", "SHARED_FIELD_RMW", func="p"),
        macro(Fm, 32, "SHARED_FIELD_RMW", "SHARED_FIELD_RMW", func="m"),
        call(Fp, 13, "cfg_rmw", func="p", arg0=""),  # macro-expanded arg -> empty
    ]
    fb = FactBase("quasar", facts)
    fb.addr32 = {"SHARED_FIELD_ADDR32": 5}
    out = CfgWordOverlap().run(fb)
    shared = [f for f in out if f.hint == "CROSS_THREAD_SHARED_WORD"]
    assert len(shared) == 1 and ":5" in shared[0].kind, shared
    assert all(
        "could not resolve ?" not in f.detail for f in out
    ), "empty cfg_rmw noise"


@case
def test_mmio_quasar_autottsync_ordered():
    # On Quasar the per-RISC TTSync HW-orders the write->consume direction, so an
    # unguarded cfg write is AUTOTTSYNC_ORDERED, not a race candidate. Same write
    # on WH stays NO_LOCAL_ORDERING.
    F = "tt_llk_quasar/common/inc/ckernel_trisc_common.h"
    facts = [
        fn("f", F, 100, 200),
        pw(F, 110, "get_cfg_pointer", "SOME_ADDR32", func="f"),
    ]
    assert MmioRace().run(FactBase("quasar", facts))[0].hint == "AUTOTTSYNC_ORDERED"
    FW = "tt_llk_wormhole_b0/common/inc/cpack_common.h"
    facts_wh = [
        fn("f", FW, 100, 200),
        pw(FW, 110, "get_cfg_pointer", "SOME_ADDR32", func="f"),
    ]
    assert MmioRace().run(FactBase("wormhole", facts_wh))[0].hint == "NO_LOCAL_ORDERING"


@case
def test_reconfig_detects_cfg_rmw_write():
    # A reconfig whose config write is a cfg_rmw call (Quasar's idiom) must be
    # detected, not skipped as "writes nothing".
    F = "tt_llk_quasar/common/inc/cpack_common.h"
    facts = [
        fn("_llk_pack_reconfig_data_format_", F, 100, 200),
        call(F, 120, "cfg_rmw", func="_llk_pack_reconfig_data_format_", arg0=""),
    ]
    out = ReconfigStall().run(FactBase("quasar", facts))
    assert len(out) == 1 and out[0].hint == "NO_UNIT_DRAIN", out


@case
def test_cfg_word_intra_thread_full_word_clobber():
    # Same thread: a full-word cfg[]= write of field A to word W, and a masked
    # cfg_reg_rmw_tensix of sibling field B (same word) elsewhere -> the full-word
    # write may zero B. Flagged INTRA_THREAD_CLOBBER.
    F = "tt_llk_wormhole_b0/common/inc/cpack_common.h"  # PACK thread
    facts = [
        fn("f", F, 100, 300),
        pw(F, 110, "get_cfg_pointer", "FIELD_A_ADDR32", func="f"),  # full-word cfg[]=
        call(
            F, 200, "cfg_reg_rmw_tensix", "cfg_reg_rmw_tensix<FIELD_B_ADDR32>", func="f"
        ),  # masked sibling
    ]
    fb = FactBase("wormhole", facts)
    fb.addr32 = {"FIELD_A_ADDR32": 9, "FIELD_B_ADDR32": 9}  # same word
    out = CfgWordOverlap().run(fb)
    clob = [f for f in out if f.hint == "INTRA_THREAD_CLOBBER"]
    assert len(clob) == 1 and "clobber@CONFIG:9" == clob[0].kind, clob
    assert "FIELD_B" in clob[0].detail, clob[0].detail
    # Control: a lone full-word write with NO masked sibling -> not a clobber.
    facts2 = [
        fn("g", F, 100, 300),
        pw(F, 110, "get_cfg_pointer", "FIELD_A_ADDR32", func="g"),
    ]
    fb2 = FactBase("wormhole", facts2)
    fb2.addr32 = {"FIELD_A_ADDR32": 9}
    assert not any(f.hint == "INTRA_THREAD_CLOBBER" for f in CfgWordOverlap().run(fb2))


@case
def test_cfg_word_reg2flop_sibling_is_clobber_victim():
    # A full-word cfg[]= of field A + a NON-RMW field-scoped TTI_REG2FLOP of sibling
    # B in the SAME word/thread: the REG2FLOP set is a clobber victim just like a
    # masked RMW (the victim bucket is any resolved-field non-full-word write, not
    # only the RMW helpers — cpack_common.h writes THCON_SEC0_REG1 both ways).
    F = "tt_llk_wormhole_b0/common/inc/cpack_common.h"  # PACK thread
    facts = [
        fn("f", F, 100, 300),
        pw(F, 110, "get_cfg_pointer", "FIELD_A_ADDR32", func="f"),  # full-word cfg[]=
        macro(
            F,
            200,
            "TTI_REG2FLOP",
            "TTI_REG2FLOP(1,0,0,0, FIELD_B_ADDR32 - THCON_CFGREG_BASE_ADDR32, OUT)",
            func="f",
        ),  # field-scoped sibling, NOT an RMW helper
    ]
    fb = FactBase("wormhole", facts)
    fb.addr32 = {"FIELD_A_ADDR32": 9, "FIELD_B_ADDR32": 9}  # same word
    clob = [x for x in CfgWordOverlap().run(fb) if x.hint == "INTRA_THREAD_CLOBBER"]
    assert len(clob) == 1 and clob[0].kind == "clobber@CONFIG:9", clob
    assert "FIELD_B" in clob[0].detail, clob[0].detail


@case
def test_cfg_word_safety_annotation_kept_finding():
    # Cross-thread word is ALWAYS reported; `safety` is a sub-annotation.
    Fu = "tt_llk_wormhole_b0/common/inc/cunpack_common.h"  # UNPACK
    Fp = "tt_llk_wormhole_b0/common/inc/cpack_common.h"  # PACK

    def run(fa_mask, fb_mask, pack_full_word=False):
        facts = [
            fn("u", Fu, 10, 20),
            fn("p", Fp, 30, 40),
            call(
                Fu, 12, "cfg_reg_rmw_tensix", "cfg_reg_rmw_tensix<FA_ADDR32>", func="u"
            ),
        ]
        if pack_full_word:
            facts.append(pw(Fp, 32, "get_cfg_pointer", "FB_ADDR32", func="p"))
        else:
            facts.append(
                call(
                    Fp,
                    32,
                    "cfg_reg_rmw_tensix",
                    "cfg_reg_rmw_tensix<FB_ADDR32>",
                    func="p",
                )
            )
        fb = FactBase("wormhole", facts)
        fb.addr32 = {
            "FA_ADDR32": 5,
            "FB_ADDR32": 5,
            "FA_MASK": fa_mask,
            "FB_MASK": fb_mask,
        }
        return [
            f for f in CfgWordOverlap().run(fb) if f.hint == "CROSS_THREAD_SHARED_WORD"
        ]

    # disjoint bits, both atomic RMWCIB -> SAFE_BY_MASKING, still reported
    out = run(0x1, 0x3C)
    assert len(out) == 1 and out[0].safety == "SAFE_BY_MASKING", out
    # overlapping bits -> POTENTIAL_CLOBBER (still reported)
    out = run(0x3, 0x1)
    assert len(out) == 1 and out[0].safety == "POTENTIAL_CLOBBER", out
    # a full-word writer -> POTENTIAL_CLOBBER even if masks look disjoint
    out = run(0x1, 0xFFFFFFFF, pack_full_word=True)
    assert len(out) == 1 and out[0].safety == "POTENTIAL_CLOBBER", out


@case
def test_cfg_word_dedup_double_captured_rmw():
    # cfg_reg_rmw_tensix<FA_RMW> is captured as BOTH a call and the FA_RMW macro
    # at the same site; dedup -> one atomic writer, so a disjoint pair stays SAFE.
    Fu = "tt_llk_wormhole_b0/common/inc/cunpack_common.h"
    Fp = "tt_llk_wormhole_b0/common/inc/cpack_common.h"
    facts = [
        fn("u", Fu, 10, 20),
        fn("p", Fp, 30, 40),
        call(Fu, 12, "cfg_reg_rmw_tensix", "cfg_reg_rmw_tensix<FA_RMW>", func="u"),
        macro(Fu, 12, "FA_RMW", "FA_RMW", func="u"),  # duplicate capture of same write
        call(Fp, 32, "cfg_reg_rmw_tensix", "cfg_reg_rmw_tensix<FB_RMW>", func="p"),
        macro(Fp, 32, "FB_RMW", "FB_RMW", func="p"),
    ]
    fb = FactBase("wormhole", facts)
    fb.addr32 = {"FA_ADDR32": 5, "FB_ADDR32": 5, "FA_MASK": 0x1, "FB_MASK": 0x2}
    out = [f for f in CfgWordOverlap().run(fb) if f.hint == "CROSS_THREAD_SHARED_WORD"]
    assert len(out) == 1 and out[0].safety == "SAFE_BY_MASKING", out
    assert len(out[0].evidence) == 2, out[0].evidence  # not double-listed


# --- srcreg-bank ----------------------------------------------------------


@case
def test_srcreg_raw_setdvalid_blackhole_flagged():
    # A raw TTI_SETDVALID on Blackhole is ISA-unsupported -> RAW_SETDVALID_BH.
    F = "tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_datacopy.h"
    facts = [
        fn("_llk_math_eltwise_unary_datacopy_", F, 100, 300),
        macro(F, 109, "TTI_SETDVALID", "TTI_SETDVALID(0b10)", func="dc"),
    ]
    out = SrcRegBank().run(FactBase("blackhole", facts))
    assert len(out) == 1 and out[0].hint == "RAW_SETDVALID_BH", out
    # Same macro on Wormhole is a normal dvalid control point, NOT the BH flag.
    out_wh = SrcRegBank().run(FactBase("wormhole", facts))
    assert len(out_wh) == 1 and out_wh[0].hint == "DVALID_SET", out_wh


@case
def test_srcreg_cleardvalid_and_supported_forms():
    F = "tt_llk_blackhole/common/inc/cunpack_common.h"
    facts = [
        fn("u", F, 100, 300),
        macro(F, 110, "TTI_CLEARDVALID", "TTI_CLEARDVALID(0b01)", func="u"),
        # UNPACR_NOP(...,SET_DVALID,...) is the SUPPORTED set form -> not a raw
        # SETDVALID, must NOT be flagged.
        macro(F, 120, "TTI_UNPACR_NOP", "TTI_UNPACR_NOP(SrcA, SET_DVALID)", func="u"),
        # TT_OP_SETDVALID is the opcode-VALUE constant, not an issue -> excluded.
        macro(F, 130, "TT_OP_SETDVALID", "", func="u"),
    ]
    out = SrcRegBank().run(FactBase("blackhole", facts))
    assert len(out) == 1 and out[0].hint == "DVALID_CLEAR", out


# --- mailbox-sync (lite) --------------------------------------------------


@case
def test_mailbox_paired_channel():
    # cmath writes dst_index to UNPACK (MATH->UNPACK); cunpack reads from MATH
    # (MATH->UNPACK) -> same resolved channel -> both PAIRED.
    Fm = "tt_llk_wormhole_b0/common/inc/cmath_common.h"
    Fu = "tt_llk_wormhole_b0/common/inc/cunpack_common.h"
    facts = [
        fn("m", Fm, 200, 260),
        fn("u", Fu, 990, 1010),
        call(Fm, 239, "mailbox_write", func="m", arg0="ThreadId::UnpackThreadId"),
        call(Fu, 1000, "mailbox_read", func="u", arg0="ThreadId::MathThreadId"),
    ]
    out = MailboxSync().run(FactBase("wormhole", facts))
    assert len(out) == 2 and all(f.hint == "PAIRED_CHANNEL" for f in out), out
    assert all("MATH->UNPACK" in f.detail for f in out), out
    # the write finding lists the read as its partner evidence
    w = next(f for f in out if f.kind == "mailbox:write")
    assert any("read" in e and "1000" in e for e in w.evidence), w.evidence


@case
def test_mailbox_unpaired_and_unresolved():
    # A write with no in-tree reader -> UNPAIRED. A debug endpoint whose issuing
    # thread isn't derivable from the file -> UNRESOLVED.
    Fm = "tt_llk_wormhole_b0/common/inc/cmath_common.h"
    Fd = "tt_llk_wormhole_b0/common/inc/ckernel_debug.h"
    facts = [
        fn("m", Fm, 200, 260),
        fn("d", Fd, 100, 160),
        call(Fm, 239, "mailbox_write", func="m", arg0="ThreadId::UnpackThreadId"),
        call(Fd, 113, "mailbox_write", func="d", arg0="ThreadId::MathThreadId"),
    ]
    out = MailboxSync().run(FactBase("wormhole", facts))
    by_line = {f.line: f.hint for f in out}
    assert by_line[239] == "UNPAIRED_ENDPOINT", out  # no reader present
    assert by_line[113] == "UNRESOLVED_ENDPOINT", out  # ckernel_debug -> no thread


@case
def test_diff_scope_filter():
    from llkaudit.cli import scope_to_changed

    findings = [
        {"file": "/x/cpack_common.h", "evidence": ["cunpack_common.h:5 [UNPACK] F"]},
        {"file": "/x/llk_math_reduce.h", "evidence": ["llk_math_reduce.h:9 [MATH] G"]},
    ]
    # no changed files -> unfiltered
    assert len(scope_to_changed(findings, [])) == 2
    # scope to cunpack: keeps the cpack finding (evidence touches cunpack), drops math
    out = scope_to_changed(findings, ["/repo/.../cunpack_common.h"])
    assert len(out) == 1 and out[0]["file"].endswith("cpack_common.h"), out
    # scope to the math file: anchor match
    out = scope_to_changed(findings, ["/repo/.../llk_math_reduce.h"])
    assert len(out) == 1 and out[0]["file"].endswith("llk_math_reduce.h"), out


# --- full-tool-review regression fixes ------------------------------------


@case
def test_cfg_word_wrcfg_128b_spans_four_words():
    # WRCFG_128b overwrites base..base+3; a sibling write to base+2 must overlap.
    Fp = "tt_llk_wormhole_b0/common/inc/cpack_common.h"  # PACK
    Fm = "tt_llk_wormhole_b0/llk_lib/llk_math_common.h"  # MATH
    facts = [
        fn("p", Fp, 100, 200),
        fn("m", Fm, 10, 30),
        macro(
            Fp,
            110,
            "TTI_WRCFG",
            "TTI_WRCFG(x, p_cfg::WRCFG_128b, BASE_ADDR32)",
            func="p",
        ),
        call(Fm, 12, "cfg_reg_rmw_tensix", "cfg_reg_rmw_tensix<SIB_ADDR32>", func="m"),
    ]
    fb = FactBase("wormhole", facts)
    fb.addr32 = {"BASE_ADDR32": 40, "SIB_ADDR32": 42}  # 128b spans 40..43
    shared = [
        f for f in CfgWordOverlap().run(fb) if f.hint == "CROSS_THREAD_SHARED_WORD"
    ]
    assert any(f.kind == "shared_word@CONFIG:42" for f in shared), shared
    # a WRCFG_32b (control) spans one word only -> no overlap at 42
    facts[2] = macro(
        Fp, 110, "TTI_WRCFG", "TTI_WRCFG(x, p_cfg::WRCFG_32b, BASE_ADDR32)", func="p"
    )
    fb2 = FactBase("wormhole", facts)
    fb2.addr32 = {"BASE_ADDR32": 40, "SIB_ADDR32": 42}
    assert not any(
        f.kind == "shared_word@CONFIG:42"
        for f in CfgWordOverlap().run(fb2)
        if f.hint == "CROSS_THREAD_SHARED_WORD"
    )


@case
def test_cfg_word_sfpu_file_attributed_to_math():
    # An SFPU file (MATH thread, no unpack/math/pack token) must be attributed to
    # MATH so its config write participates in the cross-thread analysis.
    Fu = "tt_llk_blackhole/common/inc/cunpack_common.h"  # UNPACK
    Fs = "tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_x.h"  # SFPU -> MATH
    facts = [
        fn("u", Fu, 10, 20),
        fn("s", Fs, 30, 40),
        call(Fu, 12, "cfg_reg_rmw_tensix", "cfg_reg_rmw_tensix<W_ADDR32>", func="u"),
        call(Fs, 32, "cfg_reg_rmw_tensix", "cfg_reg_rmw_tensix<W_ADDR32>", func="s"),
    ]
    fb = FactBase("blackhole", facts)
    fb.addr32 = {"W_ADDR32": 3, "W_MASK": 0x1}
    shared = [
        f for f in CfgWordOverlap().run(fb) if f.hint == "CROSS_THREAD_SHARED_WORD"
    ]
    assert len(shared) == 1, shared
    assert "MATH" in shared[0].detail and "UNPACK" in shared[0].detail, shared[0].detail


@case
def test_cfg_word_threadconfig_not_a_cross_thread_race():
    # Two threads SETC16 the same ThreadConfig word -> DIFFERENT physical
    # registers (per-thread), NOT a data race -> must not be flagged.
    Fu = "tt_llk_wormhole_b0/common/inc/cunpack_common.h"
    Fm = "tt_llk_wormhole_b0/common/inc/cmath_common.h"
    facts = [
        fn("u", Fu, 10, 20),
        fn("m", Fm, 30, 40),
        macro(Fu, 12, "TTI_SETC16", "TTI_SETC16(TC_ADDR32, x)", func="u"),
        macro(Fm, 32, "TTI_SETC16", "TTI_SETC16(TC_ADDR32, y)", func="m"),
    ]
    fb = FactBase("wormhole", facts)
    fb.addr32 = {"TC_ADDR32": 5}
    assert not any(
        f.hint == "CROSS_THREAD_SHARED_WORD" for f in CfgWordOverlap().run(fb)
    )


@case
def test_cfg_word_full_word_beats_unresolved_label():
    # A full-word writer + a sibling whose mask is unresolved must stay
    # POTENTIAL_CLOBBER (a certain clobber), not downgrade to UNKNOWN.
    Fu = "tt_llk_wormhole_b0/common/inc/cunpack_common.h"
    Fm = "tt_llk_wormhole_b0/common/inc/cmath_common.h"
    facts = [
        fn("u", Fu, 10, 20),
        fn("m", Fm, 30, 40),
        pw(Fu, 12, "get_cfg_pointer", "W_ADDR32", func="u"),  # full-word cfg[]=
        call(Fm, 32, "cfg_reg_rmw_tensix", "cfg_reg_rmw_tensix<W_ADDR32>", func="m"),
    ]
    fb = FactBase("wormhole", facts)
    fb.addr32 = {"W_ADDR32": 7}  # no W_MASK -> masked writer unresolved
    shared = [
        f for f in CfgWordOverlap().run(fb) if f.hint == "CROSS_THREAD_SHARED_WORD"
    ]
    assert len(shared) == 1 and shared[0].safety == "POTENTIAL_CLOBBER", shared


@case
def test_mailbox_query_does_not_pair_a_write():
    # A non-blocking not_empty query does NOT drain the FIFO, so it must not make
    # a writer look PAIRED (that would mask a no-reader deadlock).
    Fm = "tt_llk_wormhole_b0/common/inc/cmath_common.h"
    Fu = "tt_llk_wormhole_b0/common/inc/cunpack_common.h"
    facts = [
        fn("m", Fm, 10, 20),
        fn("u", Fu, 30, 40),
        call(Fm, 12, "mailbox_write", func="m", arg0="ThreadId::UnpackThreadId"),
        call(Fu, 32, "mailbox_not_empty", func="u", arg0="ThreadId::MathThreadId"),
    ]
    out = MailboxSync().run(FactBase("wormhole", facts))
    w = next(f for f in out if f.kind == "mailbox:write")
    assert w.hint == "UNPAIRED_ENDPOINT", w  # query alone does not pair a write
    q = next(f for f in out if f.kind == "mailbox:query")
    assert q.hint == "PAIRED_CHANNEL", q  # the query IS satisfied by the writer


@case
def test_reconfig_latched_first_then_undrained_sampled():
    # A latched write FIRST must not abandon the whole function: a later
    # undrained SAMPLED write must still be flagged.
    F = "tt_llk_wormhole_b0/common/inc/cpack_common.h"  # PACK
    facts = [
        fn("program_packer_reconfig", F, 100, 300),
        macro(
            F,
            110,
            "TTI_REG2FLOP",
            "TTI_REG2FLOP(1,0,0,0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32 - X, Y)",
            func="program_packer_reconfig",
        ),
        macro(
            F,
            150,
            "TTI_WRCFG",
            "TTI_WRCFG(TMP, WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32)",
            func="program_packer_reconfig",
        ),
    ]
    out = ReconfigStall().run(FactBase("wormhole", facts))
    assert len(out) == 1 and out[0].hint == "NO_UNIT_DRAIN" and out[0].line == 150, out


@case
def test_reconfig_drain_invalidated_by_reissue():
    # drain -> write (ok) -> PACR re-arms PACK -> write (flagged DRAIN_REARMED).
    F = "tt_llk_wormhole_b0/common/inc/cpack_common.h"  # PACK
    facts = [
        fn("set_packer_config", F, 100, 300),
        macro(
            F,
            110,
            "TTI_STALLWAIT",
            "TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK)",
            func="set_packer_config",
        ),
        macro(
            F,
            120,
            "TTI_WRCFG",
            "TTI_WRCFG(a, WRCFG_32b, WA_ADDR32)",
            func="set_packer_config",
        ),
        macro(
            F, 130, "TTI_PACR", "TTI_PACR(0)", func="set_packer_config"
        ),  # re-arms PACK
        macro(
            F,
            140,
            "TTI_WRCFG",
            "TTI_WRCFG(b, WRCFG_32b, WB_ADDR32)",
            func="set_packer_config",
        ),
    ]
    out = ReconfigStall().run(FactBase("wormhole", facts))
    assert len(out) == 1 and out[0].hint == "DRAIN_REARMED" and out[0].line == 140, out


@case
def test_mmio_matrix_op_is_a_consumer():
    # A matrix issue (TTI_MVMUL) consumes config; a guard placed AFTER it cannot
    # un-race it -> NO_LOCAL_ORDERING.
    F = "tt_llk_wormhole_b0/common/inc/cmath_common.h"
    facts = [
        fn("f", F, 100, 200),
        pw(F, 110, "get_cfg_pointer", "ALU_ADDR32", func="f"),
        macro(F, 120, "TTI_MVMUL", "TTI_MVMUL(0)", func="f"),  # consumer_math
        macro(
            F,
            130,
            "TTI_STALLWAIT",
            "TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::TRISC_CFG)",
            func="f",
        ),
    ]
    out = MmioRace().run(FactBase("wormhole", facts))
    assert len(out) == 1 and out[0].hint == "NO_LOCAL_ORDERING", out
    # control: guard BEFORE the matrix op -> LOCALLY_ORDERED
    facts[2] = macro(
        F,
        120,
        "TTI_STALLWAIT",
        "TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::TRISC_CFG)",
        func="f",
    )
    facts[3] = macro(F, 130, "TTI_MVMUL", "TTI_MVMUL(0)", func="f")
    assert MmioRace().run(FactBase("wormhole", facts))[0].hint == "LOCALLY_ORDERED"


@case
def test_mmio_mop_sync_and_tensix_sync_order_writes():
    F = "tt_llk_wormhole_b0/common/inc/cunpack_common.h"
    for drain in ("mop_sync", "tensix_sync"):
        facts = [
            fn("f", F, 100, 200),
            pw(F, 110, "get_cfg_pointer", "SOME_ADDR32", func="f"),
            call(F, 120, drain, func="f"),
        ]
        out = MmioRace().run(FactBase("wormhole", facts))
        assert out[0].hint == "LOCALLY_ORDERED", (drain, out)


@case
def test_diff_scope_none_sentinel():
    # run.sh passes --changed-files __none__ when no LLK header changed; it must
    # scope to nothing (empty), not everything.
    from llkaudit.cli import scope_to_changed

    findings = [{"file": "/x/cpack_common.h", "evidence": ["cpack_common.h:5 F"]}]
    assert scope_to_changed(findings, ["__none__"]) == []


@case
def test_is_ctor_or_dtor_tightened():
    from llkaudit import registry

    assert registry.is_ctor_or_dtor("~T6MutexLockGuard")  # dtor
    assert registry.is_ctor_or_dtor("T6MutexLockGuard")  # CamelCase ctor/type
    assert not registry.is_ctor_or_dtor("set_packer_config")  # snake fn
    assert not registry.is_ctor_or_dtor("Process_tile")  # Capitalized_snake kept
    assert not registry.is_ctor_or_dtor("")


@case
def test_cli_empty_checks_is_hard_error():
    import os
    import tempfile

    from llkaudit import cli

    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    with open(path, "w") as f:
        f.write('{"arch":"wormhole","parse_errors":0,"facts":[]}')
    try:
        for bad in ("", ",", " "):
            try:
                cli.main(["--arch", "wormhole", "--facts", path, "--checks", bad])
                assert False, f"empty --checks {bad!r} should have errored"
            except SystemExit as e:
                assert e.code != 0
    finally:
        os.remove(path)


@case
def test_cli_empty_fact_base_marks_degraded():
    # The CLI is a documented standalone entry point; run on an empty / failed facts
    # file it must NOT read as a clean audit — the envelope is marked `degraded`.
    import io
    import json as _json
    import os
    import tempfile
    from contextlib import redirect_stdout

    from llkaudit import cli

    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    with open(path, "w") as f:
        f.write('{"arch":"wormhole","parse_errors":0,"facts":[]}')
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli.main(
                ["--arch", "wormhole", "--facts", path, "--checks", "mmio-race"]
            )
        out = _json.loads(buf.getvalue())
        assert rc == 0  # still emits the envelope (contract) — but marked degraded
        assert "degraded" in out and any(
            "empty fact base" in d for d in out["degraded"]
        ), out
        assert out["checks"]["mmio-race"]["count"] == 0
    finally:
        os.remove(path)


# --- cb-sync / noc-sync (kernel tier — deterministic, fed a KERNEL fact base) ---


@case
def test_cb_sync_reserve_push_imbalance():
    K = "ttnn/cpp/.../writer_kernel.cpp"
    # balanced reserve/push -> no finding; a reserve with no push -> imbalance.
    balanced = [
        fn("kernel_main", K, 0, 100),
        call(K, 10, "cb_reserve_back", func="kernel_main", arg0="cb_out0"),
        call(K, 20, "cb_push_back", func="kernel_main", arg0="cb_out0"),
    ]
    assert CbSync().run(FactBase("wormhole", balanced)) == []
    leaked = [
        fn("kernel_main", K, 0, 100),
        call(K, 10, "cb_reserve_back", func="kernel_main", arg0="cb_out0"),
        call(K, 20, "cb_reserve_back", func="kernel_main", arg0="cb_out0"),
        call(K, 30, "cb_push_back", func="kernel_main", arg0="cb_out0"),
    ]
    out = CbSync().run(FactBase("wormhole", leaked))
    assert len(out) == 1 and out[0].hint == "CB_RESERVE_PUSH_IMBALANCE", out
    assert "cb_out0" in out[0].detail


@case
def test_cb_sync_wait_pop_imbalance_and_per_cb():
    K = "ttnn/cpp/.../reader_kernel.cpp"
    facts = [
        fn("kernel_main", K, 0, 100),
        # cb_in0: wait without pop -> imbalance
        call(K, 10, "cb_wait_front", func="kernel_main", arg0="cb_in0"),
        # cb_in1: balanced -> no finding (per-CB grouping)
        call(K, 40, "cb_wait_front", func="kernel_main", arg0="cb_in1"),
        call(K, 50, "cb_pop_front", func="kernel_main", arg0="cb_in1"),
    ]
    out = CbSync().run(FactBase("wormhole", facts))
    assert len(out) == 1 and out[0].hint == "CB_WAIT_POP_IMBALANCE", out
    assert "cb_in0" in out[0].kind


@case
def test_noc_sync_signal_without_flush():
    K = "ttnn/cpp/.../writer_kernel.cpp"
    # flush before the signal -> ok; signal with no preceding flush -> flagged.
    ok = [
        fn("kernel_main", K, 0, 100),
        call(K, 10, "noc_async_write_barrier", func="kernel_main"),
        call(K, 20, "noc_semaphore_inc", func="kernel_main", arg0="sem_addr"),
    ]
    assert NocSync().run(FactBase("wormhole", ok)) == []
    bad = [
        fn("kernel_main", K, 0, 100),
        call(K, 20, "noc_semaphore_inc", func="kernel_main", arg0="sem_addr"),
        call(K, 30, "noc_async_write_barrier", func="kernel_main"),  # AFTER — too late
    ]
    out = NocSync().run(FactBase("wormhole", bad))
    assert len(out) == 1 and out[0].hint == "NOC_SIGNAL_NO_FLUSH", out


@case
def test_cb_noc_empty_over_tt_llk():
    # No cb_*/noc_* calls in a tt-llk fact base -> both trivially empty.
    F = "tt_llk_wormhole_b0/llk_lib/llk_unpack_A.h"
    facts = [
        fn("_llk_unpack_A_", F, 0, 50),
        macro(F, 10, "TTI_UNPACR", "TTI_UNPACR(0)", func="_llk_unpack_A_"),
    ]
    assert CbSync().run(FactBase("wormhole", facts)) == []
    assert NocSync().run(FactBase("wormhole", facts)) == []


# --- bot-review follow-up fixes -------------------------------------------


@case
def test_cli_list_without_required_args():
    # `--list` must work WITHOUT --arch/--facts (they are validated only for an
    # actual audit) — previously it exited 2 on the missing required args.
    import contextlib
    import io

    from llkaudit import cli

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli.main(["--list"])
    assert rc == 0, rc
    assert "mmio-race" in buf.getvalue() and "cb-sync" in buf.getvalue()


@case
def test_cfg_word_nonliteral_index_surfaces_unresolved():
    # A config write through a NON-LITERAL index (cfg[upk0_reg]=) can't resolve to
    # a word; it must surface as UNRESOLVED, not be silently dropped.
    F = "tt_llk_wormhole_b0/llk_lib/llk_unpack_A.h"
    facts = [
        fn("_llk_unpack_A_", F, 100, 200),
        pw(F, 110, "get_cfg_pointer", "upk0_reg", func="_llk_unpack_A_"),
    ]
    out = CfgWordOverlap().run(FactBase("wormhole", facts))
    assert any(f.hint == "UNRESOLVED" for f in out), out


# --- package-review follow-up (registry primitive completeness) -----------


@case
def test_noc_sync_remote_signals_and_local_set():
    K = "ttnn/cpp/x/writer.cpp"
    # remote credit signals with no preceding flush -> flagged
    for sig in ("noc_semaphore_inc_multicast", "noc_semaphore_set_remote"):
        facts = [
            fn("kernel_main", K, 0, 100),
            call(K, 20, sig, func="kernel_main", arg0="sem"),
        ]
        out = NocSync().run(FactBase("wormhole", facts))
        assert len(out) == 1 and out[0].hint == "NOC_SIGNAL_NO_FLUSH", (sig, out)
    # plain noc_semaphore_set is a LOCAL reset, NOT a remote signal -> not flagged
    facts = [
        fn("kernel_main", K, 0, 100),
        call(K, 20, "noc_semaphore_set", func="kernel_main", arg0="sem"),
    ]
    assert NocSync().run(FactBase("wormhole", facts)) == []
    # a bare FLUSHED (not an ack barrier) does NOT clear an ATOMIC inc (conservative:
    # departure != commit) -> still flagged, but TAGGED FLUSH_NOT_BARRIER since a flush
    # is present (for the skill to confirm vs the NoC Ordering + posted_writes docs,
    # not a pre-declared safe idiom).
    facts = [
        fn("kernel_main", K, 0, 100),
        call(K, 10, "noc_async_write_flushed_with_trid", func="kernel_main"),
        call(K, 20, "noc_semaphore_inc", func="kernel_main", arg0="sem"),
    ]
    r = NocSync().run(FactBase("wormhole", facts))
    assert r[0].hint == "NOC_SIGNAL_NO_FLUSH" and r[0].safety == "FLUSH_NOT_BARRIER"
    # a write BARRIER before the atomic inc DOES clear it (data landed).
    facts = [
        fn("kernel_main", K, 0, 100),
        call(K, 10, "noc_async_write_barrier", func="kernel_main"),
        call(K, 20, "noc_semaphore_inc", func="kernel_main", arg0="sem"),
    ]
    assert NocSync().run(FactBase("wormhole", facts)) == []
    # a WRITE credit (set_remote) IS ordered by a bare flushed (it is a write).
    facts = [
        fn("kernel_main", K, 0, 100),
        call(K, 10, "noc_async_writes_flushed", func="kernel_main"),
        call(K, 20, "noc_semaphore_set_remote", func="kernel_main", arg0="sem"),
    ]
    assert NocSync().run(FactBase("wormhole", facts)) == []


@case
def test_cb_sync_remote_cb_family():
    K = "ttnn/cpp/x/remote_writer.cpp"
    facts = [
        fn("kernel_main", K, 0, 100),
        call(K, 10, "remote_cb_reserve_back", func="kernel_main", arg0="cb_r"),
        call(K, 20, "remote_cb_reserve_back", func="kernel_main", arg0="cb_r"),
        call(
            K,
            30,
            "remote_cb_push_back_and_write_pages",
            func="kernel_main",
            arg0="cb_r",
        ),
    ]
    out = CbSync().run(FactBase("wormhole", facts))
    assert len(out) == 1 and out[0].hint == "CB_RESERVE_PUSH_IMBALANCE", out


# --- object/method CB/NoC API (modern ttnn kernels) -----------------------


@case
def test_cb_sync_method_api_grouped_by_receiver():
    # cb_obj.reserve_back()/push_back() on a CircularBuffer: cb id is the RECEIVER,
    # not arg0 (the count). An imbalance per receiver is flagged; a std::vector
    # push_back must NOT be mistaken for a CB push.
    K = "ttnn/cpp/x/reader.cpp"
    facts = [
        fn("kernel_main", K, 0, 100),
        call(
            K,
            10,
            "reserve_back",
            func="kernel_main",
            arg0="1",
            recv="cb_in0",
            recv_type="CircularBuffer",
        ),
        call(
            K,
            20,
            "reserve_back",
            func="kernel_main",
            arg0="1",
            recv="cb_in0",
            recv_type="CircularBuffer",
        ),
        call(
            K,
            30,
            "push_back",
            func="kernel_main",
            arg0="1",
            recv="cb_in0",
            recv_type="CircularBuffer",
        ),
        # a std::vector push_back — different receiver type, must be ignored
        call(
            K,
            40,
            "push_back",
            func="kernel_main",
            arg0="x",
            recv="vec",
            recv_type="vector<int>",
        ),
        # a DIFFERENT cb, balanced -> no finding for it (per-receiver grouping)
        call(
            K,
            50,
            "wait_front",
            func="kernel_main",
            arg0="1",
            recv="cb_out",
            recv_type="CircularBuffer",
        ),
        call(
            K,
            60,
            "pop_front",
            func="kernel_main",
            arg0="1",
            recv="cb_out",
            recv_type="CircularBuffer",
        ),
    ]
    out = CbSync().run(FactBase("wormhole", facts))
    assert len(out) == 1 and out[0].hint == "CB_RESERVE_PUSH_IMBALANCE", out
    assert "cb_in0" in out[0].kind, out[0].kind  # grouped by the receiver object


@case
def test_noc_sync_method_write_flush_recognized():
    # A Noc-method write-flush (noc.async_write_barrier()) before a free-function
    # noc_semaphore_inc signal must be recognized -> not flagged.
    K = "ttnn/cpp/x/writer.cpp"
    ok = [
        fn("kernel_main", K, 0, 100),
        call(
            K,
            10,
            "async_write_barrier",
            func="kernel_main",
            recv="noc",
            recv_type="Noc",
        ),
        call(K, 20, "noc_semaphore_inc", func="kernel_main", arg0="sem"),
    ]
    assert NocSync().run(FactBase("wormhole", ok)) == []
    # no flush before the signal -> flagged
    bad = [
        fn("kernel_main", K, 0, 100),
        call(K, 20, "noc_semaphore_inc", func="kernel_main", arg0="sem"),
    ]
    assert NocSync().run(FactBase("wormhole", bad))[0].hint == "NOC_SIGNAL_NO_FLUSH"


@case
def test_noc_sync_object_api_signals_recognized():
    # Modern ttnn kernels signal via the Semaphore OBJECT API. A remote signal
    # (mcast or the NoC-carrying up overload) with no preceding write flush must be
    # flagged; the LOCAL up(value)/set(value) forms must NOT be (no NoC, no flush).
    K = "ttnn/cpp/x/writer.cpp"
    # sem.set_multicast(...) with no flush -> flagged (mcast is unambiguously remote)
    mcast = [
        fn("kernel_main", K, 0, 100),
        call(
            K,
            20,
            "set_multicast",
            func="kernel_main",
            recv="sem",
            recv_type="Semaphore",
        ),
    ]
    r = NocSync().run(FactBase("wormhole", mcast))
    assert (
        len(r) == 1
        and r[0].hint == "NOC_SIGNAL_NO_FLUSH"
        and r[0].kind == "noc_signal:mcast"
    )
    # remote sem.up(noc, x, y, v) (argc>=2) with no flush -> flagged
    up_remote = [
        fn("kernel_main", K, 0, 100),
        call(
            K, 20, "up", func="kernel_main", recv="sem", recv_type="Semaphore", argc=5
        ),
    ]
    assert NocSync().run(FactBase("wormhole", up_remote))[0].kind == "noc_signal:inc"
    # LOCAL sem.up(value) (argc==1) is NOT a remote signal -> no finding
    up_local = [
        fn("kernel_main", K, 0, 100),
        call(
            K, 20, "up", func="kernel_main", recv="sem", recv_type="Semaphore", argc=1
        ),
    ]
    assert NocSync().run(FactBase("wormhole", up_local)) == []
    # a flush before the object-API mcast signal clears it
    guarded = [
        fn("kernel_main", K, 0, 100),
        call(
            K,
            10,
            "async_write_barrier",
            func="kernel_main",
            recv="noc",
            recv_type="Noc",
        ),
        call(
            K,
            20,
            "set_multicast",
            func="kernel_main",
            recv="sem",
            recv_type="Semaphore",
        ),
    ]
    assert NocSync().run(FactBase("wormhole", guarded)) == []
    # `up` on a NON-Semaphore receiver is never a signal (guard against false match)
    other = [
        fn("kernel_main", K, 0, 100),
        call(
            K, 20, "up", func="kernel_main", recv="counter", recv_type="Widget", argc=5
        ),
    ]
    assert NocSync().run(FactBase("wormhole", other)) == []


@case
def test_noc_sync_nested_scope_not_double_counted():
    # A signal recorded in an inner (lambda) function is attributed ONCE — by its
    # recorded `function` field — not once per enclosing offset-range. The old
    # fb.functions + facts_in(offset) approach emitted it for BOTH outer and lambda.
    K = "ttnn/cpp/x/w.cpp"
    facts = [
        fn("outer", K, 0, 100),
        fn("outer_lambda", K, 40, 60),  # nested range fully inside outer
        call(K, 50, "noc_semaphore_inc", func="outer_lambda", arg0="sem"),
    ]
    r = NocSync().run(FactBase("wormhole", facts))
    assert len(r) == 1 and r[0].function == "outer_lambda", r


@case
def test_factbase_stray_brace_does_not_eat_following_objects():
    # A stray close-brace / leaked diagnostic between objects must be COUNTED and
    # skipped, never desync the scan and silently drop every following object.
    good = '{"facts": [], "parse_errors": 0}'
    withfact = (
        '{"facts": [{"family":"function","name":"f","file":"a.h",'
        '"off":0,"end_off":9,"line":0}], "parse_errors": 0}'
    )
    text = "}\n" + good + "\nleaked diagnostic line\n" + withfact
    fb = FactBase.from_concatenated_json("wormhole", text)
    assert any(x.get("name") == "f" for x in fb.facts), fb.facts  # later obj survived
    assert fb.parse_errors >= 1  # the garbage was counted, not silent


@case
def test_noc_sync_atomic_multicast_and_relay_unicast():
    K = "ttnn/cpp/x/w.cpp"
    # inc_multicast is an ATOMIC increment (despite the mcast op label) -> a bare
    # writes_flushed does NOT clear it (multicast genuinely needs the ack barrier —
    # departure-order doesn't guarantee all receivers landed); only a barrier clears.
    im_flushed = [
        fn("k", K, 0, 100),
        call(K, 10, "noc_async_writes_flushed", func="k"),
        call(K, 20, "noc_semaphore_inc_multicast", func="k", arg0="sem"),
    ]
    assert (
        NocSync().run(FactBase("wormhole", im_flushed))[0].hint == "NOC_SIGNAL_NO_FLUSH"
    )
    im_barrier = [
        fn("k", K, 0, 100),
        call(K, 10, "noc_async_write_barrier", func="k"),
        call(K, 20, "noc_semaphore_inc_multicast", func="k", arg0="sem"),
    ]
    assert NocSync().run(FactBase("wormhole", im_barrier)) == []
    # Semaphore.relay_unicast -> a WRITE credit (set_remote): flagged with no flush,
    # cleared by a bare flushed (it is a write, ordered by any flush).
    ru = [
        fn("k", K, 0, 100),
        call(
            K, 20, "relay_unicast", func="k", recv="sem", recv_type="Semaphore", argc=2
        ),
    ]
    r = NocSync().run(FactBase("wormhole", ru))
    assert len(r) == 1 and r[0].kind == "noc_signal:set", r
    ru_flushed = [
        fn("k", K, 0, 100),
        call(K, 10, "noc_async_writes_flushed", func="k"),
        call(
            K, 20, "relay_unicast", func="k", recv="sem", recv_type="Semaphore", argc=2
        ),
    ]
    assert NocSync().run(FactBase("wormhole", ru_flushed)) == []


@case
def test_cfg_word_unresolved_cowriter_widens():
    # One KNOWN thread (UNPACK) + an unattributable co-writer (llk_io.h -> UNKNOWN
    # thread) writing the SAME Config word -> surfaced as CROSS_THREAD_SHARED_WORD
    # with safety=UNRESOLVED_COWRITER (low-confidence widen), NOT dropped.
    Fu = "tt_llk_wormhole_b0/common/inc/cunpack_common.h"  # UNPACK
    Fx = "tt_llk_wormhole_b0/llk_lib/llk_io.h"  # UNKNOWN thread
    facts = [
        fn("u", Fu, 800, 820),
        fn("x", Fx, 40, 60),
        call(
            Fu,
            810,
            "cfg_reg_rmw_tensix",
            "cfg_reg_rmw_tensix<FIELD_A_ADDR32>",
            func="u",
        ),
        call(
            Fx, 50, "cfg_reg_rmw_tensix", "cfg_reg_rmw_tensix<FIELD_B_ADDR32>", func="x"
        ),
    ]
    fb = FactBase("wormhole", facts)
    fb.addr32 = {"FIELD_A_ADDR32": 5, "FIELD_B_ADDR32": 5}  # same word
    shared = [
        f for f in CfgWordOverlap().run(fb) if f.hint == "CROSS_THREAD_SHARED_WORD"
    ]
    assert len(shared) == 1 and shared[0].safety == "UNRESOLVED_COWRITER", shared
    # TWO unattributable co-writers (both UNKNOWN thread, token-less file AND function)
    # to the same word -> also WIDENED (a possible cross-thread share the tool can't
    # confirm) with safety=UNRESOLVED_COWRITER, NOT dropped.
    facts2 = [
        fn("x", Fx, 40, 60),
        fn("y", Fx, 70, 90),
        call(
            Fx, 50, "cfg_reg_rmw_tensix", "cfg_reg_rmw_tensix<FIELD_A_ADDR32>", func="x"
        ),
        call(
            Fx, 80, "cfg_reg_rmw_tensix", "cfg_reg_rmw_tensix<FIELD_B_ADDR32>", func="y"
        ),
    ]
    fb2 = FactBase("wormhole", facts2)
    fb2.addr32 = {"FIELD_A_ADDR32": 5, "FIELD_B_ADDR32": 5}
    shared2 = [
        f for f in CfgWordOverlap().run(fb2) if f.hint == "CROSS_THREAD_SHARED_WORD"
    ]
    assert len(shared2) == 1 and shared2[0].safety == "UNRESOLVED_COWRITER", shared2
    # Real negative control: a SINGLE writer to the word is not a shared word.
    facts3 = [
        fn("x", Fx, 40, 60),
        call(
            Fx, 50, "cfg_reg_rmw_tensix", "cfg_reg_rmw_tensix<FIELD_A_ADDR32>", func="x"
        ),
    ]
    fb3 = FactBase("wormhole", facts3)
    fb3.addr32 = {"FIELD_A_ADDR32": 5}
    assert [
        f for f in CfgWordOverlap().run(fb3) if f.hint == "CROSS_THREAD_SHARED_WORD"
    ] == []


@case
def test_thread_of_fact_function_name_fallback():
    # A token-less file (Quasar common/lib header) falls back to the writing
    # function name — so a cross-function clobber (full-word write in one function,
    # masked sibling in another) there is no longer dropped as UNKNOWN.
    from llkaudit import registry

    assert (
        registry.thread_of_fact(
            {
                "file": "tt_llk_quasar/common/inc/ckernel_trisc_common.h",
                "function": "_set_packer_dest_registers_<PACK1>",
            }
        )
        == "PACK"
    )
    # A file WITH a thread token wins over the function name (WH/BH unaffected).
    assert (
        registry.thread_of_fact(
            {"file": "tt_llk_wormhole_b0/common/inc/cunpack_common.h", "function": "z"}
        )
        == "UNPACK"
    )
    # Token-less file AND token-less function -> still UNKNOWN.
    assert (
        registry.thread_of_fact({"file": "x/generic.h", "function": "helper"})
        == "UNKNOWN"
    )


@case
def test_cb_sync_recognizes_known_wrapper():
    from llkaudit import registry

    assert (
        registry.cb_classify({"name": "cb_push_back_hold_wr_ptr", "arg0": "cb0"})[0]
        == "push"
    )
    assert registry.cb_classify({"name": "cb_wait_fronts", "arg0": "cb0"})[0] == "wait"


@case
def test_cli_changed_file_with_no_facts_marks_degraded():
    # A diff-scoped run whose changed file contributed no facts must be marked
    # degraded — a 0-finding result for it is not "race-free".
    import io
    import json as _json
    import os
    import tempfile
    from contextlib import redirect_stdout

    from llkaudit import cli

    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    with open(path, "w") as f:
        f.write(
            '{"arch":"wormhole","parse_errors":0,"facts":[{"family":"function",'
            '"name":"g","file":"fileA.h","off":0,"end_off":9,"line":0}]}'
        )
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli.main(
                [
                    "--arch",
                    "wormhole",
                    "--facts",
                    path,
                    "--checks",
                    "mmio-race",
                    "--changed-files",
                    "fileB.h",
                ]
            )
        out = _json.loads(buf.getvalue())
        assert rc == 0
        assert "degraded" in out and any("fileB.h" in d for d in out["degraded"]), out
    finally:
        os.remove(path)


@case
def test_stallwait_operand_quasar_4operand():
    # Quasar's TTI_STALLWAIT is 4-operand: the wait condition spans the last three
    # operands (operand 2 is often 0). WH/BH are 2-operand. stallwait_wait_operand
    # must return ALL wait_res operands so the Quasar drain token is seen.
    from llkaudit import registry as r

    q = "TTI_STALLWAIT(p_stall::STALL_CFG, 0, p_stall::MATH, p_stall::WAIT_SFPU)"
    w = "TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::MATH)"
    assert r.condition_drains_unit(
        r.stallwait_wait_operand(q), r.DRAIN_UNIT_TOKENS["MATH"]
    )
    assert r.condition_drains_unit(
        r.stallwait_wait_operand(w), r.DRAIN_UNIT_TOKENS["MATH"]
    )
    # stall_res (operand 1) is NOT the drain condition: STALL_UNPACK must not read
    # as an UNPACK drain.
    assert not r.condition_drains_unit(
        r.stallwait_wait_operand(w), r.DRAIN_UNIT_TOKENS["UNPACK"]
    )
    # Quasar per-packer tokens are recognized (word-boundary: PACK0 != PACK).
    qp = "TTI_STALLWAIT(p_stall::STALL_CFG, 0, 0, p_stall::PACK1)"
    assert r.condition_drains_unit(
        r.stallwait_wait_operand(qp), r.DRAIN_UNIT_TOKENS["PACK"]
    )


@case
def test_cfgshiftmask_is_ordered_config_write():
    # Quasar TTI_CFGSHIFTMASK(CfgRegAddr, ...) is a config RMW write — classify_macro
    # must treat it as an ordered_write (so cfg-word folds it into the shared-word
    # map and reconfig-stall requires a drain before it).
    from llkaudit import registry

    assert registry.classify_macro("TTI_CFGSHIFTMASK") == "ordered_write"
    assert "CFGSHIFTMASK" in registry.RECONFIG_WRITE_MACRO_SUBSTR


def main():
    failed = 0
    for c in CASES:
        try:
            c()
            print(f"PASS {c.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {c.__name__}: {e}")
    print(f"\n{len(CASES)-failed}/{len(CASES)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
