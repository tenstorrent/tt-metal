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

from llkaudit.checks.cfg_word_overlap import CfgWordOverlap
from llkaudit.checks.mmio_race import MmioRace
from llkaudit.checks.reconfig_stall import ReconfigStall
from llkaudit.checks.semaphore_handshake import SemaphoreHandshake
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


def call(file, off, name, text=None, func="", arg0=""):
    return {
        "family": "call",
        "file": file,
        "off": off,
        "line": off,
        "function": func,
        "name": name,
        "text": text or (name + "()"),
        "arg0": arg0,
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
def test_cfg_word_overlap_namespace_partition():
    # THCON:56 and MAIN:56 must NOT be reported as a shared word.
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
    assert shared == [], f"THCON/MAIN 56 alias must not be flagged: {shared}"


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
