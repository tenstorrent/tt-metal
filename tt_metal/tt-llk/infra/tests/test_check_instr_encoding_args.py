#!/usr/bin/env python3
"""Tests for the instruction-argument constant-family checker."""
import contextlib
import io
import os
import runpy
import subprocess
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "check_instr_encoding_args.py")
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))


def run(*paths):
    """Run the checker as a script (`__main__`, as the hook does), in-process."""
    argv = [SCRIPT, *map(str, paths)]
    out, err, code = io.StringIO(), io.StringIO(), 0
    saved = sys.argv
    sys.argv = argv
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            runpy.run_path(SCRIPT, run_name="__main__")
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    finally:
        sys.argv = saved
    return types.SimpleNamespace(
        returncode=code, stdout=out.getvalue(), stderr=err.getvalue()
    )


def body(tmp_path, *stmts):
    p = tmp_path / "k.h"
    lines = "\n".join("    " + s for s in stmts)
    p.write_text(f"inline void f()\n{{\n{lines}\n}}\n")
    return p


# ---- STALLWAIT / SEMWAIT argument roles ----


@pytest.mark.parametrize(
    "stmt",
    [
        "TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);",
        "TT_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD);",
        "std::uint32_t w = TT_OP_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);",
        "TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::NOTHING, p_stall::MATH, p_stall::SRCB_VLD);",
        "TTI_STALLWAIT(p_stall::STALL_CFG, 0, 0, p_stall::PACK);",
        "TTI_STALLWAIT(stall_res, wait_res);",
        "TTI_SEMWAIT(p_stall::STALL_MATH, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO);",
        "TTI_SEMWAIT(p_stall::STALL_SYNC, p_stall::STALL_ON_MAX, 0, semaphore::t6_sem(idx));",
    ],
)
def test_canonical_calls_are_clean(tmp_path, stmt):
    r = run(body(tmp_path, stmt))
    assert r.returncode == 0, r.stdout


@pytest.mark.parametrize(
    "stmt, fragment",
    [
        # WH/BH: arguments swapped
        (
            "TTI_STALLWAIT(p_stall::MATH, p_stall::STALL_MATH);",
            "p_stall::MATH is not valid as the stall class",
        ),
        # a stall class in the wait slot, inside an OR-ed mask
        (
            "TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::STALL_SFPU);",
            "STALL_SFPU is not valid as a wait",
        ),
        (
            "TT_STALLWAIT(p_stall::STALL_CFG | p_stall::SRCA_VLD, p_stall::MATH);",
            "SRCA_VLD is not valid as the stall class",
        ),
        (
            "w = TT_OP_STALLWAIT(p_stall::THCON, p_stall::STALL_THCON);",
            "THCON is not valid as the stall class",
        ),
        # Quasar four-argument form, both directions
        (
            "TTI_STALLWAIT(p_stall::MATH, 0, 0, p_stall::SRCB_VLD);",
            "p_stall::MATH is not valid as the stall class",
        ),
        (
            "TTI_STALLWAIT(p_stall::STALL_MATH, 0, p_stall::STALL_CFG, p_stall::MATH);",
            "STALL_CFG is not valid as a wait",
        ),
        # an empty stall class
        (
            "TTI_STALLWAIT(p_stall::NOTHING, 0, 0, p_stall::MATH);",
            "NOTHING is not valid as the stall class",
        ),
        ("TTI_STALLWAIT(0x40, p_stall::MATH);", "literal `0x40`"),
        # a semaphore condition in a stall slot, and the reverse
        (
            "TTI_STALLWAIT(p_stall::STALL_ON_ZERO, p_stall::MATH);",
            "STALL_ON_ZERO is not valid as the stall class",
        ),
        (
            "TTI_SEMWAIT(p_stall::STALL_MATH, semaphore::t6_sem(s), p_stall::STALL_MATH);",
            "STALL_MATH is not valid as the semaphore condition",
        ),
        (
            "TTI_SEMWAIT(p_stall::STALL_SYNC, p_stall::MATH, 0, semaphore::t6_sem(s));",
            "MATH is not valid as the semaphore condition",
        ),
    ],
)
def test_role_violations_fail(tmp_path, stmt, fragment):
    r = run(body(tmp_path, stmt))
    assert r.returncode == 1
    assert fragment in r.stdout, r.stdout
    assert "k.h:3:" in r.stdout, "the finding must name the call's line"


def test_comments_and_strings_are_ignored(tmp_path):
    r = run(
        body(
            tmp_path,
            "// TTI_STALLWAIT(p_stall::MATH, p_stall::STALL_MATH);",
            '/* TTI_SFPLOAD(0, SFPLOADI_MOD0_FLOATB, 0, 0); */ const char* s = "TTI_STALLWAIT(p_stall::MATH, p_stall::STALL_MATH)";',
        )
    )
    assert r.returncode == 0, r.stdout


# ---- instruction-scoped constants ----


@pytest.mark.parametrize(
    "stmt, fragment",
    [
        # the shape of the real defect: SFPLOADI's immediate-format selector in SFPLOAD/SFPSTORE
        (
            "TTI_SFPLOAD(p_sfpu::LREG0, SFPLOADI_MOD0_FLOATB, ADDR_MOD_3, 0);",
            "SFPLOADI modifier, passed to SFPLOAD",
        ),
        (
            "TT_SFPSTORE(p_sfpu::LREG0, SFPLOADI_MOD0_FLOATB, ADDR_MOD_3, 0);",
            "SFPLOADI modifier, passed to SFPSTORE",
        ),
        (
            "TTI_SFPSHFT(0, p_sfpu::LREG1, p_sfpu::LREG2, SFPSHFT2_MOD1_SHFT_IMM);",
            "SFPSHFT2 modifier, passed to SFPSHFT",
        ),
        # TT_<X>_VALID range-checks X's operands, so it is X's argument list
        (
            "TT_SFPLOAD_VALID(p_sfpu::LREG0, SFPLOADI_MOD0_FLOATB, ADDR_MOD_3, 0)",
            "SFPLOADI modifier, passed to SFPLOAD",
        ),
        (
            "w = TT_OP_SFPIADD(0, 1, 2, SFPSETCC_MOD1_LREG_LT0);",
            "SFPSETCC modifier, passed to SFPIADD",
        ),
        (
            "TTI_MOVB2D(0, p_movb2a::SRCA_ZERO_OFFSET, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 0);",
            "p_movb2a::SRCA_ZERO_OFFSET belongs to p_movb2a",
        ),
        (
            "TTI_SFPSWAP(0, 1, 2, p_sfpgt::SOMETHING);",
            "belongs to p_sfpgt, passed to SFPSWAP",
        ),
    ],
)
def test_constant_from_another_instruction_fails(tmp_path, stmt, fragment):
    r = run(body(tmp_path, stmt))
    assert r.returncode == 1
    assert fragment in r.stdout, r.stdout


@pytest.mark.parametrize(
    "stmt",
    [
        "TTI_SFPLOADI(p_sfpu::LREG0, SFPLOADI_MOD0_FLOATB, 0x3f80);",
        "TT_SFPLOADI(p_sfpu::LREG0, (SFPLOADI_MOD0_UPPER), val >> 16);",
        "TTI_SFPLOADI_VALID(p_sfpu::LREG0, SFPLOADI_MOD0_FLOATB, 0);",
        # sfpi spells SFP_STOCH_RND's modifiers without the underscores
        "TTI_SFP_STOCH_RND(0, 0, 0, 1, 2, SFPSTOCHRND_MOD1_FP32_TO_FP16B);",
        "TTI_REG2FLOP_COMMON(1, 0, 0, 0, p_reg2flop::WRITE_4B, 0, 0);",
        "TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 0);",
        # families shared by design are not enforced
        "TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0);",
        "TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + 0, ADDR_MOD_0, p_movd2a::MOV_4_ROWS, 0);",
    ],
)
def test_constant_in_its_own_instruction_is_clean(tmp_path, stmt):
    r = run(body(tmp_path, stmt))
    assert r.returncode == 0, r.stdout


@pytest.mark.parametrize(
    "stmt",
    [
        # forwarded through a wrapper: the wrapper, not this call, decides the instruction
        "TTI_SFPLOAD(p_sfpu::LREG0, to_load_mode(SFPLOADI_MOD0_FLOATB), ADDR_MOD_3, 0);",
        "TTI_SFPLOAD(p_sfpu::LREG0, cvt<SFPLOADI_MOD0_FLOATB>(x), ADDR_MOD_3, 0);",
        "_sfpu_load_imm_(SFPLOADI_MOD0_FLOATB, 0x3f80);",
        "constexpr std::uint32_t mode = SFPLOADI_MOD0_FLOATB;",
        "__builtin_rvtt_sfpxiadd_i(a, b, SFPIADD_MOD1_CC_NONE);",
        # a non-instruction TT_ macro is not an instruction call
        "TT_INSN(word | SFPLOADI_MOD0_FLOATB);",
    ],
)
def test_forwarded_constant_is_not_checked(tmp_path, stmt):
    r = run(body(tmp_path, stmt))
    assert r.returncode == 0, r.stdout


def test_no_files_is_clean():
    assert run().returncode == 0


def test_shipped_trees_are_clean():
    """The measured baseline: zero findings in both trees, so any finding is new."""
    files = subprocess.check_output(
        [
            "git",
            "ls-files",
            "tt_metal/tt-llk/tt_llk_*/*.h",
            "tt_metal/tt-llk/tt_llk_*/*.hpp",
            "tt_metal/hw/ckernels/*/metal/*.h",
        ],
        cwd=REPO,
        text=True,
    ).split()
    assert (
        len(files) > 500
    ), f"only {len(files)} headers found -- test would pass vacuously"
    paths = [os.path.join(REPO, f) for f in files]
    joined = "".join(open(p, errors="ignore").read() for p in paths)
    assert (
        joined.count("STALLWAIT(") > 200 and "SFPLOADI_MOD0_" in joined
    ), "scan inputs missing"
    r = run(*paths)
    assert r.returncode == 0, r.stdout
