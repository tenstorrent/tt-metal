#!/usr/bin/env python3
"""Tests for the Dest->Src move source-bank wait checker."""
import contextlib
import io
import os
import runpy
import sys
import textwrap
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "check_dest_to_src_wait.py")


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


def hdr(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return p


@pytest.mark.parametrize(
    "move, vld",
    [
        ("TTI_MOVD2A(0, 0, ADDR_MOD_0, p_movd2a::MOV_4_ROWS, 0);", "SRCA_VLD"),
        ("TTI_MOVD2B(0, 0, ADDR_MOD_0, p_movd2b::MOV_4_ROWS, 0);", "SRCB_VLD"),
    ],
)
def test_move_without_wait_warns_but_never_fails(tmp_path, move, vld):
    f = hdr(
        tmp_path,
        "bad.h",
        f"""
        inline void _llk_math_thing_()
        {{
            {move}
        }}
        """,
    )
    r = run(f)
    assert r.returncode == 0, "the check is advisory: it must never fail a commit"
    assert f"no {vld} wait" in r.stdout
    assert "_llk_math_thing_" in r.stdout


@pytest.mark.parametrize(
    "wait",
    [
        # WH / BH form
        "TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH | p_stall::SRCB_VLD);",
        "TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCB_VLD | p_stall::MATH);",
        "TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH | p_stall::SRCA_VLD | p_stall::SRCB_VLD);",
        # Quasar form: wait resources are the trailing arguments
        "TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::NOTHING, p_stall::MATH, p_stall::SRCB_VLD);",
    ],
)
def test_matching_wait_before_move_is_clean(tmp_path, wait):
    f = hdr(
        tmp_path,
        "ok.h",
        f"""
        inline void _llk_math_thing_()
        {{
            {wait}
            TTI_MOVD2B(0, 0, ADDR_MOD_0, p_movd2b::MOV_4_ROWS, 0);
        }}
        """,
    )
    r = run(f)
    assert r.returncode == 0
    assert "warning" not in r.stdout


@pytest.mark.parametrize(
    "wait",
    [
        "TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCB_VLD);",
        "TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::NOTHING, p_stall::NOTHING, p_stall::SRCB_VLD);",
    ],
)
def test_valid_only_wait_warns_about_the_missing_drain(tmp_path, wait):
    """SRC?_VLD without MATH: a bank-clearing math op still in flight makes the valid bit test
    the old bank, so the wait passes vacuously. Distinct warning, still advisory."""
    f = hdr(
        tmp_path,
        "vld_only.h",
        f"""
        inline void _llk_math_thing_()
        {{
            {wait}
            TTI_MOVD2B(0, 0, ADDR_MOD_0, p_movd2b::MOV_4_ROWS, 0);
        }}
        """,
    )
    r = run(f)
    assert r.returncode == 0
    assert "MOVD2B behind a SRCB_VLD wait that does not drain math" in r.stdout
    assert "no SRCB_VLD wait" not in r.stdout


def test_valid_only_helper_call_warns_about_the_missing_drain(tmp_path):
    f = hdr(
        tmp_path,
        "vld_only_helper.h",
        """
        inline void srcb_vld_wait()
        {
            TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCB_VLD);
        }

        inline void _llk_math_thing_()
        {
            srcb_vld_wait();
            TTI_MOVD2B(0, 0, ADDR_MOD_0, p_movd2b::MOV_4_ROWS, 0);
        }
        """,
    )
    out = run(f).stdout
    assert "MOVD2B behind a SRCB_VLD wait that does not drain math" in out
    assert "_llk_math_thing_" in out


def test_drain_without_valid_bit_does_not_cover(tmp_path):
    """MATH alone drains but never waits for the bank: still 'no SRCB_VLD wait'."""
    f = hdr(
        tmp_path,
        "math_only.h",
        """
        inline void _llk_math_thing_()
        {
            TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH);
            TTI_MOVD2B(0, 0, ADDR_MOD_0, p_movd2b::MOV_4_ROWS, 0);
        }
        """,
    )
    assert "MOVD2B with no SRCB_VLD wait" in run(f).stdout


def test_wrong_bank_wait_does_not_cover(tmp_path):
    """A SrcA wait says nothing about SrcB: MOVD2B behind SRCA_VLD is still uncovered."""
    f = hdr(
        tmp_path,
        "crossed.h",
        """
        inline void _llk_math_thing_()
        {
            TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD);
            TTI_MOVD2B(0, 0, ADDR_MOD_0, p_movd2b::MOV_4_ROWS, 0);
        }
        """,
    )
    assert "MOVD2B with no SRCB_VLD wait" in run(f).stdout


def test_wait_after_move_does_not_cover(tmp_path):
    f = hdr(
        tmp_path,
        "late.h",
        """
        inline void _llk_math_thing_()
        {
            TTI_MOVD2A(0, 0, ADDR_MOD_0, p_movd2a::MOV_4_ROWS, 0);
            TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD);
        }
        """,
    )
    assert "MOVD2A with no SRCA_VLD wait" in run(f).stdout


def test_wait_helper_call_covers(tmp_path):
    """A call to a function whose body waits on the bank counts as the wait."""
    f = hdr(
        tmp_path,
        "helper.h",
        """
        inline void srca_bank_wait()
        {
            TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH | p_stall::SRCA_VLD);
        }

        inline void _llk_math_thing_()
        {
            srca_bank_wait();
            TTI_MOVD2A(0, 0, ADDR_MOD_0, p_movd2a::MOV_4_ROWS, 0);
        }
        """,
    )
    assert "warning" not in run(f).stdout


def test_move_inside_macro_counts_at_use_site(tmp_path):
    f = hdr(
        tmp_path,
        "macro.h",
        """
        #define MOVD2A_4_ROWS(o) \\
            TTI_MOVD2A(0, (o), ADDR_MOD_0, p_movd2a::MOV_4_ROWS, (o));
        #define MOVD2A_8_ROWS(o) MOVD2A_4_ROWS(o) MOVD2A_4_ROWS((o) + 4)

        inline void _llk_math_thing_()
        {
            MOVD2A_8_ROWS(0)
        }
        """,
    )
    assert "MOVD2A with no SRCA_VLD wait" in run(f).stdout


def test_one_function_wait_does_not_cover_another(tmp_path):
    f = hdr(
        tmp_path,
        "two.h",
        """
        namespace ckernel
        {
        inline void _waits_()
        {
            TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH | p_stall::SRCB_VLD);
            TTI_MOVD2B(0, 0, ADDR_MOD_0, p_movd2b::MOV_4_ROWS, 0);
        }

        inline void _moves_()
        {
            TTI_MOVD2B(0, 0, ADDR_MOD_0, p_movd2b::MOV_4_ROWS, 0);
        }
        }
        """,
    )
    out = run(f).stdout
    assert out.count(": warning:") == 1
    assert "_moves_" in out


@pytest.mark.parametrize(
    "record",
    [
        "lltt::record(0, 4);",
        "lltt::record<lltt::NoExec>(0, 4);",
    ],
)
def test_record_only_moves_are_not_checked(tmp_path, record):
    """Recorded-not-issued moves get their wait before the replay, not where they are written."""
    f = hdr(
        tmp_path,
        "rec.h",
        f"""
        inline void _mop_config_()
        {{
            {record}
            TTI_MOVD2B(0, 0, ADDR_MOD_0, p_movd2b::MOV_4_ROWS, 0);
        }}
        """,
    )
    assert "warning" not in run(f).stdout


@pytest.mark.parametrize(
    "wait, warns",
    [
        (
            "TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU | p_stall::SRCA_VLD);",
            True,
        ),
        (
            "TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH | p_stall::WAIT_SFPU | p_stall::SRCA_VLD);",
            False,
        ),
    ],
)
def test_wait_recorded_with_the_moves_is_checked(tmp_path, wait, warns):
    """The generalized moe-gate shape: the recording carries its own wait, which replays with the
    moves as the MOP start op, so a valid-only wait there is the same bug as an inline one.
    """
    f = hdr(
        tmp_path,
        "rec_wait.h",
        f"""
        inline void _mop_config_()
        {{
            lltt::record<lltt::NoExec>(0, 2);
            {wait}
            TTI_MOVD2A(0, 0, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, 0);
            std::uint32_t replay_instr = lltt::replay_insn(0, 2);
        }}
        """,
    )
    out = run(f).stdout
    assert ("MOVD2A behind a SRCA_VLD wait that does not drain math" in out) == warns
    assert "no SRCA_VLD wait" not in out


def test_wait_outside_the_recording_does_not_cover_one_inside(tmp_path):
    """A wait issued before the recording runs now, not when the replay does."""
    f = hdr(
        tmp_path,
        "rec_outer.h",
        """
        inline void _thing_()
        {
            TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH | p_stall::SRCB_VLD);
            load_replay_buf(
                0,
                2,
                []
                {
                    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCB_VLD);
                    TTI_MOVD2B(0, 0, ADDR_MOD_0, p_movd2b::MOV_4_ROWS, 0);
                });
        }
        """,
    )
    assert "MOVD2B behind a SRCB_VLD wait that does not drain math" in run(f).stdout


def test_default_scan_includes_ttnn_kernel_includes_copies():
    """Op-local LLK copies are included by relative path and drift from the tree."""
    g = runpy.run_path(SCRIPT)
    copies = __import__("glob").glob(g["COPIES_GLOB"], recursive=True)
    assert any(
        p.endswith("llk_math_deepseek_moe_gate_eltwise_binary.h") for p in copies
    )


def test_exec_record_is_still_checked(tmp_path):
    """`lltt::Exec` issues the instructions as it records them."""
    f = hdr(
        tmp_path,
        "exec.h",
        """
        inline void _thing_()
        {
            lltt::record<lltt::Exec>(0, 4);
            TTI_MOVD2B(0, 0, ADDR_MOD_0, p_movd2b::MOV_4_ROWS, 0);
        }
        """,
    )
    assert "MOVD2B with no SRCB_VLD wait" in run(f).stdout


def test_record_span_ends_with_its_block(tmp_path):
    """A recording in one branch does not hide an issued move in another."""
    f = hdr(
        tmp_path,
        "branch.h",
        """
        template <bool A>
        inline void _thing_()
        {
            if constexpr (A)
            {
                lltt::record(0, 1);
                TTI_MOVD2B(0, 0, ADDR_MOD_0, p_movd2b::MOV_4_ROWS, 0);
            }
            else
            {
                TTI_MOVD2B(0, 0, ADDR_MOD_0, p_movd2b::MOV_4_ROWS, 0);
            }
        }
        """,
    )
    assert "MOVD2B with no SRCB_VLD wait" in run(f).stdout


def test_load_replay_buf_lambda_is_not_checked_but_rest_is(tmp_path):
    f = hdr(
        tmp_path,
        "lrb.h",
        """
        inline void _thing_()
        {
            load_replay_buf(
                0,
                2,
                []
                {
                    TTI_MOVD2A(0, 0, ADDR_MOD_0, p_movd2a::MOV_4_ROWS, 0);
                });
            TTI_MOVD2B(0, 0, ADDR_MOD_0, p_movd2b::MOV_4_ROWS, 0);
        }
        """,
    )
    out = run(f).stdout
    assert "MOVD2A" not in out
    assert "MOVD2B with no SRCB_VLD wait" in out


def test_op_word_is_not_an_issue(tmp_path):
    f = hdr(
        tmp_path,
        "op.h",
        """
        inline void _mop_config_()
        {
            std::uint32_t w = TT_OP_MOVD2A(0, 0, ADDR_MOD_0, p_movd2a::MOV_4_ROWS, 0);
        }
        """,
    )
    assert "warning" not in run(f).stdout


def test_commented_out_move_is_ignored(tmp_path):
    f = hdr(
        tmp_path,
        "comment.h",
        """
        inline void _thing_()
        {
            // TTI_MOVD2A(0, 0, ADDR_MOD_0, p_movd2a::MOV_4_ROWS, 0);
        }
        """,
    )
    assert "warning" not in run(f).stdout


def test_recorded_wait_in_a_helper_does_not_cover_callers(tmp_path):
    """Calling a helper that only records its wait does not issue the wait."""
    f = hdr(
        tmp_path,
        "rec_helper.h",
        """
        inline void record_srcb_wait()
        {
            lltt::record(0, 1);
            TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH | p_stall::SRCB_VLD);
        }

        inline void _llk_math_thing_()
        {
            record_srcb_wait();
            TTI_MOVD2B(0, 0, ADDR_MOD_0, p_movd2b::MOV_4_ROWS, 0);
        }
        """,
    )
    assert "MOVD2B with no SRCB_VLD wait" in run(f).stdout


def test_shipped_tree_never_fails():
    """Whole-tree run (no files) exits 0 whether or not findings remain."""
    g = runpy.run_path(SCRIPT)
    tree = __import__("glob").glob(g["TREE_GLOB"], recursive=True)
    # Guards against a glob that silently matches nothing: the tree does issue these moves.
    assert any("MOVD2B" in open(p, errors="ignore").read() for p in tree)
    assert run().returncode == 0
