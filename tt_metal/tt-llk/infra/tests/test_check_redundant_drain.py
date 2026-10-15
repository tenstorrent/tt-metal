#!/usr/bin/env python3
"""Tests for the redundant-drain checker."""
import contextlib
import io
import os
import runpy
import sys
import textwrap
import types

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "check_redundant_drain.py")


def run(*args):
    """Run the checker as a script (`__main__`, as the hook does), in-process."""
    argv = [SCRIPT, *map(str, args)]
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


def hdr(tmp_path, name, body, arch="tt_llk_blackhole"):
    """Write a header inside a fake architecture tree so the tree lookup works."""
    d = tmp_path / arch / "llk_lib"
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(textwrap.dedent(body))
    return p


def test_mop_sync_before_tensix_sync_is_redundant(tmp_path):
    f = hdr(
        tmp_path,
        "a.h",
        """
        inline void start()
        {
            mop_sync();
            tensix_sync();
            mailbox_write(MathThreadId, READY);
        }
        """,
    )
    r = run(f)
    assert r.returncode == 0
    assert "a.h:4: REDUNDANT: mop_sync()" in r.stdout
    assert (
        r.stdout.count("a.h:") == 1
    )  # the tensix_sync that follows is required and not reported


def test_drain_before_mailbox_is_required(tmp_path):
    f = hdr(
        tmp_path,
        "a.h",
        """
        inline void release()
        {
            TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
            tensix_sync();
            mailbox_write(MathThreadId, DONE);
        }
        """,
    )
    assert run(f).stdout == ""


def test_drain_followed_only_by_tensix_pushes_is_advisory(tmp_path):
    f = hdr(
        tmp_path,
        "a.h",
        """
        inline void reconfig()
        {
            tensix_sync();
            TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(1);
            TT_SETDMAREG(0, 0, 0, LO_16(p_gpr_pack::TMP0));
        }
        """,
    )
    r = run(f)
    assert "a.h:4: ADVISORY: tensix_sync() nothing RISC-side follows" in r.stdout
    assert r.returncode == 0


def test_two_tensix_syncs_back_to_back_flags_the_first(tmp_path):
    f = hdr(
        tmp_path,
        "a.h",
        """
        inline void twice()
        {
            tensix_sync();
            TTI_NOP;
            tensix_sync();
            semaphore_read(semaphore::MATH_PACK);
        }
        """,
    )
    r = run(f)
    assert "a.h:4: REDUNDANT: tensix_sync()" in r.stdout
    assert "a.h:6:" not in r.stdout


def test_tensix_sync_before_mop_sync_is_not_covered(tmp_path):
    # mop_sync is the weaker drain, so it does not subsume an earlier tensix_sync; the scan
    # continues past it and the mailbox write makes the first drain required.
    f = hdr(
        tmp_path,
        "a.h",
        """
        inline void mixed()
        {
            tensix_sync();
            TTI_NOP;
            mop_sync();
            mailbox_write(MathThreadId, DONE);
        }
        """,
    )
    assert "REDUNDANT" not in run(f).stdout


def test_mmio_write_through_pointer_is_a_consumer(tmp_path):
    f = hdr(
        tmp_path,
        "a.h",
        """
        inline void program()
        {
            volatile std::uint32_t *mop_cfg = reinterpret_cast<volatile std::uint32_t *>(TENSIX_MOP_CFG_BASE);
            mop_sync();
            mop_cfg[0] = m_outer_loop_len;
        }
        """,
    )
    assert run(f).stdout == ""


def test_helper_in_the_tree_is_classified_by_its_body(tmp_path):
    hdr(
        tmp_path,
        "helpers.h",
        """
        inline void writes_cfg_by_mmio()
        {
            volatile std::uint32_t tt_reg_ptr *cfg = get_cfg_pointer();
            cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32] = 0;
        }
        inline void pushes_only()
        {
            TTI_SETADCXY(0b100, 0, 0, 0, 0, 0b1011);
            TTI_SETADCZW(0b100, 0, 0, 0, 0, 0b1111);
        }
        """,
    )
    f = hdr(
        tmp_path,
        "a.h",
        """
        inline void required()
        {
            tensix_sync();
            writes_cfg_by_mmio();
        }
        inline void advisory()
        {
            tensix_sync();
            pushes_only();
        }
        """,
    )
    r = run(f)
    assert "a.h:4:" not in r.stdout
    assert "a.h:9: ADVISORY" in r.stdout


def test_unknown_helper_suppresses_the_finding(tmp_path):
    f = hdr(
        tmp_path,
        "a.h",
        """
        inline void maybe()
        {
            tensix_sync();
            something_not_in_the_tree();
        }
        """,
    )
    assert run(f).stdout == ""


def test_pointer_store_and_volatile_load_are_consumers(tmp_path):
    f = hdr(
        tmp_path,
        "a.h",
        """
        inline void flag(std::uint32_t *done)
        {
            tensix_sync();
            *done = 1;
        }
        inline std::uint32_t poll(volatile std::uint32_t *p)
        {
            tensix_sync();
            return p[0];
        }
        """,
    )
    assert run(f).stdout == ""


def test_drain_inside_a_condition_is_not_a_statement(tmp_path):
    f = hdr(
        tmp_path,
        "a.h",
        """
        inline void guarded(bool x)
        {
            if (x)
            {
                tensix_sync();
            }
            mailbox_write(MathThreadId, DONE);
        }
        """,
    )
    assert run(f).stdout == ""


def test_definitions_and_comments_are_ignored(tmp_path):
    f = hdr(
        tmp_path,
        "a.h",
        """
        inline void tensix_sync()
        {
            store_blocking(&pc_buf_base[1], 0);
        }
        inline void documented()
        {
            // tensix_sync(); is not needed here
            TTI_NOP;
        }
        """,
    )
    assert run(f).stdout == ""


def test_quasar_is_skipped(tmp_path):
    f = hdr(
        tmp_path,
        "a.h",
        """
        inline void q()
        {
            mop_sync();
            tensix_sync();
        }
        """,
        arch="tt_llk_quasar",
    )
    r = run(f)
    assert "REDUNDANT" not in r.stdout
    assert "Quasar" in r.stdout


def test_baseline_suppresses_a_reviewed_site(tmp_path):
    f = hdr(
        tmp_path,
        "a.h",
        """
        inline void kept()
        {
            tensix_sync();
            TTI_NOP;
        }
        """,
    )
    rel = os.path.relpath(f).replace(os.sep, "/")
    b = tmp_path / "baseline.txt"
    b.write_text(
        f"{rel}:inline void kept()  # the caller's Tensix ops must see the RISC stores above\n"
    )
    assert run(f, "--baseline", b).stdout == ""
    assert "ADVISORY" in run(f).stdout


def test_json_output_and_strict_exit(tmp_path):
    f = hdr(
        tmp_path,
        "a.h",
        """
        inline void start()
        {
            mop_sync();
            tensix_sync();
            mailbox_write(MathThreadId, READY);
        }
        """,
    )
    r = run(f, "--json")
    assert '"kind": "REDUNDANT"' in r.stdout and '"drain": "mop_sync"' in r.stdout
    assert run(f, "--strict").returncode == 1
    assert run(f).returncode == 0


def test_real_tree_runs_clean_of_errors():
    """The checker parses every real header; what it finds there is advisory, not asserted."""
    llk = os.path.normpath(os.path.join(HERE, "..", ".."))
    r = run(
        "--tree",
        os.path.join(llk, "tt_llk_blackhole"),
        "--tree",
        os.path.join(llk, "tt_llk_wormhole_b0"),
    )
    assert r.returncode == 0
    assert "Traceback" not in r.stderr
