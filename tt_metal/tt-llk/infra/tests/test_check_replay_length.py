#!/usr/bin/env python3
"""Tests for the replay recording length checker."""

import contextlib
import io
import os
import re
import runpy
import sys
import textwrap
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "check_replay_length.py")


def run(*paths, baseline=None, stats=False):
    """Run the checker as a script (`__main__`, as the hook does), in-process."""
    argv = [SCRIPT, *map(str, paths)]
    if baseline:
        argv += ["--baseline", str(baseline)]
    if stats:
        argv += ["--stats"]
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


def hdr(tmp_path, body, arch="blackhole", name="k.h"):
    """Write body under a path that names the arch, as the real trees do."""
    d = tmp_path / f"tt_llk_{arch}" / "llk_lib"
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(textwrap.dedent(body))
    return p


def body(stmts, decl="inline void _cfg_()"):
    return f"{decl}\n{{\n" + textwrap.indent(textwrap.dedent(stmts), "    ") + "}\n"


# ---------------------------------------------------------------- clean


def test_record_exact_literal_is_clean(tmp_path):
    f = hdr(
        tmp_path,
        body(
            """
            lltt::record(0, 3);
            TTI_SFPNOP;
            TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);
            TTI_NOP;
            """
        ),
    )
    r = run(f, stats=True)
    assert r.returncode == 0
    assert "1  ok" in r.stdout


def test_constexpr_length_and_loop_are_counted(tmp_path):
    f = hdr(
        tmp_path,
        body(
            """
            constexpr std::uint32_t PER_ROW = 2;
            constexpr std::uint32_t LEN = 4 * PER_ROW;
            lltt::record<lltt::NoExec>(0, LEN);
            #pragma GCC unroll 4
            for (std::uint32_t i = 0; i < 4; i++)
            {
                TTI_SFPNOP;
                TT_SFPNOP;
            }
            """
        ),
    )
    r = run(f, stats=True)
    assert r.returncode == 0 and "1  ok" in r.stdout


def test_load_replay_buf_lambda_exact_is_clean(tmp_path):
    f = hdr(
        tmp_path,
        body(
            """
            load_replay_buf(
                0,
                2,
                []
                {
                    TTI_SFPNOP;
                    TTI_SFPNOP;
                });
            """
        ),
    )
    r = run(f, stats=True)
    assert r.returncode == 0 and "1  ok" in r.stdout


def test_quasar_template_form_is_counted(tmp_path):
    f = hdr(
        tmp_path,
        body(
            """
            constexpr std::uint32_t replay_buf_len = 2;
            load_replay_buf<0, replay_buf_len>(
                []
                {
                    TTI_NOP;
                    TTI_NOP;
                });
            """
        ),
        arch="quasar",
    )
    r = run(f, stats=True)
    assert r.returncode == 0 and "1  ok" in r.stdout


def test_overlong_record_is_clean(tmp_path):
    """Instructions after the first N of an lltt::record are issued normally, not recorded."""
    f = hdr(
        tmp_path,
        body(
            """
            lltt::record(0, 1);
            TTI_NOP;
            TTI_NOP;
            """
        ),
    )
    assert run(f).returncode == 0


def test_record_continues_past_a_closing_block(tmp_path):
    """Recording does not stop at a `}`: the statements after the enclosing block are recorded."""
    f = hdr(
        tmp_path,
        body(
            """
            if constexpr (A)
            {
                lltt::record(0, 2);
                TTI_NOP;
            }
            TTI_NOP;
            """,
            decl="template <bool A>\ninline void _cfg_()",
        ),
    )
    r = run(f, stats=True)
    assert r.returncode == 0 and "1  ok" in r.stdout


def test_file_local_macro_is_expanded(tmp_path):
    f = hdr(
        tmp_path,
        "#define TWO_NOPS() TTI_NOP; TTI_NOP;\n"
        "#define FOUR_NOPS() TWO_NOPS() TWO_NOPS()\n"
        + body(
            """
            lltt::record(0, 4);
            FOUR_NOPS()
            """
        ),
    )
    r = run(f, stats=True)
    assert r.returncode == 0 and "1  ok" in r.stdout


def test_raw_insn_word_counts_as_one(tmp_path):
    f = hdr(
        tmp_path,
        body(
            """
            lltt::record(0, 2);
            TTI_INSN(TT_OP_NOP);
            std::uint32_t row = 4 * 2;
            TT_INSN(TT_OP_NOP);
            """
        ),
    )
    r = run(f, stats=True)
    assert r.returncode == 0 and "1  ok" in r.stdout


def test_playback_is_not_a_recording(tmp_path):
    f = hdr(tmp_path, body("TTI_REPLAY(0, 8, 0, 0);\n"))
    r = run(f, stats=True)
    assert r.returncode == 0 and "playback" in r.stdout


# ---------------------------------------------------------------- findings


def test_short_record_is_flagged(tmp_path):
    f = hdr(
        tmp_path,
        body(
            """
            lltt::record(0, 3);
            TTI_NOP;
            TTI_NOP;
            """
        ),
    )
    r = run(f)
    assert r.returncode == 1
    assert "records 3 instructions but only 2 follow" in r.stdout


def test_raw_replay_record_is_checked(tmp_path):
    f = hdr(
        tmp_path,
        body(
            """
            TTI_REPLAY(0, 2, 0, 1);
            TTI_NOP;
            """
        ),
    )
    assert run(f).returncode == 1


def test_quasar_raw_replay_record_reads_its_load_mode(tmp_path):
    """Quasar REPLAY carries load_mode in its sixth argument."""
    f = hdr(
        tmp_path,
        body(
            """
            TTI_REPLAY(0, 2, 0, 0, 0, 1);
            TTI_NOP;
            """
        ),
        arch="quasar",
    )
    assert run(f).returncode == 1


@pytest.mark.parametrize("n, what", [(3, "only 2 follow"), (1, "callable emits 2")])
def test_lambda_mismatch_is_flagged(tmp_path, n, what):
    f = hdr(
        tmp_path,
        body(
            f"""
            load_replay_buf(
                0,
                {n},
                []
                {{
                    TTI_NOP;
                    TTI_NOP;
                }});
            """
        ),
    )
    r = run(f)
    assert r.returncode == 1 and what in r.stdout


# ---------------------------------------------------------------- undecidable, never guessed


@pytest.mark.parametrize(
    "stmts, decl, reason",
    [
        (
            "lltt::record(0, LEN);\nTTI_NOP;\n",
            "template <int LEN>\ninline void _cfg_()",
            "length not constant",
        ),
        (
            "lltt::record(0, len);\nTTI_NOP;\n",
            "inline void _cfg_(int len)",
            "length not constant",
        ),
        (
            "lltt::record(0, 3);\nTTI_NOP;\nhelper();\n",
            None,
            "call or non-instruction statement",
        ),
        ("lltt::record(0, 3);\nTTI_NOP;\nif (x)\n{\n    TTI_NOP;\n}\n", None, "branch"),
        (
            "lltt::record(0, 3);\nfor (int i = 0; i < n; i++)\n{\n    TTI_NOP;\n}\n",
            None,
            "loop bound",
        ),
        (
            "lltt::record(0, 3);\nTTI_NOP;\n#if defined(X)\nTTI_NOP;\n#endif\n",
            None,
            "preprocessor conditional",
        ),
        (
            "lltt::record(0, 3);\nsfpi::vFloat v = sfpi::dst_reg[0];\n",
            None,
            "call or non-instruction statement",
        ),
        (
            "lltt::record(0, 2);\nTTI_NOP;\nlltt::replay(0, 1);\n",
            None,
            "call or non-instruction statement",
        ),
        ("load_replay_buf(0, 2, fn);\n", None, "callable is not an inline lambda"),
        ("lltt::record(0, 0);\n", None, "zero length"),
        (
            "if constexpr (A)\n{\n    constexpr int L = 1;\n}\nelse\n{\n    constexpr int L = 2;\n}\nlltt::record(0, L);\nTTI_NOP;\n",
            "template <bool A>\ninline void _cfg_()",
            "length not constant",
        ),
        (
            "lltt::record(0, 2);\nTTI_NOP;\nstd::uint32_t a = next_addr();\n",
            None,
            "statement",
        ),
    ],
)
def test_undecidable_shapes_are_skipped(tmp_path, stmts, decl, reason):
    f = hdr(tmp_path, body(stmts, decl=decl or "inline void _cfg_()"))
    r = run(f, stats=True)
    assert r.returncode == 0
    assert f"undecidable: {reason}" in r.stdout


def test_comments_are_ignored(tmp_path):
    f = hdr(
        tmp_path,
        body(
            """
            lltt::record(0, 2);
            TTI_NOP;
            // TTI_NOP; would make it three
            /* TTI_NOP; */
            TTI_NOP;
            """
        ),
    )
    r = run(f, stats=True)
    assert r.returncode == 0 and "1  ok" in r.stdout


def test_baseline_suppresses_by_source_line(tmp_path):
    f = hdr(tmp_path, body("lltt::record(0, 5);\nTTI_NOP;\n"))
    bl = tmp_path / "baseline.txt"
    bl.write_text(f"{os.path.relpath(f)}:lltt::record(0, 5);\n")
    assert run(f, baseline=bl).returncode == 0


def test_shipped_trees_have_no_findings_and_are_really_scanned():
    r = run(stats=True)
    assert r.returncode == 0, r.stdout
    counts = {
        m.group(2): int(m.group(1))
        for m in re.finditer(r"(?m)^\s*(\d+)  (.+)$", r.stdout)
    }
    assert counts.get("ok", 0) > 50, "too few decided sites -- the scan did not run"
    assert sum(counts.values()) > 200
