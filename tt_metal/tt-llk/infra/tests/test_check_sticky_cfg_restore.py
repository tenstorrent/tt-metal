#!/usr/bin/env python3
"""Tests for the sticky config-field restore checker."""
import contextlib
import io
import os
import runpy
import sys
import textwrap
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
INFRA = os.path.normpath(os.path.join(HERE, ".."))
SCRIPT = os.path.join(INFRA, "check_sticky_cfg_restore.py")
BASELINE = os.path.join(INFRA, "sticky_cfg_baseline.txt")
FP32 = "ALU_ACC_CTRL_Fp32_enabled"
ZF = "ALU_ACC_CTRL_Zero_Flag_disabled_src"


def run(*paths, baseline=None, table=None):
    """Run the checker as a script (`__main__`, as the hook does), in-process."""
    argv = [SCRIPT, *map(str, paths)]
    if baseline:
        argv += ["--baseline", str(baseline)]
    if table:
        argv += ["--table", str(table)]
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
    return types.SimpleNamespace(returncode=code, stdout=out.getvalue())


def hdr(tmp_path, body, name="op.h", arch="blackhole"):
    d = tmp_path / f"tt_llk_{arch}" / "llk_lib"
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(textwrap.dedent(body))
    return p


def rmw(field, v):
    return f"cfg_reg_rmw_tensix<{field}_RMW>({v});"


# ---- restore policy ----


def test_unrestored_fixed_write_fails(tmp_path):
    f = hdr(tmp_path, f"inline void _llk_math_thing_() {{ {rmw(FP32, 0)} }}\n")
    r = run(f)
    assert r.returncode == 1
    assert "[restore]" in r.stdout and "_llk_math_thing_" in r.stdout


def test_restored_in_same_function_passes(tmp_path):
    f = hdr(
        tmp_path,
        f"""
        inline void _llk_math_thing_()
        {{
            {rmw(FP32, 0)}
            do_work();
            {rmw(FP32, 1)}
        }}
        """,
    )
    assert run(f).returncode == 0


def test_restored_with_configured_value_passes(tmp_path):
    f = hdr(
        tmp_path,
        f"""
        template <bool is_fp32_dest_acc_en>
        inline void _llk_math_thing_()
        {{
            {rmw(FP32, 0)}
            {rmw(FP32, "is_fp32_dest_acc_en")}
        }}
        """,
    )
    assert run(f).returncode == 0


def test_restored_by_matching_uninit_passes(tmp_path):
    f = hdr(
        tmp_path,
        f"""
        inline void _llk_math_thing_init_() {{ {rmw(FP32, 0)} }}
        inline void _llk_math_thing_uninit_() {{ {rmw(FP32, 1)} }}
        """,
    )
    assert run(f).returncode == 0


def test_uninit_of_another_op_does_not_cover(tmp_path):
    f = hdr(
        tmp_path,
        f"""
        inline void _llk_math_thing_init_() {{ {rmw(FP32, 0)} }}
        inline void _llk_math_other_uninit_() {{ {rmw(FP32, 1)} }}
        """,
    )
    assert run(f).returncode == 1


def test_uninit_for_another_field_does_not_cover(tmp_path):
    f = hdr(
        tmp_path,
        f"""
        inline void _llk_math_thing_init_() {{ {rmw(FP32, 0)} }}
        inline void _llk_math_thing_uninit_() {{ {rmw("ALU_ACC_CTRL_SFPU_Fp32_enabled", 1)} }}
        """,
    )
    assert run(f).returncode == 1


def test_same_value_twice_is_not_a_restore(tmp_path):
    f = hdr(
        tmp_path,
        f"inline void _llk_math_thing_() {{ {rmw(FP32, 0)} {rmw(FP32, 0)} }}\n",
    )
    assert run(f).returncode == 1


def test_bitfield_struct_write_counts(tmp_path):
    f = hdr(tmp_path, f"inline void _llk_math_thing_() {{ alu.f.{FP32} = 0; }}\n")
    assert run(f).returncode == 1


def test_configuring_function_is_exempt(tmp_path):
    f = hdr(
        tmp_path,
        f"inline void _llk_set_fp32_dest_acc_(bool enable) {{ {rmw(FP32, 'enable')} {rmw(FP32, 0)} }}\n",
    )
    r = run(f)
    assert r.returncode == 0
    assert "note" not in r.stdout


def test_unresolved_value_is_a_note_not_a_failure(tmp_path):
    f = hdr(tmp_path, f"inline void _llk_math_thing_(bool v) {{ {rmw(FP32, 'v')} }}\n")
    r = run(f)
    assert r.returncode == 0
    assert "cannot be judged" in r.stdout


def test_table_site_covers(tmp_path):
    f = hdr(tmp_path, f"inline void _llk_math_thing_() {{ {rmw(FP32, 0)} }}\n")
    rel = os.path.relpath(
        str(f), os.path.normpath(os.path.join(INFRA, "..", "..", ".."))
    )
    t = tmp_path / "t.yaml"
    t.write_text(
        open(os.path.join(INFRA, "sticky_cfg_fields.yaml"))
        .read()
        .replace(
            "sites: []",
            textwrap.dedent(
                f"""\
                sites:
                  - path: {rel}
                    function: _llk_math_thing_
                    field: {FP32}
                    restored_by: _llk_math_thing_done_
                    reason: the op's done hook writes the configured value back
                """
            ),
        )
    )
    assert run(f).returncode == 1
    assert run(f, table=t).returncode == 0


def test_field_not_in_table_is_ignored(tmp_path):
    f = hdr(
        tmp_path,
        "inline void _llk_math_thing_() { cfg_reg_rmw_tensix<ALU_ROUNDING_MODE_Fpu_srnd_en_RMW>(1); }\n",
    )
    assert run(f).returncode == 0


def test_arch_not_listed_is_ignored(tmp_path):
    f = hdr(
        tmp_path,
        f"inline void _llk_math_thing_() {{ {rmw(FP32, 0)} }}\n",
        arch="quasar",
    )
    assert run(f).returncode == 0


def test_comments_are_ignored(tmp_path):
    f = hdr(tmp_path, f"inline void _llk_math_thing_() {{ // {rmw(FP32, 0)}\n }}\n")
    assert run(f).returncode == 0


# ---- tracked policy ----


def test_raw_tracked_write_without_invalidate_fails(tmp_path):
    f = hdr(
        tmp_path, f"inline void _llk_math_thing_() {{ {rmw(ZF, 1)} {rmw(ZF, 0)} }}\n"
    )
    r = run(f)
    assert r.returncode == 1
    assert "[tracked]" in r.stdout


def test_raw_tracked_write_then_invalidate_passes(tmp_path):
    f = hdr(
        tmp_path,
        f"inline void _llk_math_thing_() {{ {rmw(ZF, 1)} math::_invalidate_src_zero_flag_state_(); }}\n",
    )
    assert run(f).returncode == 0


def test_invalidate_before_the_write_does_not_cover(tmp_path):
    f = hdr(
        tmp_path,
        f"inline void _llk_math_thing_() {{ math::_invalidate_src_zero_flag_state_(); {rmw(ZF, 1)} }}\n",
    )
    assert run(f).returncode == 1


def test_tracker_writer_is_exempt(tmp_path):
    f = hdr(
        tmp_path,
        f"inline __attribute__((noinline)) void _apply_src_zero_flag_(const std::uint32_t value) {{ {rmw(ZF, 'value')} }}\n",
    )
    assert run(f).returncode == 0


def test_caller_in_same_file_that_invalidates_covers(tmp_path):
    f = hdr(
        tmp_path,
        f"""
        inline void enter_block() {{ {rmw(ZF, 1)} }}
        inline void run_block()
        {{
            enter_block();
            math::_invalidate_src_zero_flag_state_();
        }}
        """,
    )
    assert run(f).returncode == 0


def test_tracked_value_does_not_matter(tmp_path):
    """Even a raw write of the 'flush' value desyncs the cache."""
    f = hdr(tmp_path, f"inline void _llk_math_thing_() {{ {rmw(ZF, 0)} }}\n")
    assert run(f).returncode == 1


# ---- baseline, scope, table ----


def test_baseline_suppresses_by_function(tmp_path):
    f = hdr(tmp_path, f"inline void _llk_math_thing_() {{ {rmw(FP32, 0)} }}\n")
    rel = os.path.relpath(
        str(f), os.path.normpath(os.path.join(INFRA, "..", "..", ".."))
    )
    b = tmp_path / "b.txt"
    b.write_text(f"{rel}:_llk_math_thing_:{FP32}\n")
    assert run(f, baseline=b).returncode == 0
    b.write_text(f"{rel}:_llk_math_other_:{FP32}\n")
    assert run(f, baseline=b).returncode == 1


def test_only_touched_headers_are_checked(tmp_path):
    clean = hdr(tmp_path, "inline void _llk_math_thing_() {}\n", name="clean.h")
    hdr(tmp_path, f"inline void _llk_math_bad_() {{ {rmw(FP32, 0)} }}\n", name="bad.h")
    assert run(clean).returncode == 0


def test_bad_table_is_a_hard_error(tmp_path):
    f = hdr(tmp_path, "inline void _llk_math_thing_() {}\n")
    src = open(os.path.join(INFRA, "sticky_cfg_fields.yaml")).read()
    cases = [
        src.replace("    policy: tracked", "    policy: tracked\n    bogus_key: 1"),
        src.replace("    policy: restore", "    policy: sticky", 1),
        src.replace("ALU_ACC_CTRL_Fp32_enabled:", "ALU_ACC_CTRL_Not_A_Field:", 1),
        src.replace("sites: []", "sites:\n  - path: x.h\n    function: f\n"),
        src + "\nextra: 1\n",
        src + "\n  : [unbalanced\n",
    ]
    for i, text in enumerate(cases):
        assert text != src, f"case {i} did not change the table"
        t = tmp_path / f"t{i}.yaml"
        t.write_text(text)
        r = run(f, table=t)
        assert r.returncode == 1, f"case {i} accepted a bad table"
        assert "sticky_cfg" not in r.stdout or ".yaml:" in r.stdout


def test_shipped_tree_is_clean_against_baseline_and_not_vacuous():
    r = run(baseline=BASELINE)
    assert r.returncode == 0, r.stdout
    assert "no longer match" not in r.stdout
    unbaselined = run()
    assert (
        unbaselined.returncode == 1
    ), "whole-tree run found nothing -- the scan did not run"
    assert unbaselined.stdout.count("[tracked]") == sum(
        1 for l in open(BASELINE) if l.strip() and not l.startswith("#")
    )


def test_stale_baseline_entry_is_reported(tmp_path):
    b = tmp_path / "b.txt"
    b.write_text(open(BASELINE).read() + f"tt_metal/tt-llk/gone.h:_gone_:{FP32}\n")
    r = run(baseline=b)
    assert r.returncode == 0
    assert "no longer match" in r.stdout and "gone.h" in r.stdout


@pytest.mark.parametrize("field", [FP32, "ALU_ACC_CTRL_SFPU_Fp32_enabled", ZF])
def test_each_table_field_is_checked(tmp_path, field):
    f = hdr(tmp_path, f"inline void _llk_math_thing_() {{ {rmw(field, 0)} }}\n")
    assert run(f).returncode == 1
