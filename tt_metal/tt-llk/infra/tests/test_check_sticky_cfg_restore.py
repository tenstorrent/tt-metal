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
LLK = os.path.normpath(os.path.join(INFRA, ".."))
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


def test_uninit_writing_the_same_value_is_not_a_restore(tmp_path):
    f = hdr(
        tmp_path,
        f"""
        inline void _llk_math_thing_init_() {{ {rmw(FP32, 0)} }}
        inline void _llk_math_thing_uninit_() {{ {rmw(FP32, 0)} }}
        """,
    )
    r = run(f)
    assert r.returncode == 1
    assert "[restore]" in r.stdout and "writes the same value" in r.stdout


def test_uninit_writing_the_configured_value_restores(tmp_path):
    f = hdr(
        tmp_path,
        f"""
        inline void _llk_math_thing_init_() {{ {rmw(FP32, 0)} }}
        template <bool is_fp32_dest_acc_en>
        inline void _llk_math_thing_uninit_() {{ {rmw(FP32, "is_fp32_dest_acc_en")} }}
        """,
    )
    r = run(f)
    assert r.returncode == 0 and "note" not in r.stdout


def test_uninit_writing_an_unjudgeable_value_is_a_note(tmp_path):
    f = hdr(
        tmp_path,
        f"""
        inline void _llk_math_thing_init_() {{ {rmw(FP32, 0)} }}
        inline void _llk_math_thing_uninit_(bool v) {{ {rmw(FP32, "v")} }}
        """,
    )
    r = run(f)
    assert r.returncode == 0
    assert "cannot be judged" in r.stdout and "_llk_math_thing_uninit_" in r.stdout


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


def test_computed_value_after_a_fixed_write_is_a_note(tmp_path):
    f = hdr(
        tmp_path,
        f"""
        inline void _llk_math_thing_(std::uint32_t x)
        {{
            {rmw(FP32, 0)}
            {rmw(FP32, "x")}
        }}
        """,
    )
    r = run(f)
    assert r.returncode == 0
    assert "cannot be judged" in r.stdout and "assumed to put back" in r.stdout


def test_configured_value_after_a_fixed_write_is_silent(tmp_path):
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
    r = run(f)
    assert r.returncode == 0 and "note" not in r.stdout


def test_unresolved_value_is_a_note_not_a_failure(tmp_path):
    f = hdr(tmp_path, f"inline void _llk_math_thing_(bool v) {{ {rmw(FP32, 'v')} }}\n")
    r = run(f)
    assert r.returncode == 0
    assert "cannot be judged" in r.stdout


def site_table(tmp_path, path, function, field, restored_by="_llk_math_thing_done_"):
    rel = os.path.relpath(
        str(path), os.path.normpath(os.path.join(INFRA, "..", "..", ".."))
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
                    function: {function}
                    field: {field}
                    restored_by: {restored_by}
                    reason: the op's done hook puts the field back
                """
            ),
        )
    )
    return t


def test_table_site_covers(tmp_path):
    f = hdr(
        tmp_path,
        f"""
        inline void _llk_math_thing_() {{ {rmw(FP32, 0)} }}
        template <bool is_fp32_dest_acc_en>
        inline void _llk_math_thing_done_() {{ {rmw(FP32, "is_fp32_dest_acc_en")} }}
        """,
    )
    t = site_table(tmp_path, f, "_llk_math_thing_", FP32)
    assert run(f).returncode == 1
    assert run(f, table=t).returncode == 0


def test_table_site_covers_tracked_field(tmp_path):
    """A site also records a tracked write whose invalidate lives where the check cannot see."""
    f = hdr(
        tmp_path,
        f"""
        inline void _llk_math_thing_() {{ {rmw(ZF, 1)} }}
        inline void _llk_math_thing_done_() {{ math::_invalidate_src_zero_flag_state_(); }}
        """,
    )
    t = site_table(tmp_path, f, "_llk_math_thing_", ZF)
    assert run(f).returncode == 1
    assert run(f, table=t).returncode == 0


def test_table_site_restorer_must_exist(tmp_path):
    f = hdr(tmp_path, f"inline void _llk_math_thing_() {{ {rmw(FP32, 0)} }}\n")
    t = site_table(
        tmp_path, f, "_llk_math_thing_", FP32, restored_by="_no_such_function_"
    )
    r = run(f, table=t)
    assert r.returncode == 1
    assert ".yaml:" in r.stdout and "_no_such_function_" in r.stdout


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


def test_baseline_key_is_shared_by_overloads(tmp_path):
    """The key is path:function:field, so same-named functions in a file share one entry."""
    f = hdr(
        tmp_path,
        f"""
        inline void _llk_math_thing_(int a) {{ {rmw(FP32, 0)} }}
        inline void _llk_math_thing_(int a, int b) {{ {rmw(FP32, 0)} }}
        """,
    )
    assert run(f).stdout.count("[restore]") == 2
    rel = os.path.relpath(
        str(f), os.path.normpath(os.path.join(INFRA, "..", "..", ".."))
    )
    b = tmp_path / "b.txt"
    b.write_text(f"{rel}:_llk_math_thing_:{FP32}\n")
    assert run(f, baseline=b).returncode == 0


def test_missing_pyyaml_is_a_table_error(tmp_path):
    f = hdr(tmp_path, "inline void _llk_math_thing_() {}\n")
    saved = sys.modules.get("yaml")
    sys.modules["yaml"] = None  # makes `import yaml` raise ImportError
    try:
        r = run(f)
    finally:
        if saved is None:
            del sys.modules["yaml"]
        else:
            sys.modules["yaml"] = saved
    assert r.returncode == 1
    assert "PyYAML is required" in r.stdout


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


def test_shipped_tree_is_clean_and_scan_is_not_vacuous(tmp_path):
    r = run()
    assert r.returncode == 0, r.stdout
    # A clean tree must not be a scan that saw nothing: a shipped header that clears Fp32_enabled
    # and sets it back passes as is, and fails once every restoring write is dropped from a copy.
    name = "llk_math_eltwise_unary_datacopy.h"
    src = open(os.path.join(LLK, "tt_llk_blackhole", "llk_lib", name)).read()
    assert rmw(FP32, 0) in src and rmw(FP32, 1) in src
    assert run(hdr(tmp_path, src, name=name)).returncode == 0
    broken = run(hdr(tmp_path, src.replace(rmw(FP32, 1), ""), name=name))
    assert broken.returncode == 1 and "[restore]" in broken.stdout


def test_stale_baseline_entry_is_reported(tmp_path):
    b = tmp_path / "b.txt"
    b.write_text(f"tt_metal/tt-llk/gone.h:_gone_:{FP32}\n")
    r = run(baseline=b)
    assert r.returncode == 0
    assert "no longer match" in r.stdout and "gone.h" in r.stdout


@pytest.mark.parametrize("field", [FP32, "ALU_ACC_CTRL_SFPU_Fp32_enabled", ZF])
def test_each_table_field_is_checked(tmp_path, field):
    f = hdr(tmp_path, f"inline void _llk_math_thing_() {{ {rmw(field, 0)} }}\n")
    assert run(f).returncode == 1
