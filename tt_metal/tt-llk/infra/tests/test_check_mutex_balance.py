#!/usr/bin/env python3
"""Tests for the Tensix mutex-balance checker."""
import contextlib
import io
import os
import runpy
import sys
import textwrap
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "check_mutex_balance.py")
LLK = os.path.normpath(os.path.join(HERE, "..", ".."))  # tt_metal/tt-llk


def run(*paths, baseline=None):
    """Run the checker as a script (`__main__`, as the hook does), in-process."""
    argv = [SCRIPT, *map(str, paths)]
    if baseline:
        argv += ["--baseline", str(baseline)]
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


def test_balanced_is_clean(tmp_path):
    f = hdr(
        tmp_path,
        "ok.h",
        """
        inline void _llk_unpack_thing_()
        {
            t6_mutex_acquire(mutex::REG_RMW);
            do_work();
            t6_mutex_release(mutex::REG_RMW);
        }
        """,
    )
    assert run(f).returncode == 0


def test_acquire_without_release_is_flagged(tmp_path):
    f = hdr(
        tmp_path,
        "leak.h",
        """
        inline void _llk_pack_thing_()
        {
            t6_mutex_acquire(mutex::REG_RMW);
        }
        """,
    )
    r = run(f)
    assert r.returncode == 1
    assert "acquired but never released" in r.stdout
    assert (
        "_llk_pack_thing_" in r.stdout
    ), "the signature must be reported, not the brace"


def test_release_without_acquire_is_flagged(tmp_path):
    f = hdr(
        tmp_path,
        "double.h",
        """
        inline void _llk_math_thing_()
        {
            t6_mutex_acquire(mutex::REG_RMW);
            t6_mutex_release(mutex::REG_RMW);
            t6_mutex_release(mutex::REG_RMW);
        }
        """,
    )
    r = run(f)
    assert r.returncode == 1 and "released without acquiring" in r.stdout


def test_raw_atgetm_atrelm_counted(tmp_path):
    """The raw instruction form is the same mutex; TTI_ATGETM without ATRELM leaks."""
    f = hdr(
        tmp_path,
        "raw.h",
        """
        inline void _llk_pack_raw_()
        {
            TTI_ATGETM(mutex::REG_RMW);
        }
        """,
    )
    assert run(f).returncode == 1


def test_two_overloads_do_not_mask_each_other(tmp_path):
    """Balance is per body, not per name: a leak in one overload must still be seen."""
    f = hdr(
        tmp_path,
        "ovl.h",
        """
        inline void _llk_pack_thing_(int a)
        {
            t6_mutex_acquire(mutex::REG_RMW);
            t6_mutex_release(mutex::REG_RMW);
        }

        inline void _llk_pack_thing_(float b)
        {
            t6_mutex_acquire(mutex::REG_RMW);
        }
        """,
    )
    r = run(f)
    assert r.returncode == 1, "the leaking overload must be reported"


def test_namespace_does_not_merge_functions(tmp_path):
    """Every LLK header wraps its functions in `namespace ckernel`; the brace must not scope a body."""
    f = hdr(
        tmp_path,
        "ns.h",
        """
        namespace ckernel
        {
        inline void _llk_math_leaks_()
        {
            t6_mutex_acquire(mutex::REG_RMW);
        }

        inline void _llk_math_frees_()
        {
            t6_mutex_release(mutex::REG_RMW);
        }
        }
        """,
    )
    r = run(f)
    assert r.returncode == 1, "a leak and an unrelated release must not cancel"
    assert "_llk_math_leaks_" in r.stdout and "_llk_math_frees_" in r.stdout, r.stdout


def test_guard_exemption_does_not_cover_the_rest_of_the_file(tmp_path):
    """Only T6MutexLockGuard's own body is exempt, not every function beside it."""
    f = hdr(
        tmp_path,
        "guard_plus_leak.h",
        """
        namespace ckernel
        {
        class T6MutexLockGuard final
        {
        public:
            explicit T6MutexLockGuard(const std::uint8_t i) : mutex_index(i)
            {
                t6_mutex_acquire(mutex_index);
            }

            ~T6MutexLockGuard()
            {
                t6_mutex_release(mutex_index);
            }

        private:
            const std::uint8_t mutex_index;
        };

        inline void _llk_pack_leaks_()
        {
            t6_mutex_acquire(mutex::REG_RMW);
        }
        }
        """,
    )
    r = run(f)
    assert r.returncode == 1, "a raw leak beside the guard class must still be reported"
    assert "_llk_pack_leaks_" in r.stdout, r.stdout
    # The guard's own ctor and dtor stay exempt, so the leak is the only finding.
    assert "1 function(s)" in r.stdout, r.stdout


# Declarator shapes whose brace or keyword once hid the body from the check. `{acq}` is the
# acquire, `{rel}` the release (empty in the leaking variant); `name` must be reported.
SHAPES = {
    "template_class_param": (
        "_llk_tmpl_",
        """
        template <class T, typename std::enable_if<(sizeof(T) > 1), int>::type = 0>
        inline void _llk_tmpl_(T x)
        {{
            {acq}
            {rel}
        }}
        """,
    ),
    "enum_param": (
        "_llk_enum_",
        """
        inline void _llk_enum_(enum Mode m)
        {{
            {acq}
            {rel}
        }}
        """,
    ),
    "brace_default_argument": (
        "_llk_dflt_",
        """
        struct Cfg {{ int a; }};
        inline void _llk_dflt_(Cfg c = {{}}, int k = int{{3}})
        {{
            {acq}
            {rel}
        }}
        """,
    ),
    "ctor_member_brace_init": (
        "Guardless",
        """
        class Guardless
        {{
            int m;

          public:
            Guardless() : m{{0}}
            {{
                {acq}
                {rel}
            }}
        }};
        """,
    ),
    "template_argument_call": (
        "_llk_targ_",
        """
        inline void _llk_targ_()
        {{
            t6_mutex_acquire<mutex::REG_RMW>();
            {rel_t}
        }}
        """,
    ),
    "lambda_argument": (
        "[](const std::uint32_t m)",
        """
        static constexpr auto _table_ = make_table<8>(
            [](const std::uint32_t m) -> std::uint32_t
            {{
                {acq}
                {rel}
                return m;
            }});
        """,
    ),
    "macro_defined_function": (
        "vector_##op",
        """
        #define DEFINE_VECTOR_OP(op) \\
            inline void vector_##op() \\
            {{ \\
                {acq} \\
                {rel} \\
            }}
        """,
    ),
    "digit_separator_before_body": (
        "_llk_after_sep_",
        """
        #define PULSE (0x8000'0000)
        constexpr int bits = 0b00'01;
        inline void _llk_after_sep_()
        {{
            {acq}
            {rel}
        }}
        """,
    ),
}


def _shape(body, leak):
    acq, rel = "t6_mutex_acquire(mutex::REG_RMW);", "t6_mutex_release(mutex::REG_RMW);"
    rel_t = "t6_mutex_release<mutex::REG_RMW>();"
    src = body.format(acq=acq, rel="" if leak else rel, rel_t="" if leak else rel_t)
    return "namespace ckernel\n{\n" + textwrap.dedent(src) + "}\n"


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_declarator_shape_leak_is_flagged(tmp_path, shape):
    name, body = SHAPES[shape]
    p = tmp_path / f"{shape}.h"
    p.write_text(_shape(body, leak=True))
    r = run(p)
    assert r.returncode == 1, f"{shape}: leak missed\n{p.read_text()}"
    assert name in r.stdout, r.stdout


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_declarator_shape_balanced_is_clean(tmp_path, shape):
    _, body = SHAPES[shape]
    p = tmp_path / f"{shape}.h"
    p.write_text(_shape(body, leak=False))
    r = run(p)
    assert r.returncode == 0, f"{shape}: false positive\n{r.stdout}"


def test_wrapper_definitions_are_exempt():
    """t6_mutex_acquire/release each hold one half by definition."""
    seen = 0
    for arch in ("tt_llk_wormhole_b0", "tt_llk_blackhole"):
        p = os.path.join(LLK, arch, "common", "inc", "ckernel.h")
        if os.path.exists(p):
            seen += 1
            assert run(p).returncode == 0, p
    assert seen, f"no ckernel.h found under {LLK} -- test would pass vacuously"


def test_raii_guard_is_exempt():
    """T6MutexLockGuard acquires in its ctor and releases in its dtor: balanced per object."""
    p = os.path.join(LLK, "tt_llk_blackhole", "common", "inc", "ckernel_mutex_guard.h")
    assert os.path.exists(p), f"{p} not found -- test would pass vacuously"
    assert run(p).returncode == 0


def test_shipped_trees_are_clean():
    """The measured baseline: zero imbalances across every LLK tree, so any finding is new."""
    files = [
        os.path.join(d, f)
        for tree in ("tt_llk_wormhole_b0", "tt_llk_blackhole", "tt_llk_quasar")
        for d, _, fs in os.walk(os.path.join(LLK, tree))
        for f in fs
        if f.endswith(".h")
    ]
    assert files, f"no headers found under {LLK} -- test would skip vacuously"
    r = run(*files)
    assert r.returncode == 0, r.stdout


def test_baseline_suppresses(tmp_path):
    f = hdr(
        tmp_path,
        "leak.h",
        """
        inline void _llk_pack_thing_()
        {
            t6_mutex_acquire(mutex::REG_RMW);
        }
        """,
    )
    assert run(f).returncode == 1
    b = tmp_path / "base.txt"
    b.write_text(f"{os.path.relpath(f)}:inline void _llk_pack_thing_()\n")
    assert run(f, baseline=b).returncode == 0
