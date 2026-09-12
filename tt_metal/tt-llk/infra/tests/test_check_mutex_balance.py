#!/usr/bin/env python3
"""Tests for the Tensix mutex-balance checker."""
import os
import subprocess
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "check_mutex_balance.py")
LLK = os.path.normpath(os.path.join(HERE, "..", ".."))  # tt_metal/tt-llk


def run(*paths, baseline=None):
    cmd = [sys.executable, SCRIPT, *map(str, paths)]
    if baseline:
        cmd += ["--baseline", str(baseline)]
    return subprocess.run(cmd, capture_output=True, text=True)


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
