#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for the replay-record helper checker."""
import importlib.util
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "check_replay_record_helper.py")


def load_module():
    spec = importlib.util.spec_from_file_location("check_replay_record_helper", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = load_module()


def findings(
    text, rel="tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_x.h"
):
    return [msg for _, _, msg in MOD.check_text(rel, text)]


@pytest.mark.parametrize(
    "code",
    [
        "lltt::record(0, 4);",
        "lltt::record<lltt::Exec>(start, len);",
        "lltt :: record(0, 4);",
        "TTI_REPLAY(0, 4, 0, 1);",
        "TT_REPLAY(start, len, 1, 1);",
        "std::uint32_t w = TT_OP_REPLAY(0, 4, 0, 1);",
        "TTI_REPLAY(0, 4, 0, load_mode);",  # not provably a playback
        "TTI_REPLAY(0, 4, 0, (1));",
        "TT_REPLAY_VALID(0, 4, 0, 1);",
    ],
)
def test_record_outside_helper_is_flagged(code):
    assert findings(f"void f() {{ {code} }}\n"), code


@pytest.mark.parametrize(
    "code",
    [
        "TTI_REPLAY(0, 4, 0, 0);",
        "TT_REPLAY(start, len, 0, false);",
        "std::uint32_t w = TT_OP_REPLAY(start, len, 0, 0);",
        "lltt::replay(0, 4);",
        "std::uint32_t w = lltt::replay_insn(0, 4);",
        "load_replay_buf(0, 4, [] { TTI_NOP; });",
        "// lltt::record(0, 4);",
        "/* TTI_REPLAY(0, 4, 0, 1); */",
        'const char *s = "lltt::record(0, 4)";',
    ],
)
def test_playback_helper_and_comments_are_clean(code):
    assert not findings(f"void f() {{ {code} }}\n"), code


def test_macro_definition_is_not_a_use():
    text = "#define TTI_REPLAY(a, b, c, d) \\\n    TTI_INSN(TT_OP_REPLAY(a, b, c, d))\n"
    assert not findings(text)


def test_record_inside_helper_body_is_allowed_only_in_the_helper_files():
    text = (
        "template <ExecBool Exec = NoExec, typename Callable, typename... Args>\n"
        "inline void load_replay_buf(std::uint32_t start, std::uint32_t len, Callable &&c, Args &&...a)\n"
        "{\n    lltt::record<lltt::ExecBool(Exec)>(start, len);\n    c(a...);\n}\n"
        "void g() { lltt::record(0, 1); }\n"
    )
    helper = "tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel.h"
    assert len(MOD.check_text(helper, text)) == 1  # g's record only
    assert len(findings(text)) == 2  # same body anywhere else is not the helper


@pytest.mark.parametrize(
    "rel, excluded",
    [
        ("tt_metal/tt-llk/tt_llk_quasar/llk_lib/llk_math_reduce.h", True),
        ("tt_metal/hw/ckernels/quasar/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h", True),
        ("tt_metal/third_party/x/y.h", True),
        ("tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_reduce.h", False),
        ("ttnn/cpp/ttnn/operations/x/device/kernels/compute.cpp", False),
    ],
)
def test_scope(rel, excluded):
    assert bool(MOD.EXCLUDED.search(rel)) == excluded


def test_repository_is_clean():
    run = subprocess.run([sys.executable, SCRIPT], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
