# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Which named part of an LLK 2.0 op's ``accepts`` refuses a call
(``llk2/inc/common/resolver/resolver.h``, ``Accepts``; the Blackhole Resolvers in
``llk2/inc/blackhole/resolver/``). The Compute API prints one sentence per part,
so the part decides the message a kernel author reads.

The checks are static_asserts in ``llk2/tests/host/accepts_test.cpp``: building
it is the test. Device compiler and the kernels' flags, ``-fsyntax-only``: no
card, no simulator.
"""

import subprocess
from pathlib import Path

import pytest

_TT_LLK = Path(__file__).resolve().parents[3]
_LLK2 = _TT_LLK / "llk2"
_SOURCE = _LLK2 / "tests/host/accepts_test.cpp"
_INCLUDES = [
    f"-I{_LLK2 / 'tests/host/stubs'}",  # llk_assert.h
    f"-I{_LLK2 / 'inc/common'}",
    f"-I{_LLK2 / 'inc/blackhole'}",
    f"-I{_TT_LLK.parent / 'hw/inc/internal/tt-1xx/blackhole'}",  # tensix_types.h: DataFormat
]
# What the harness and tt-metal's JIT compile kernels with (helpers/test_config.py, jit_build/build.cpp).
_FLAGS = [
    "-std=c++17",
    "-mcpu=tt-bh-tensix",
    "-ftt-nttp",
    "-ftt-constinit",
    "-ftt-consteval",
    "-ftt-no-dyninit",
    "-fno-exceptions",
    "-fno-rtti",
    "-Wall",
    "-Wextra",
    "-Werror",
    "-DTENSIX_FIRMWARE",
    "-DLLK2_DEST_FP32=0",
    "-DLLK2_MATH_FIDELITY=0",
]
_DEVICE_COMPILER = _TT_LLK / "tests/sfpi/compiler/bin/riscv-tt-elf-g++"

needs_device_compiler = pytest.mark.skipif(
    not _DEVICE_COMPILER.exists(), reason="no sfpi compiler"
)


def _compile(*args):
    return subprocess.run(
        [
            str(_DEVICE_COMPILER),
            *_FLAGS,
            *_INCLUDES,
            "-fsyntax-only",
            *args,
            str(_SOURCE),
        ],
        capture_output=True,
        text=True,
    )


@needs_device_compiler
def test_each_refused_call_names_its_part():
    built = _compile()
    assert built.returncode == 0, built.stderr


@needs_device_compiler
def test_a_wrong_part_fails_the_build():
    """The checks can fail: with the expected part of one case changed, the build stops at that case."""
    built = _compile("-DEXPECT_FAIL=1")
    assert built.returncode != 0, "EXPECT_FAIL=1 compiled"
    assert "refused by layout alone: matmul_op(a24, t11)" in built.stderr, built.stderr
