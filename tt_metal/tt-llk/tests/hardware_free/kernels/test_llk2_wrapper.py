# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
LLK 2.0's compile-time / run-time value wrappers (``llk2/inc/common/wrapper/``):
``ct``, ``rt`` and ``ctrt``.

The rules are static_asserts in ``llk2/tests/host/wrapper_test.cpp``, so building
it is most of the test. This driver builds and runs it with the host compiler
(plain C++17), builds it for the device with the sfpi compiler and the kernels'
flags (at -O0 too: a ``ct`` must not need the optimizer), and builds each
``-DEXPECT_FAIL=<n>`` case, which holds one statement the wrappers must refuse,
with both compilers, checking that the compiler does refuse it.

Compile-only on the device side: no card, no simulator.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

_TT_LLK = Path(__file__).resolve().parents[3]
_HOST_TESTS = _TT_LLK / "llk2/tests/host"
_SOURCE = _HOST_TESTS / "wrapper_test.cpp"
_INCLUDES = [f"-I{_TT_LLK / 'llk2/inc/common'}"]
_FLAGS = ["-std=c++17", "-Wall", "-Wextra", "-Werror"]
# What the harness and tt-metal's JIT compile kernels with (helpers/test_config.py, jit_build/build.cpp).
_DEVICE_FLAGS = [
    "-mcpu=tt-bh-tensix",
    "-ftt-nttp",
    "-ftt-constinit",
    "-ftt-consteval",
    "-ftt-no-dyninit",
    "-fno-exceptions",
    "-fno-rtti",
    "-fno-use-cxa-atexit",
]
_DEVICE_COMPILER = _TT_LLK / "tests/sfpi/compiler/bin/riscv-tt-elf-g++"
_HOST_COMPILER = shutil.which("g++")

# EXPECT_FAIL case -> the pieces of the diagnostic that show it failed for the right reason: what was refused, of
# which type.
_MUST_NOT_COMPILE = {
    1: ("cannot convert", "llk2::ct<5>", "std::uint32_t"),
    2: ("conversion from", "llk2::ct<5>", "llk2::rt<"),
    3: ("no matching function", "sum_to(llk2::ct<-4>)"),
    4: ("could not convert", "llk2::ct<5>", "bool"),
}

needs_host_compiler = pytest.mark.skipif(_HOST_COMPILER is None, reason="no host g++")
needs_device_compiler = pytest.mark.skipif(
    not _DEVICE_COMPILER.exists(), reason="no sfpi compiler"
)


# Every command this file runs, built once from the constants above. A test picks one by its parameter and passes it
# to subprocess.run as it is; outputs get a fixed name and land in the test's tmp_path through cwd.
_HOST_BUILD = {
    opt: [
        str(_HOST_COMPILER),
        *_FLAGS,
        *_INCLUDES,
        opt,
        "-o",
        "wrapper_test",
        str(_SOURCE),
    ]
    for opt in ("-O0", "-O2")
}
_DEVICE_BUILD = {
    opt: [
        str(_DEVICE_COMPILER),
        *_FLAGS,
        *_INCLUDES,
        *_DEVICE_FLAGS,
        opt,
        "-c",
        "-o",
        "wrapper_test.o",
        str(_SOURCE),
    ]
    for opt in ("-O0", "-O3")
}
_HOST_REFUSE = {
    case: [
        str(_HOST_COMPILER),
        *_FLAGS,
        *_INCLUDES,
        "-fsyntax-only",
        f"-DEXPECT_FAIL={case}",
        str(_SOURCE),
    ]
    for case in _MUST_NOT_COMPILE
}
_DEVICE_REFUSE = {
    case: [
        str(_DEVICE_COMPILER),
        *_FLAGS,
        *_INCLUDES,
        *_DEVICE_FLAGS,
        "-fsyntax-only",
        f"-DEXPECT_FAIL={case}",
        str(_SOURCE),
    ]
    for case in _MUST_NOT_COMPILE
}


@needs_host_compiler
@pytest.mark.parametrize("opt", sorted(_HOST_BUILD))
def test_wrapper_rules_on_host(tmp_path, opt):
    built = subprocess.run(
        _HOST_BUILD[opt], cwd=tmp_path, capture_output=True, text=True
    )
    assert built.returncode == 0, built.stderr
    assert subprocess.run(["./wrapper_test"], cwd=tmp_path).returncode == 0


@needs_device_compiler
@pytest.mark.parametrize("opt", sorted(_DEVICE_BUILD))
def test_wrapper_rules_on_device_compiler(tmp_path, opt):
    built = subprocess.run(
        _DEVICE_BUILD[opt], cwd=tmp_path, capture_output=True, text=True
    )
    assert built.returncode == 0, built.stderr


def _assert_refused(built, case):
    assert built.returncode != 0, f"EXPECT_FAIL={case} compiled"
    for piece in _MUST_NOT_COMPILE[case]:
        assert piece in built.stderr, built.stderr


@needs_host_compiler
@pytest.mark.parametrize("case", sorted(_MUST_NOT_COMPILE))
def test_wrapper_refuses_on_host(case):
    built = subprocess.run(_HOST_REFUSE[case], capture_output=True, text=True)
    _assert_refused(built, case)


@needs_device_compiler
@pytest.mark.parametrize("case", sorted(_MUST_NOT_COMPILE))
def test_wrapper_refuses_on_device_compiler(case):
    built = subprocess.run(_DEVICE_REFUSE[case], capture_output=True, text=True)
    _assert_refused(built, case)
