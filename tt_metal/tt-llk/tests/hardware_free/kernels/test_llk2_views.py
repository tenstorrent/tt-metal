# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
LLK 2.0's Tensor views (``llk2/inc/common/tensor/tensor.h``: ``narrow``,
``narrow_up_to``, ``select``, ``row``, ``rows``, ``rows_up_to``, ``cols``, ``cols_up_to``,
``tile``) and the argument kinds of ``wait_front`` / ``reserve_back`` /
``pop_front`` (``llk2/inc/common/compute_api/dataflow.h``).

What holds at compile time is static_asserts in
``llk2/tests/helpers/tensor_views_selftest.h``; ``llk2/tests/host/tensor_views_test.cpp``
includes it and adds what only a run can check (a run-time start or length
outside the tensor reaches LLK_ASSERT). This driver builds and runs it with the
host compiler (C++20: the specs are class-type template arguments), builds it
for the device with the sfpi compiler and the kernels' flags, and builds each
``-DEXPECT_FAIL=<n>`` case, which holds one call the views or the buffers must
refuse, checking that the compiler refuses it with the message that says why.

Compile-only on the device side: no card, no simulator.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

_TT_LLK = Path(__file__).resolve().parents[3]
_HOST_TESTS = _TT_LLK / "llk2/tests/host"
_SOURCE = _HOST_TESTS / "tensor_views_test.cpp"
_INCLUDES = [
    f"-I{_HOST_TESTS / 'stubs'}",  # llk_assert.h: counts failed LLK_ASSERTs
    f"-I{_TT_LLK / 'llk2/tests/helpers'}",
    f"-I{_TT_LLK / 'llk2/inc/common'}",
    f"-I{_TT_LLK.parent / 'hw/inc/internal/tt-1xx/blackhole'}",  # tensix_types.h: DataFormat
]
_FLAGS = ["-Wall", "-Wextra", "-Werror", "-DTENSIX_FIRMWARE"]
_HOST_FLAGS = ["-std=c++20"]
# What the harness and tt-metal's JIT compile kernels with (helpers/test_config.py, jit_build/build.cpp).
_DEVICE_FLAGS = [
    "-std=c++17",
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

# EXPECT_FAIL case -> a piece of the diagnostic that shows it failed for the right reason.
_MUST_NOT_COMPILE = {
    1: "narrow: start + length is past the tensor's end",
    2: "narrow: the length is from 1 to the tensor's size along the dim",
    3: "a view is of the tile grid",
    4: "narrow_up_to: the length's bound is from 1 to the tensor's size along the dim",
    5: "a view of a tensor with a run-time size: not ported",
    6: "tile(t, i) is for a tensor one tile wide",
    7: "wait_front(buffer, count): the spec marks no run-time size",
    8: "wait_front(buffer, when): a spec with a run-time size takes its count",
    9: "reserve_back(buffer, count): the spec marks no run-time size",
    10: "reserve_back(buffer, when): a spec with a run-time size takes its count",
    11: "pop_front<S>: S holds a bound",
    12: "a tensor whose spec marks a run-time size takes its count",
    13: "a run-time number of tile columns: of one tile row",
}


def _host_has_cpp20():
    if _HOST_COMPILER is None:
        return False
    probe = subprocess.run(
        [_HOST_COMPILER, "-std=c++20", "-fsyntax-only", "-x", "c++", "-"],
        input="struct S { int v; }; template <S s> struct T {}; T<S {1}> t;",
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


needs_host_compiler = pytest.mark.skipif(
    not _host_has_cpp20(), reason="no host g++ with C++20 class-type template arguments"
)
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
        *_HOST_FLAGS,
        opt,
        "-o",
        "tensor_views_test",
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
        "tensor_views_test.o",
        str(_SOURCE),
    ]
    for opt in ("-O0", "-O3")
}
_HOST_REFUSE = {
    case: [
        str(_HOST_COMPILER),
        *_FLAGS,
        *_INCLUDES,
        *_HOST_FLAGS,
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
def test_views_on_host(tmp_path, opt):
    built = subprocess.run(
        _HOST_BUILD[opt], cwd=tmp_path, capture_output=True, text=True
    )
    assert built.returncode == 0, built.stderr
    assert subprocess.run(["./tensor_views_test"], cwd=tmp_path).returncode == 0


@needs_device_compiler
@pytest.mark.parametrize("opt", sorted(_DEVICE_BUILD))
def test_views_on_device_compiler(tmp_path, opt):
    built = subprocess.run(
        _DEVICE_BUILD[opt], cwd=tmp_path, capture_output=True, text=True
    )
    assert built.returncode == 0, built.stderr


def _assert_refused(built, case):
    assert built.returncode != 0, f"EXPECT_FAIL={case} compiled"
    assert _MUST_NOT_COMPILE[case] in built.stderr, built.stderr


@needs_host_compiler
@pytest.mark.parametrize("case", sorted(_MUST_NOT_COMPILE))
def test_views_refuse_on_host(case):
    built = subprocess.run(_HOST_REFUSE[case], capture_output=True, text=True)
    _assert_refused(built, case)


@needs_device_compiler
@pytest.mark.parametrize("case", sorted(_MUST_NOT_COMPILE))
def test_views_refuse_on_device_compiler(case):
    built = subprocess.run(_DEVICE_REFUSE[case], capture_output=True, text=True)
    _assert_refused(built, case)
