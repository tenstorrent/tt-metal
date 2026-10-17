#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
# SPDX-License-Identifier: Apache-2.0
"""include/l2cpu_sampling.h <-> host/l2cpu/sampling/layout.py agreement, and the header's static asserts.

1. The mirror is current (digest of l2cpu_link.h + l2cpu_sampling.h).
2. A C program printing every mirrored constant (host gcc) gives exactly the Python values.
3. The header compiles with riscv64-unknown-elf-gcc (rv64 C) and, when found, the tt-metal riscv32 JIT compiler
   (C++), so Tensix kernels can include it. Runs as a script or under pytest.
"""
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLING = os.path.dirname(HERE)
L2CPU = os.path.dirname(SAMPLING)
REPO = os.path.dirname(os.path.dirname(L2CPU))
INCS = [os.path.join(SAMPLING, "include"), os.path.join(L2CPU, "include"), os.path.join(L2CPU, "tensix", "kernels")]
MIRROR = os.path.join(L2CPU, "host", "l2cpu", "sampling", "layout.py")
sys.path.insert(0, os.path.join(SAMPLING, "tools"))
import gen_sampling_py  # noqa: E402


def _mirror():
    spec = importlib.util.spec_from_file_location("l2s_layout", MIRROR)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _inc():
    return sum((["-I", d] for d in INCS), [])


def _rv32_compiler():
    for base in (os.environ.get("TT_METAL_HOME", ""), REPO):
        p = os.path.join(base, "runtime", "sfpi", "compiler", "bin", "riscv-tt-elf-g++")
        if base and os.path.exists(p):
            return p
    return None


def test_mirror_current():
    assert _mirror().HEADER_SHA256_16 == gen_sampling_py.digest(), "run sampling/tools/gen_sampling_py.py"


def test_constants_match_c():
    m = _mirror()
    names = list(m.CONSTANTS)
    src = ["#include <stdio.h>", '#include "l2cpu_sampling.h"', "int main(void) {"]
    src += [f'  printf("{n} %llu\\n", (unsigned long long)({n}));' for n in names] + ["  return 0;", "}"]
    with tempfile.TemporaryDirectory() as tmp:
        c, exe = os.path.join(tmp, "t.c"), os.path.join(tmp, "t")
        open(c, "w").write("\n".join(src) + "\n")
        subprocess.run(["gcc", "-std=c11", "-Wall", "-Werror", *_inc(), "-o", exe, c], check=True)
        out = dict(
            l.split(" ", 1)
            for l in subprocess.run([exe], check=True, capture_output=True, text=True).stdout.splitlines()
        )
    bad = [n for n in names if int(out[n]) != m.CONSTANTS[n]]
    assert not bad, bad[:10]


def _static_asserts_cross_abi():
    rv64 = shutil.which("riscv64-unknown-elf-gcc")
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "h.c")
        open(f, "w").write('#include "l2cpu_sampling.h"\nint l2s_dummy;\n')
        if rv64:
            subprocess.run(
                [
                    rv64,
                    "-march=rv64gc",
                    "-mabi=lp64d",
                    "-std=c11",
                    "-ffreestanding",
                    "-Wall",
                    "-Werror",
                    *_inc(),
                    "-c",
                    "-o",
                    os.path.join(tmp, "h64.o"),
                    f,
                ],
                check=True,
            )
        rv32 = _rv32_compiler()
        if rv32:
            g = os.path.join(tmp, "h.cpp")
            shutil.copy(f, g)
            subprocess.run(
                [
                    rv32,
                    "-mcpu=tt-bh",
                    "-std=c++17",
                    "-Wall",
                    "-Werror",
                    *_inc(),
                    "-c",
                    "-o",
                    os.path.join(tmp, "h32.o"),
                    g,
                ],
                check=True,
            )
    return bool(rv64), bool(rv32)


def test_static_asserts_cross_abi():
    _static_asserts_cross_abi()


if __name__ == "__main__":
    test_mirror_current()
    test_constants_match_c()
    rv64, rv32 = _static_asserts_cross_abi()
    print(
        f"SAMPLING MIRROR PASS ({len(_mirror().CONSTANTS)} constants; rv64 {'checked' if rv64 else 'not found'}, "
        f"rv32 JIT compiler {'checked' if rv32 else 'not found'})"
    )
