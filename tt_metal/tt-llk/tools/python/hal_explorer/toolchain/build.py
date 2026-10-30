# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Wrap the input, compile it with the Blackhole HAL lit flags and return GCC's dump."""

import os
import re
import signal
import subprocess
import tempfile
from pathlib import Path

LLK_ROOT = Path(__file__).resolve().parents[4]
INPUT_FILE = "input"
PARAMETERS_FILE = "parameters"
BODY_FUNCTION = "hal_replay_body"
COMPILE_TIMEOUT_SECONDS = 120

HAL_DIR = LLK_ROOT / "tt_llk_blackhole" / "llk_lib" / "hal"
# Header -> namespace aliases it provides. Only headers present in this checkout are
# used, so the tool works on any branch of the HAL stack.
HAL_HEADERS = {
    "atomic.h": {"atomic": "hal::atomic"},
    "cfg.h": {"cfg": "hal::cfg"},
    "misc.h": {"misc": "hal::misc"},
    "mop.h": {"mop": "hal::mop"},
    "replay.h": {"replay": "hal::replay"},
    "scalar.h": {"scalar": "hal::scalar"},
    "sync.h": {"sync": "hal::sync"},
    "tensix_counters.h": {
        "adc": "hal::counters::adc",
        "rwc": "hal::counters::rwc",
        "counters": "hal::counters",
    },
}


def available_headers(hal_dir=HAL_DIR):
    return [name for name in HAL_HEADERS if (Path(hal_dir) / name).is_file()]


def includes(hal_dir=HAL_DIR):
    return '#include "ckernel.h"\n' + "".join(
        f'#include "hal/{name}"\n' for name in available_headers(hal_dir)
    )


def aliases(hal_dir=HAL_DIR):
    lines = [
        f"namespace {alias:<8} = {target};"
        for name in available_headers(hal_dir)
        for alias, target in HAL_HEADERS[name].items()
    ]
    return "\n".join(lines + ["using namespace ckernel;"]) + "\n"


INCLUDES = includes()
ALIASES = aliases()

DIAGNOSTIC = re.compile(r"^(input|parameters):(\d+):(\d+): (error|warning): (.*)$")


class BuildError(Exception):
    """The input could not be compiled; the message is meant for the user."""


def compile_flags(llk_root=LLK_ROOT):
    """The compile line of tools/tests/unit/lit/hal/blackhole/lit.local.cfg (math thread)."""
    llk_root = Path(llk_root)
    blackhole = llk_root / "tt_llk_blackhole"
    metal_inc = llk_root.parent / "hw" / "inc"
    includes = [
        llk_root / "tests" / "sfpi" / "include",
        blackhole / "llk_lib",
        blackhole / "llk_lib" / "hal",
        blackhole / "common" / "inc",
        blackhole / "common" / "inc" / "sfpu",
        llk_root / "common",
        metal_inc,
        metal_inc / "internal" / "tt-1xx" / "blackhole",
    ]
    return [
        str(llk_root / "tests" / "sfpi" / "compiler" / "bin" / "riscv-tt-elf-g++"),
        "-mcpu=tt-bh-tensix",
        "-O3",
        "-ffunction-sections",
        "-std=c++17",
        "-ftt-nttp",
        "-ftt-constinit",
        "-ftt-consteval",
        "-ftt-no-dyninit",
        "-ffast-math",
        "-fno-exceptions",
        "-fno-rtti",
        "-fno-use-cxa-atexit",
        "-mno-tt-fix-whbhebreak",
        "-DTENSIX_FIRMWARE",
        "-DENV_LLK_INFRA",
        "-DKERNEL_BUILD",
        "-DARCH_BLACKHOLE",
        "-DLLK_BOOT_MODE_BRISC",
        # Instruction words do not depend on the thread; the headers need one defined.
        "-DCOMPILE_FOR_TRISC=1",
        "-DLLK_TRISC_MATH",
        *(f"-I{path}" for path in includes),
    ]


def wrap_code(code):
    return f'{INCLUDES}\n#line 1 "{INPUT_FILE}"\n{code}\n'


def wrap_body(body, variables):
    """A replay lambda body inside one function; variables become its parameters."""
    if not braces_balanced(body):
        raise BuildError("The body has unbalanced braces, parentheses or brackets.")
    # One parameter per line, so a diagnostic at parameters:N names variable N.
    parameters = ",\n".join(
        f"const {v['type'].strip()} {v['name'].strip()}" for v in variables
    )
    return (
        f"{INCLUDES}{ALIASES}\n"
        f'extern "C" __attribute__((noinline, used)) void {BODY_FUNCTION}(\n'
        f'#line 1 "{PARAMETERS_FILE}"\n{parameters}\n)\n'
        "{\n"
        "    [&]() __attribute__((always_inline))\n"
        "    {\n"
        f'#line 1 "{INPUT_FILE}"\n{body}\n'
        "    }();\n"
        "}\n"
    )


def braces_balanced(code):
    """True when {}, () and [] nest correctly outside comments and literals."""
    code = re.sub(r"//[^\n]*|/\*.*?\*/", " ", code, flags=re.S)
    code = re.sub(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'', '""', code)
    closing = {")": "(", "]": "[", "}": "{"}
    stack = []
    for char in code:
        if char in "([{":
            stack.append(char)
        elif char in closing:
            if not stack or stack.pop() != closing[char]:
                return False
    return not stack


def compile_source(source, llk_root=LLK_ROOT):
    """(dump text, diagnostics, errors); raises BuildError on failure."""
    flags = compile_flags(llk_root)
    if not os.path.isfile(flags[0]):
        raise BuildError(
            f"SFPI compiler not found at {flags[0]}; install the toolchain in tests/sfpi."
        )
    with tempfile.TemporaryDirectory() as tmp:
        source_path = os.path.join(tmp, "input.cpp")
        dump = os.path.join(tmp, "input.dump")
        with open(source_path, "w", encoding="utf-8", errors="replace") as f:
            f.write(source)
        command = flags + [
            "-c",
            source_path,
            "-o",
            os.path.join(tmp, "input.o"),
            f"-fdump-tree-optimized-lineno-blocks={dump}",
        ]
        process = subprocess.Popen(
            command,
            cwd=tmp,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
        )
        try:
            _, stderr = process.communicate(timeout=COMPILE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise BuildError(
                f"Compilation timed out after {COMPILE_TIMEOUT_SECONDS} s."
            )
        diagnostics = "\n".join(
            line
            for line in stderr.replace(tmp + "/", "").splitlines()
            if "In lambda function" not in line
        ).strip()
        errors = [
            {
                "file": m.group(1),
                "line": int(m.group(2)),
                "column": int(m.group(3)),
                "message": m.group(5),
            }
            for m in map(DIAGNOSTIC.match, diagnostics.splitlines())
            if m and m.group(4) == "error"
        ]
        if process.returncode != 0:
            return None, diagnostics, errors
        if not os.path.isfile(dump):
            return "", diagnostics, errors
        with open(dump, encoding="utf-8", errors="replace") as f:
            return f.read(), diagnostics, errors
