# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""LIT configuration for tt-llk host-side unit tests.

Run with:
    python3 -m lit -sv tools/tests/unit

The scaffold supports compile-only diagnostics and host-side functional tests.
It requires Clang because diagnostics can use ``-verify`` and ``split-file``.

Override the compiler with CXX and the LLVM tool directory with LLVM_BIN:
    CXX=/path/to/clang++ LLVM_BIN=/usr/lib/llvm-20/bin \
        python3 -m lit -sv tools/tests/unit
"""

import os
import shutil
import subprocess

import lit.formats

config.name = "tt-llk-unit"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = [".cpp"]

unit_root = os.path.dirname(__file__)
config.test_source_root = os.path.join(unit_root, "lit")
config.test_exec_root = os.path.abspath(
    os.path.join(unit_root, "..", "..", "build", "lit")
)

# Allow DEFINE substitutions in test files to build on substitutions below.
config.recursiveExpansionLimit = 10


def _require(path, what, hint):
    if path is None:
        lit_config.fatal(f"could not find {what}. {hint}")
    return path


# Clang is required for -verify diagnostics and matches the kernel toolchain.
cxx = _require(
    shutil.which(os.environ.get("CXX", "clang++")),
    "a clang++ binary",
    "Install clang or set CXX to a clang++ binary.",
)

version_result = subprocess.run(
    [cxx, "--version"],
    check=False,
    capture_output=True,
    text=True,
)
version = version_result.stdout or version_result.stderr
if "clang" not in version.lower():
    first_line = version.splitlines()[0] if version else "unknown"
    lit_config.fatal(
        f"{cxx} is not clang ({first_line}).\n"
        "These tests use -verify and require clang. Set CXX to a clang++ binary."
    )

# split-file lets one source contain several independent diagnostic cases.
llvm_bin = os.environ.get("LLVM_BIN")
if llvm_bin is None:
    found = shutil.which("split-file")
    if found is None:
        candidate = os.path.dirname(os.path.realpath(cxx))
        if os.path.exists(os.path.join(candidate, "split-file")):
            found = os.path.join(candidate, "split-file")
    llvm_bin = os.path.dirname(found) if found else None

split_file = _require(
    shutil.which("split-file", path=llvm_bin) if llvm_bin else None,
    "split-file",
    "Set LLVM_BIN to an LLVM bin directory (for example, /usr/lib/llvm-20/bin).",
)

config.environment["PATH"] = os.pathsep.join([llvm_bin, os.environ.get("PATH", "")])

# Stable source roots keep RUN lines independent of a test's directory depth.
tt_llk_root = os.path.abspath(os.path.join(unit_root, "..", "..", ".."))
blackhole_root = os.path.join(tt_llk_root, "tt_llk_blackhole")

_require(
    tt_llk_root if os.path.isdir(tt_llk_root) else None,
    f"the tt-llk source root ({tt_llk_root})",
    "Run LIT against tools/tests/unit inside a tt-llk checkout.",
)
_require(
    blackhole_root if os.path.isdir(blackhole_root) else None,
    f"the Blackhole source root ({blackhole_root})",
    "Run LIT against tools/tests/unit inside a tt-llk checkout.",
)

config.substitutions.append(("%clangxx", cxx))
config.substitutions.append(("%split-file", split_file))
config.substitutions.append(("%{tt_llk_root}", tt_llk_root))
config.substitutions.append(("%{blackhole_root}", blackhole_root))
config.substitutions.append(
    (
        "%{blackhole_common_include}",
        os.path.join(blackhole_root, "common", "inc"),
    )
)
config.substitutions.append(
    (
        "%{blackhole_llk_include}",
        os.path.join(blackhole_root, "llk_lib"),
    )
)

lit_config.note(f"clang: {cxx}")
lit_config.note(f"LLVM tools: {llvm_bin}")
lit_config.note(f"tt-llk root: {tt_llk_root}")
