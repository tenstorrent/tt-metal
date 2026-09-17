# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""lit configuration for the tt-llk host unit tests.

Run with:
    lit -v tools/tests/unit

The tests are host-side: the diagnostics suite is compile-only and the functional suite
builds and runs a small binary, so nothing here needs a device. clang is required rather
than merely preferred -- the tests use -verify and split-file, which are LLVM tools, and
the kernel toolchain is clang anyway.

Override the compiler with CXX, and the LLVM tool directory with LLVM_BIN:
    CXX=/path/to/clang++ LLVM_BIN=/usr/lib/llvm-20/bin lit -v tools/tests/unit
"""

import glob
import os
import shutil
import subprocess
import sys
import tempfile

import lit.formats

config.name = "tt-llk-unit"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = [".cpp"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(
    config.test_source_root, "..", "..", "build", "lit"
)

# Lets a DEFINE'd substitution reference other substitutions, so a test can build up
# %{check} from %{cflags} and %{verify} instead of repeating the whole command per case.
config.recursiveExpansionLimit = 10


def _require(path, what, hint):
    if path is None:
        lit_config.fatal(f"could not find {what}. {hint}")
    return path


# ---- compiler -------------------------------------------------------------------------
# Must be clang: -verify is a clang feature, and gcc has been observed to accept code in
# these headers that clang correctly rejects.
cxx = _require(
    shutil.which(os.environ.get("CXX", "clang++")),
    "a clang++ binary",
    "Install clang or set CXX to a clang++ binary.",
)

version = os.popen(f"{cxx} --version").read()
if "clang" not in version.lower():
    lit_config.fatal(
        f"{cxx} is not clang ({version.splitlines()[0] if version else 'unknown'}).\n"
        "These tests use -verify and require clang. Set CXX to a clang++ binary."
    )

# ---- LLVM tools -----------------------------------------------------------------------
# split-file carves each test file into one translation unit per case, and the RUN lines
# invoke it by bare name, so the LLVM bin directory has to be on PATH for the tests.
llvm_bin = os.environ.get("LLVM_BIN")
if llvm_bin is None:
    found = shutil.which("split-file")
    if found is None:
        # clang lives beside the LLVM tools in most distro layouts.
        candidate = os.path.dirname(os.path.realpath(cxx))
        if os.path.exists(os.path.join(candidate, "split-file")):
            found = os.path.join(candidate, "split-file")
    llvm_bin = os.path.dirname(found) if found else None

split_file = _require(
    shutil.which("split-file", path=llvm_bin) if llvm_bin else None,
    "split-file",
    "Set LLVM_BIN to an LLVM bin directory (e.g. /usr/lib/llvm-20/bin).",
)

# Make the rest of the LLVM tools reachable from RUN lines.
config.environment["PATH"] = os.pathsep.join([llvm_bin, os.environ.get("PATH", "")])

# ---- headers under test ---------------------------------------------------------------
# The tests include sanitizer/types.h straight out of the source tree. Resolving the
# include root here rather than with a %S-relative path from each RUN line keeps it correct
# for both suites: diagnostics/ and functional/ sit at different depths, and the
# diagnostics cases are compiled from %t after split-file, not from %S at all.
sanitizer_include = os.path.abspath(
    os.path.join(config.test_source_root, "..", "..", "include")
)

_require(
    (
        sanitizer_include
        if os.path.isdir(os.path.join(sanitizer_include, "sanitizer"))
        else None
    ),
    f"the sanitizer headers (looked for a sanitizer/ directory in {sanitizer_include})",
    "Run lit against the tools/tests/unit directory inside a tt-llk checkout.",
)

# ---- libfmt (host mocks) --------------------------------------------------------------
# sanitizer/deps/host.h formats reports with header-only libfmt. Probe the
# candidate include locations; when none works, tests marked REQUIRES: fmt are UNSUPPORTED
# rather than failing. Override with FMT_INCLUDE=/path/to/include.


def _find_fmt_flags():
    candidates = []
    if os.environ.get("FMT_INCLUDE"):
        candidates.append(["-I", os.environ["FMT_INCLUDE"]])
    candidates.append([])  # system include path
    # A tt-metal python_env carries fmt inside torch; lit usually runs from that venv.
    for path in sorted(
        glob.glob(
            os.path.join(
                sys.prefix, "lib", "python3*", "site-packages", "torch", "include"
            )
        )
    ):
        candidates.append(["-I", path])

    probe = (
        "#define FMT_HEADER_ONLY\n"
        "#include <fmt/format.h>\n"
        'int main() { fmt::print("{}", 1); }\n'
    )
    fd, probe_path = tempfile.mkstemp(suffix=".cpp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(probe)
        for flags in candidates:
            result = subprocess.run(
                [cxx, "-std=c++17", "-fsyntax-only"] + flags + [probe_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode == 0:
                return flags
    finally:
        os.unlink(probe_path)
    return None


_fmt_flags = _find_fmt_flags()
if _fmt_flags is None:
    config.substitutions.append(("%{fmt_flags}", ""))
    lit_config.note(
        "libfmt not found: REQUIRES: fmt tests are UNSUPPORTED "
        "(set FMT_INCLUDE to an include dir containing fmt/)"
    )
else:
    config.available_features.add("fmt")
    config.substitutions.append(("%{fmt_flags}", " ".join(_fmt_flags)))
    lit_config.note(f"libfmt: {' '.join(_fmt_flags) or 'system include path'}")

# ---- substitutions --------------------------------------------------------------------
# Each test file builds its own command line out of DEFINE'd substitutions on top of
# %clangxx, so the flags a case depends on -- -verify in particular -- stay visible in the
# file being read rather than hidden here.
#
# The brace form is deliberate for the include path: a bare %sanitizer_include would be
# eaten by lit's built-in %s substitution before it ever matched.

config.substitutions.append(("%clangxx", cxx))
config.substitutions.append(("%split-file", split_file))
config.substitutions.append(("%{sanitizer_include}", sanitizer_include))

lit_config.note(f"clang: {cxx}")
lit_config.note(f"LLVM tools: {llvm_bin}")
lit_config.note(f"sanitizer headers: {sanitizer_include}")
