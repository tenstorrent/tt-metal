#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Turn the kernel compile commands captured from a real JIT run into a
clang-tidy-compatible compile_commands.json, and (optionally) run clang-tidy.

Two capture sources are supported:

  --input-log FILE   (recommended) A run log produced with
                     TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 and
                     TT_LOGGER_LEVEL=info: jit_build/build.cpp logs the exact
                     compile argv of every kernel ("g++ compile cmd: ...").
                     No interception tooling needed; works in any container.
  --input FILE       A bear-captured compile_commands.json. Local use only:
                     bear 3.0.x's gRPC-over-localhost intercept channel fails
                     in containers that set http_proxy, CI's included.

Device kernels are JIT-compiled at runtime by tt_metal/jit_build/build.cpp using
the SFPI cross-compiler (riscv-tt-elf-g++, a GCC), so a captured run yields real
compile-time args, real defines and the real generated headers (chlkc_*.cpp,
chlkc_descriptors.h, kernel_includes.hpp, defines_generated.h, all durable in
the tt-metal-cache dir). clang-tidy needs *clang* to parse, so each captured
SFPI-GCC invocation is translated into an equivalent clang one:

  * compiler -> clang++ with --target=riscv32-unknown-elf -march=... -mabi=...
    (mapped from -mcpu=tt-wh / tt-bh / tt-wh-tensix / ...)
  * SFPI-GCC-only flags dropped: -ftt-nttp -ftt-constinit -ftt-consteval
    -ftt-no-dyninit, -mno-tt-fix-whbhebreak, --param=min-pagesize=0,
    -fno-tree-loop-distribute-patterns, every -flto* spelling, dep-file flags
    (-MMD/-MF)
  * -std=c++17 -> -std=c++20 (the dropped -ftt-nttp/-ftt-constinit/
    -ftt-consteval backport C++20 features into SFPI's C++17 mode; tt-llk
    headers rely on them)
  * libc/libc++: clang has no runtime headers for riscv32-unknown-elf, so the
    SFPI toolchain's OWN newlib + libstdc++ headers are wired in via
    -nostdinc++/-nostdlibinc + -isystem (derived from the captured compiler
    path), keeping the header set identical to the real device build
  * int32_t: riscv32 GCC/newlib define int32_t as `long`; clang's default is
    `int`. sfpi.h static_asserts int32_t == long (see the comment there about
    misconfigured analysis tools), so __INT32_TYPE__ et al are overridden.

No mock/stub SFPI headers are needed: SFPI >= 7.x ships an analysis-tool
fallback (include/tensix_builtins.h detects a non-SFPI compiler, defines the
__xtt_vector types and pulls machine-generated __builtin_rvtt_* declarations
from tensix_builtins.def). See tt_metal/jit_build/kernel_clang_tidy/README.md.

Unlike scripts/build_kernel_compile_commands_json.py (the IDE indexing flow),
this script does NOT rewrite the TU to the user's kernel source. The captured
TUs are kept as-is (brisck.cc / ncrisck.cc / trisck.cc / erisck.cc wrapping the
generated per-kernel files), so all three TRISC roles stay distinct and the real
preprocessing context is preserved. Findings inside kernel sources surface via
--header-filter, since the kernel .cpp is #included by the generated wrapper.

Many entries share one TU path (every compute kernel's TU is trisck.cc), so
plain `clang-tidy -p <dir> <file>` would analyze only one entry per file. --run
therefore uses the `clang-tidy <file> -- <flags>` form per entry, with cwd set
to the entry's directory so the -I. / -I.. generated-file includes resolve
exactly as they did in the real compile.

Typical use (see tech_reports/code-indexing/kernel-clang-tidy.md):

  export TT_METAL_FORCE_JIT_COMPILE=1
  export CCACHE_DISABLE=1   # if kernel ccache is enabled
  export TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1
  export TT_LOGGER_LEVEL=info   # the compile-cmd lines are logged at info level
  python3 /abs/path/to/some_test.py 2>&1 | tee /tmp/run.log
  python3 scripts/build_kernel_clang_tidy_commands.py \
      --input-log /tmp/run.log --output-dir /tmp/kernel_tidy \
      --run --config-file tt_metal/jit_build/kernel_clang_tidy/.clang-tidy
"""

import argparse
import concurrent.futures
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

# The SFPI cross-compiler, optionally prefixed by ccache. Both the current
# (riscv-tt-elf-g++) and older (riscv32-tt-elf-g++) names are matched.
SFPI_GXX_RE = re.compile(r"riscv(32)?-tt-elf-g\+\+$")

# Flags dropped verbatim (SFPI-GCC-only or irrelevant/harmful for parsing).
DROP_EXACT = {
    "--param=min-pagesize=0",
    "-fno-tree-loop-distribute-patterns",
    "-mno-tt-fix-whbhebreak",
    "-MMD",
    "-MD",
    "-MP",
    "-Werror",  # lint pass: keep warnings as warnings
    "-save-temps=obj",
}
# Flags dropped by prefix.
DROP_PREFIX = (
    "-ftt-",  # SFPI-GCC extensions (-ftt-nttp, -ftt-constinit, ...)
    "-fdump-",  # GCC dump flags (TT_METAL_JIT_ANALYTICS / build-map modes)
    # Every LTO spelling, matched by prefix so the build's choice can change without
    # breaking lint again. clang's -flto= accepts only thin|full, so GCC's job-count
    # and partitioning forms are hard errors ("unsupported argument '1'", "unknown
    # argument: '-flto-partition=one'") rather than something clang-tidy can ignore.
    "-flto",
)
# Flags that consume the NEXT argv element and are dropped with it.
DROP_WITH_ARG = {"-MF", "-o"}

# -mcpu=<tt cpu> -> (clang --target, -march, -mabi, sfpi libstdc++ multilib dir)
MCPU_MAP = {
    "tt-wh": ("riscv32-unknown-elf", "rv32im", "ilp32", "wh-ilp32"),
    "tt-wh-tensix": ("riscv32-unknown-elf", "rv32im", "ilp32", "wh-ilp32"),
    "tt-bh": ("riscv32-unknown-elf", "rv32im", "ilp32", "bh-ilp32"),
    "tt-bh-tensix": ("riscv32-unknown-elf", "rv32im", "ilp32", "bh-ilp32"),
    # Quasar, untested. Names as emitted by qa_hal.cpp, where DM is rv64.
    "tt-qsr32-tensix": ("riscv32-unknown-elf", "rv32im", "ilp32", "qsr32-ilp32"),
    "tt-qsr64-rocc": ("riscv64-unknown-elf", "rv64im", "lp64", "qsr64-lp64"),
}

# riscv32 GCC/newlib type-model overrides (int32_t is `long` there; sfpi.h
# static_asserts this -- see the "analysis tools sometimes are misconfigured"
# comment in sfpi.h).
INT32_OVERRIDES = [
    "-U__INT32_TYPE__",
    "-D__INT32_TYPE__=long int",
    "-U__UINT32_TYPE__",
    "-D__UINT32_TYPE__=long unsigned int",
    "-U__INT_LEAST32_TYPE__",
    "-D__INT_LEAST32_TYPE__=long int",
    "-U__UINT_LEAST32_TYPE__",
    "-D__UINT_LEAST32_TYPE__=long unsigned int",
]

# Diagnostics that are pure GCC/clang divergence noise for device code.
NOISE_SUPPRESSIONS = [
    "-Wno-unknown-attributes",  # rvtt_l1_ptr / rvtt_reg_ptr address-space attrs
    "-Wno-missing-template-arg-list-after-template-kw",  # error-by-default in clang>=20, GCC accepts
]

# Matches the per-kernel JIT cache dir layout:
#   <cache>/<key>/kernels/<kernel_name>/<hash>/<target>/
KERNEL_DIR_RE = re.compile(r"/kernels/(?P<kname>[^/]+)/(?P<khash>[^/]+)/(?P<target>[^/]+)/?$")

# build.cpp logs two line kinds under TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1:
#   "    g++ compile cmd: <space-joined argv>"
#   "    g++ link cmd: cd <dir> && <shell cmd>"
# Only compiles are wanted; a link fed to clang-tidy errors out or, worse,
# quietly produces a bogus result. Matching the literal "compile cmd" label
# discards links here, and two downstream layers enforce the same invariant:
# entries_from_log requires a .cc/.cpp token, is_sfpi_compile requires -c and
# rejects -E. All three are exercised by --self-test.
#
# The logger appends a "(build.cpp:NNN)" suffix, wrapped in SGR escapes when
# colour is on, which would push it past the "$" anchor and leave the escape run
# spliced into the argv as a bogus input file. Strip SGR before matching.
#
# argv is joined with single spaces and never quoted, so split() is used rather
# than shlex, which would eat the literal quotes some defines carry. That loses
# the boundary for defines whose value holds a space; rejoin_split_defines puts
# those back.
ANSI_SGR_RE = re.compile(r"\x1b\[[0-9;]*m")
LOG_CMD_RE = re.compile(r"g\+\+ compile cmd: (?P<cmd>.+?)(?:\s*\(build\.cpp:\d+\))?\s*$")


def rejoin_split_defines(argv):
    """Repair -D values that the log's unquoted join split on a space.

    ttnn emits defines of the form -DFILL_WITH_VALUE=fill_with_val<1024, int32_t>,
    which arrive as two tokens and reach clang as a truncated template-id plus a
    stray input file. Rejoin while the angle brackets are unbalanced, and leave
    the tokens alone if the run does not close before the next option, so a
    define holding a bare '<' as less-than cannot swallow the rest of the
    command.
    """
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("-D") and a.count("<") > a.count(">"):
            merged, j = a, i + 1
            while j < len(argv) and not argv[j].startswith("-"):
                merged += " " + argv[j]
                j += 1
                if merged.count("<") == merged.count(">"):
                    break
            if merged.count("<") == merged.count(">"):
                out.append(merged)
                i = j
                continue
        out.append(a)
        i += 1
    return out


def entries_from_log(path):
    """Reconstruct compile_commands-style entries from a TT run log."""
    entries = []
    seen_lines = set()
    with open(path, errors="replace") as f:
        for line in f:
            m = LOG_CMD_RE.search(ANSI_SGR_RE.sub("", line))
            if not m:
                continue
            cmd = m.group("cmd")
            if cmd in seen_lines:
                continue
            seen_lines.add(cmd)
            argv = rejoin_split_defines(cmd.split())
            # directory: the JIT build runs the compiler with cwd = the kernel's
            # out_dir, which is also the (absolute) dirname of the -o object.
            directory = None
            src = None
            for i, a in enumerate(argv):
                if a == "-o" and i + 1 < len(argv):
                    directory = os.path.dirname(argv[i + 1])
                if a.endswith((".cc", ".cpp")):
                    src = a
            if not directory or not src:
                continue
            entries.append({"directory": directory, "file": src, "arguments": argv})
    return entries


def entry_argv(entry):
    if "arguments" in entry and entry["arguments"]:
        return list(entry["arguments"])
    if "command" in entry:
        return shlex.split(entry["command"])
    return []


def is_sfpi_compile(argv):
    """True for SFPI g++ *compile* invocations (not links/preprocesses)."""
    if not argv:
        return False
    i = 1 if os.path.basename(argv[0]) == "ccache" and len(argv) > 1 else 0
    if not SFPI_GXX_RE.search(os.path.basename(argv[i])):
        return False
    return "-c" in argv and "-E" not in argv


def find_sfpi_compiler_root(compiler_path):
    """.../sfpi/compiler/bin/riscv-tt-elf-g++ -> .../sfpi/compiler (or None)."""
    p = Path(compiler_path)
    if len(p.parts) < 3:
        return None
    root = p.parent.parent  # strip bin/<exe>
    return root if (root / "riscv-tt-elf" / "include").is_dir() else None


def is_vendor_include(path):
    """True for the pinned SFPI release, which is not ours to fix.

    The device build passes `-I /opt/tenstorrent/sfpi/include` while every other
    SFPI path already arrives as `-isystem`, so clang-tidy treats sfpi.h as
    first-party and reports on it. Demoting it to a system include is the same
    thing CMake's SYSTEM keyword does for host dependencies.
    """
    return "/sfpi/" in path.replace(os.sep, "/")


def gcc_version_key(name):
    """Order 10.2.0 after 9.3.0, which a lexicographic sort gets backwards."""
    return tuple(int(p) for p in name.split(".") if p.isdigit())


def sfpi_isystem_flags(compiler_root, multilib):
    """The SFPI toolchain's own newlib + libstdc++ header paths, for clang."""
    if compiler_root is None:
        return []
    cxx_root = compiler_root / "riscv-tt-elf" / "include" / "c++"
    versions = (
        sorted((d.name for d in cxx_root.iterdir() if d.is_dir()), key=gcc_version_key) if cxx_root.is_dir() else []
    )
    if not versions:
        return []
    v = versions[-1]
    gcc_inc = compiler_root / "lib" / "gcc" / "riscv-tt-elf" / v / "include"
    flags = ["-nostdinc++", "-nostdlibinc"]
    for d in (
        cxx_root / v,
        cxx_root / v / "riscv-tt-elf" / multilib,
        cxx_root / v / "backward",
        gcc_inc,
        compiler_root / "riscv-tt-elf" / "include",
    ):
        flags += ["-isystem", str(d)]
    return flags


def transform(argv, clang):
    """SFPI-GCC argv -> clang argv. Returns None if the -mcpu is unknown."""
    if os.path.basename(argv[0]) == "ccache":
        argv = argv[1:]
    compiler_root = find_sfpi_compiler_root(argv[0])

    out = [clang]
    target_info = None
    saw_ftt = False
    i = 1
    while i < len(argv):
        a = argv[i]
        if a in DROP_WITH_ARG:
            i += 2
            continue
        if a in DROP_EXACT or a.startswith(DROP_PREFIX):
            saw_ftt = saw_ftt or a.startswith("-ftt-")
            i += 1
            continue
        if a.startswith("-mcpu="):
            target_info = MCPU_MAP.get(a[len("-mcpu=") :])
            i += 1
            continue
        if a == "-I" and i + 1 < len(argv) and is_vendor_include(argv[i + 1]):
            out += ["-isystem", argv[i + 1]]
            i += 2
            continue
        if len(a) > 2 and a.startswith("-I") and is_vendor_include(a[2:]):
            out += ["-isystem", a[2:]]
            i += 1
            continue
        out.append(a)
        i += 1

    if target_info is None:
        return None
    triple, march, mabi, multilib = target_info
    out += [f"--target={triple}", f"-march={march}", f"-mabi={mabi}"]
    if saw_ftt:
        # -ftt-nttp/-ftt-constinit/-ftt-consteval backport C++20 features that
        # tt-llk headers use; clang needs real C++20 to accept them.
        out = [("-std=c++20" if a == "-std=c++17" else a) for a in out]
    out += sfpi_isystem_flags(compiler_root, multilib)
    if mabi == "ilp32":
        # `long` is 64-bit under LP64, so forcing int32_t to it would give the
        # rv64 target (Quasar DM) a 64-bit int32_t.
        out += INT32_OVERRIDES
    out += NOISE_SUPPRESSIONS
    return out


def dedupe_key(entry, mode):
    d = entry.get("directory", "")
    f = entry.get("file", "")
    if mode == "kernel-role":
        m = KERNEL_DIR_RE.search(d)
        if m:
            # One entry per (kernel source name, RISC target). Multiple CTA
            # configs of the same kernel collapse to the first one seen --
            # acceptable for a lint pass, revisit if per-config coverage is
            # ever needed (drop --dedupe to keep everything).
            return (m.group("kname"), m.group("target"))
    return (
        d,
        f,
    )


def find_clang_tidy(explicit):
    if explicit:
        return explicit
    for name in ("clang-tidy-21", "clang-tidy-20", "clang-tidy-19", "clang-tidy-18", "clang-tidy-17", "clang-tidy"):
        if shutil.which(name):
            return name
    return None


def run_tidy_entry(tidy_bin, cfg, header_filter, entry):
    argv = entry["arguments"]
    src = entry["file"]
    # `clang-tidy <file> -- <flags>`: bypasses compile_commands.json lookup so
    # every entry is analyzed even when many entries share one TU path.
    flags = [a for a in argv[1:] if a != src and a != "-c"]
    cmd = [tidy_bin, src, "--quiet"]
    if cfg:
        cmd.append(f"--config-file={cfg}")
    if header_filter:
        cmd.append(f"--header-filter={header_filter}")
    cmd += ["--"] + flags
    proc = subprocess.run(cmd, cwd=entry["directory"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return entry, proc.returncode, proc.stdout, proc.stderr


FINDING_RE = re.compile(r"(?:warning|error): .*\[([A-Za-z0-9.,\-]+)\]\s*$")


def self_test():
    """Fixture-based assertions for the parse/filter/transform invariants.

    Run with --self-test. Covers, explicitly rather than implicitly:
      * link-command log lines are discarded (never reach clang-tidy),
      * duplicate compile lines dedupe,
      * preprocess (-E) and link argvs are rejected by is_sfpi_compile,
      * the GCC->clang flag translation drops/maps what it must.
    """
    import tempfile

    gxx = "/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-g++"
    out_dir = "/home/u/.cache/tt-metal-cache/k/1/kernels/reduce_h/42/trisc1"
    compile_argv = [
        gxx,
        "-O3",
        "-std=c++17",
        "-ftt-nttp",
        "-flto=1",
        "-flto-partition=one",
        "-MMD",
        "-mcpu=tt-wh-tensix",
        "-I.",
        "-I..",
        "-DARCH_WORMHOLE",
        '-DFULL_KERNEL_NAME="reduce_h/42"',
        "-DKERNEL_COMPILE_TIME_ARGS=1,2,3",
        "-DFILL_WITH_VALUE=fill_with_val<1024, int32_t>",
        "-c",
        "-o",
        f"{out_dir}/._7_0_trisck.o",
        "-MF",
        f"{out_dir}/._7_0_trisck.d",
        "/repo/tt_metal/hw/firmware/src/tt-1xx/trisck.cc",
    ]
    compile_line = (
        "2026-01-01 00:00:01.000 | info     |    BuildKernels |     g++ compile cmd: "
        + " ".join(compile_argv)
        + " (build.cpp:686)\n"
    )
    link_line = (
        "2026-01-01 00:00:02.000 | info     |    BuildKernels |     g++ link cmd: "
        f"cd {out_dir}/ && {gxx} -O3 -Wl,--just-symbols=/x/trisc1_weakened.elf -mcpu=tt-wh "
        f"-flto=1 -flto-partition=one -T/x/kernel_trisc1.ld -Wl,--emit-relocs ._7_0_trisck.o /x/substitutes.o "
        f"-o {out_dir}/trisc1.elf (build.cpp:777)\n"
    )
    # Same line as tt-logger emits it with colour on: SGR escapes wrap the
    # source-location suffix. Must strip to exactly the same argv, hence dedupe
    # to a single entry rather than adding a malformed second one.
    colored_line = (
        "2026-01-01 00:00:03.000 | info     |    BuildKernels |     g++ compile cmd: "
        + " ".join(compile_argv)
        + " \x1b[90m(build.cpp:686)\x1b[0m\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
        f.write("2026-01-01 00:00:00.000 | info     | Metal | unrelated (foo.cpp:1)\n")
        f.write(compile_line)
        f.write(compile_line)  # duplicate (forced recompile) -> must dedupe
        f.write(colored_line)  # colourized suffix -> must strip to the same argv
        f.write(link_line)  # link -> must be discarded
        log_path = f.name

    entries = entries_from_log(log_path)
    os.unlink(log_path)
    assert len(entries) == 1, f"expected 1 entry (compile only, deduped), got {len(entries)}"
    assert entries[0]["file"].endswith("trisck.cc")
    assert entries[0]["directory"] == out_dir
    argv_text = " ".join(entries[0]["arguments"])
    assert "\x1b" not in argv_text, "SGR escape leaked into the captured argv"
    assert "build.cpp" not in argv_text, "source-location suffix leaked into the captured argv"
    # A define whose value holds a space must survive the log's unquoted join as
    # one element, or the macro reaches clang as a truncated template-id.
    assert (
        "-DFILL_WITH_VALUE=fill_with_val<1024, int32_t>" in entries[0]["arguments"]
    ), f"split define was not rejoined: {entries[0]['arguments']}"
    # A bare '<' must not let the rejoin swallow the rest of the command.
    kept = rejoin_split_defines(["-DCOND=a<b", "-DOTHER=1", "x.cc"])
    assert kept == ["-DCOND=a<b", "-DOTHER=1", "x.cc"], f"unbalanced define was over-merged: {kept}"

    # The link argv must also be rejected by the secondary filter (bear-mode path).
    link_argv = link_line.split("g++ link cmd: ", 1)[1].split()
    assert not is_sfpi_compile(link_argv), "link argv must not classify as a compile"
    # ... and so must a preprocess invocation.
    pre_argv = [gxx, "-E", "-c", "x.cpp"]
    assert not is_sfpi_compile(pre_argv), "-E argv must not classify as a compile"
    assert is_sfpi_compile(entries[0]["arguments"]), "compile argv must classify as a compile"

    out = transform(entries[0]["arguments"], "clang++")
    assert "--target=riscv32-unknown-elf" in out and "-march=rv32im" in out
    assert "-std=c++20" in out and "-std=c++17" not in out
    for banned in ("-ftt-nttp", "-flto=1", "-flto-partition=one", "-MMD", "-MF", "-o", "-mcpu=tt-wh-tensix"):
        assert banned not in out, f"{banned} must be dropped"

    # Every -flto spelling must be dropped, not just the one the build happens to pass
    # today: clang errors out on GCC's job-count and partitioning forms, and logs
    # captured before the build was pinned still carry -flto=auto.
    for spelling in ("-flto", "-flto=auto", "-flto=1", "-flto-partition=one"):
        got = transform([gxx, "-c", "-mcpu=tt-wh-tensix", spelling, "x.cc"], "clang++")
        assert got is not None, f"transform rejected {spelling}"
        assert spelling not in got, f"{spelling} must be dropped"
    assert '-DFULL_KERNEL_NAME="reduce_h/42"' in out, "defines must pass through verbatim"

    # SFPI headers must reach clang as system includes, in either -I spelling,
    # so clang-tidy does not analyze what we cannot fix. Project -I is untouched.
    for spelling in (["-I", "/opt/tenstorrent/sfpi/include"], ["-I/opt/tenstorrent/sfpi/include"]):
        got = transform([gxx, "-c", "-mcpu=tt-wh-tensix", *spelling, "-I/work/tt_metal", "x.cc"], "clang++")
        assert got is not None, f"transform rejected {spelling}"
        assert (
            "-isystem" in got and got[got.index("-isystem") + 1] == "/opt/tenstorrent/sfpi/include"
        ), f"SFPI include not demoted to -isystem for {spelling}: {got}"
        assert "-I/opt/tenstorrent/sfpi/include" not in got, f"SFPI kept as -I for {spelling}"
        assert "-I/work/tt_metal" in got, "project includes must stay first-party"

    # Every -mcpu the HAL emits must map, or those entries vanish from the
    # report with only a skip count to show for it.
    for cpu in ("tt-wh", "tt-wh-tensix", "tt-bh", "tt-bh-tensix", "tt-qsr32-tensix", "tt-qsr64-rocc"):
        assert cpu in MCPU_MAP, f"{cpu} is emitted by the HAL but unmapped"
    # int32_t is `long` only on the ilp32 targets.
    rv64 = transform([gxx, "-c", "-mcpu=tt-qsr64-rocc", "x.cc"], "clang++")
    assert "--target=riscv64-unknown-elf" in rv64
    assert "-D__INT32_TYPE__=long int" not in rv64, "32-bit type model must not reach rv64"
    rv32 = transform([gxx, "-c", "-mcpu=tt-wh-tensix", "x.cc"], "clang++")
    assert "-D__INT32_TYPE__=long int" in rv32, "ilp32 needs the newlib type model"

    assert gcc_version_key("10.2.0") > gcc_version_key("9.3.0"), "GCC versions must sort numerically"

    print("[kernel-clang-tidy] self-test PASSED")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src_group = ap.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--input", help="Raw compile_commands.json captured by bear")
    src_group.add_argument(
        "--input-log",
        help="Run log captured with TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 and TT_LOGGER_LEVEL=info",
    )
    src_group.add_argument("--self-test", action="store_true", help="Run built-in fixture assertions and exit")
    ap.add_argument(
        "--output-dir",
        help="Where to write the translated compile_commands.json (and findings); required except with --self-test",
    )
    ap.add_argument("--clang", default="clang++", help="clang driver name to put in the translated commands")
    ap.add_argument(
        "--dedupe",
        choices=["kernel-role", "none"],
        default="kernel-role",
        help="kernel-role (default): one entry per (kernel, RISC target); none: keep every captured config",
    )
    ap.add_argument("--limit", type=int, default=0, help="Cap the number of entries (0 = no cap)")
    ap.add_argument("--run", action="store_true", help="Also run clang-tidy over the translated entries")
    ap.add_argument("--clang-tidy", default="", help="clang-tidy binary (default: autodetect newest)")
    ap.add_argument("--config-file", default="", help=".clang-tidy config for --run")
    ap.add_argument(
        "--header-filter",
        default=r".*/(kernels|kernels_ng|kernels_dfb|test_kernels)/.*|.*/tt_metal/(tt-llk|hw|fabric)/.*",
        help="--header-filter passed to clang-tidy (findings in #included kernel sources)",
    )
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument(
        "--fail-on-findings", action="store_true", help="Exit nonzero if any finding is emitted (default: report only)"
    )
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.output_dir:
        ap.error("--output-dir is required (except with --self-test)")

    if args.input_log:
        raw = entries_from_log(args.input_log)
    else:
        with open(args.input) as f:
            raw = json.load(f)

    seen = set()
    entries = []
    skipped_unknown_cpu = 0
    collapsed = 0
    for e in raw:
        argv = entry_argv(e)
        if not is_sfpi_compile(argv):
            continue
        key = dedupe_key(e, args.dedupe)
        if key in seen:
            collapsed += 1
            continue
        new_argv = transform(argv, args.clang)
        if new_argv is None:
            skipped_unknown_cpu += 1
            continue
        seen.add(key)
        entries.append({"directory": e["directory"], "file": e["file"], "arguments": new_argv})
        if args.limit and len(entries) >= args.limit:
            break

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_db = outdir / "compile_commands.json"
    with open(out_db, "w") as f:
        json.dump(entries, f, indent=1)
    print(f"[kernel-clang-tidy] {len(raw)} captured commands -> {len(entries)} kernel compile entries -> {out_db}")
    if collapsed:
        # The coverage cost of --dedupe belongs in the CI log, not just here.
        print(f"[kernel-clang-tidy] --dedupe {args.dedupe} collapsed {collapsed} further compile-time-arg configs")
    if skipped_unknown_cpu:
        print(f"[kernel-clang-tidy] skipped {skipped_unknown_cpu} entries with unrecognized -mcpu", file=sys.stderr)
    if not entries:
        print(
            "[kernel-clang-tidy] nothing captured. Did the run compile anything? "
            "(TT_METAL_FORCE_JIT_COMPILE=1 and CCACHE_DISABLE=1 must be set; for --input-log "
            "the run also needs TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 and TT_LOGGER_LEVEL=info; "
            "for --input the command must be wrapped in `bear --`)",
            file=sys.stderr,
        )
        return 0

    if not args.run:
        return 0

    tidy_bin = find_clang_tidy(args.clang_tidy)
    if tidy_bin is None:
        print("[kernel-clang-tidy] no clang-tidy binary found", file=sys.stderr)
        return 2

    findings_path = outdir / "findings.txt"
    check_counts = {}
    total_findings = 0
    failed_entries = 0
    with open(findings_path, "w") as findings, concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        # Absolute: each process runs in the kernel cache dir, where a relative
        # config path does not resolve.
        cfg = os.path.abspath(args.config_file) if args.config_file else ""
        futures = [pool.submit(run_tidy_entry, tidy_bin, cfg, args.header_filter, e) for e in entries]
        for fut in concurrent.futures.as_completed(futures):
            entry, rc, out, err = fut.result()
            n_before = total_findings
            for line in out.splitlines():
                m = FINDING_RE.search(line)
                if m:
                    total_findings += 1
                    for check in m.group(1).split(","):
                        check_counts[check] = check_counts.get(check, 0) + 1
            if rc != 0:
                failed_entries += 1
            if out.strip() or rc != 0:
                findings.write(f"==== {entry['file']} (dir: {entry['directory']}, rc: {rc}) ====\n")
                findings.write(out)
                if rc != 0 and err.strip():
                    # Keep only the tail of stderr; clang-tidy repeats the full
                    # error list there.
                    findings.write("\n[stderr tail]\n" + "\n".join(err.splitlines()[-15:]) + "\n")
                findings.write("\n")

    print(
        f"[kernel-clang-tidy] {total_findings} findings across {len(entries)} entries "
        f"({failed_entries} entries had parse/config errors) -> {findings_path}"
    )
    for check, count in sorted(check_counts.items(), key=lambda kv: -kv[1])[:25]:
        print(f"  {count:6d}  {check}")

    if args.fail_on_findings and total_findings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
