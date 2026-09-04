#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Turn a bear-captured compile_commands.json from a real JIT run into a
clang-tidy-compatible one, and (optionally) run clang-tidy over it.

Background
----------
Device kernels are JIT-compiled at runtime by tt_metal/jit_build/build.cpp using
the SFPI cross-compiler (riscv-tt-elf-g++, a GCC). Running a workload under
`bear` (with TT_METAL_FORCE_JIT_COMPILE=1 so cache hits don't suppress compiler
processes) captures every real kernel compile: real compile-time args, real
generated headers (chlkc_*.cpp, chlkc_descriptors.h, kernel_includes.hpp,
defines_generated.h -- all durable in the tt-metal-cache dir), real defines.

clang-tidy needs *clang* to parse, so this script translates each captured
SFPI-GCC invocation into an equivalent clang invocation:

  * compiler -> clang++ with --target=riscv32-unknown-elf -march=... -mabi=...
    (mapped from -mcpu=tt-wh / tt-bh / tt-wh-tensix / ...)
  * SFPI-GCC-only flags dropped: -ftt-nttp -ftt-constinit -ftt-consteval
    -ftt-no-dyninit, -mno-tt-fix-whbhebreak, --param=min-pagesize=0,
    -fno-tree-loop-distribute-patterns, -flto=auto, dep-file flags (-MMD/-MF)
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

IMPORTANT: unlike scripts/build_kernel_compile_commands_json.py (the IDE
indexing flow), this script does NOT rewrite the TU to the user's kernel
source. The captured TUs are kept as-is (brisck.cc / ncrisck.cc / trisck.cc /
erisck.cc wrapping the generated per-kernel files), so all three TRISC roles
(UNPACK/MATH/PACK) stay distinct and the exact real preprocessing context is
preserved. Findings inside kernel sources surface via --header-filter, since
the kernel .cpp is #included by the generated wrapper.

Because many entries share the same TU path (e.g. every compute kernel's TU is
trisck.cc), running plain `clang-tidy -p <dir> <file>` would only analyze ONE
entry per file. The --run mode therefore invokes clang-tidy per-entry using the
`clang-tidy <file> -- <flags>` form, with cwd set to the entry's directory so
the -I. / -I.. generated-file includes resolve exactly like the real compile.

Typical use (see tech_reports/code-indexing/kernel-clang-tidy.md):

  export TT_METAL_FORCE_JIT_COMPILE=1
  export CCACHE_DISABLE=1   # if kernel ccache is enabled
  bear --output /tmp/raw_cc.json -- python3 /abs/path/to/some_test.py
  python3 scripts/build_kernel_clang_tidy_commands.py \
      --input /tmp/raw_cc.json --output-dir /tmp/kernel_tidy \
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
    "-flto=auto",
    "-flto",
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
    "-ftt-",       # SFPI-GCC extensions (-ftt-nttp, -ftt-constinit, ...)
    "-fdump-",     # GCC dump flags (TT_METAL_JIT_ANALYTICS / build-map modes)
)
# Flags that consume the NEXT argv element and are dropped with it.
DROP_WITH_ARG = {"-MF", "-o"}

# -mcpu=<tt cpu> -> (clang --target, -march, -mabi, sfpi libstdc++ multilib dir)
MCPU_MAP = {
    "tt-wh": ("riscv32-unknown-elf", "rv32im", "ilp32", "wh-ilp32"),
    "tt-wh-tensix": ("riscv32-unknown-elf", "rv32im", "ilp32", "wh-ilp32"),
    "tt-bh": ("riscv32-unknown-elf", "rv32im", "ilp32", "bh-ilp32"),
    "tt-bh-tensix": ("riscv32-unknown-elf", "rv32im", "ilp32", "bh-ilp32"),
    # Quasar mappings are untested; provided so entries aren't silently skipped.
    "tt-qsr32": ("riscv32-unknown-elf", "rv32im", "ilp32", "qsr32-ilp32"),
    "tt-qsr64": ("riscv64-unknown-elf", "rv64im", "lp64", "qsr64-lp64"),
}

# riscv32 GCC/newlib type-model overrides (int32_t is `long` there; sfpi.h
# static_asserts this -- see the "analysis tools sometimes are misconfigured"
# comment in sfpi.h).
INT32_OVERRIDES = [
    "-U__INT32_TYPE__", "-D__INT32_TYPE__=long int",
    "-U__UINT32_TYPE__", "-D__UINT32_TYPE__=long unsigned int",
    "-U__INT_LEAST32_TYPE__", "-D__INT_LEAST32_TYPE__=long int",
    "-U__UINT_LEAST32_TYPE__", "-D__UINT_LEAST32_TYPE__=long unsigned int",
]

# Diagnostics that are pure GCC/clang divergence noise for device code.
NOISE_SUPPRESSIONS = [
    "-Wno-unknown-attributes",  # rvtt_l1_ptr / rvtt_reg_ptr address-space attrs
    "-Wno-missing-template-arg-list-after-template-kw",  # error-by-default in clang>=20, GCC accepts
]

# Matches the per-kernel JIT cache dir layout:
#   <cache>/<key>/kernels/<kernel_name>/<hash>/<target>/
KERNEL_DIR_RE = re.compile(r"/kernels/(?P<kname>[^/]+)/(?P<khash>[^/]+)/(?P<target>[^/]+)/?$")


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


def sfpi_isystem_flags(compiler_root, multilib):
    """The SFPI toolchain's own newlib + libstdc++ header paths, for clang."""
    if compiler_root is None:
        return []
    cxx_root = compiler_root / "riscv-tt-elf" / "include" / "c++"
    versions = sorted(d.name for d in cxx_root.iterdir() if d.is_dir()) if cxx_root.is_dir() else []
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
            target_info = MCPU_MAP.get(a[len("-mcpu="):])
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
    return (d, f, )


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
    proc = subprocess.run(
        cmd, cwd=entry["directory"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return entry, proc.returncode, proc.stdout, proc.stderr


FINDING_RE = re.compile(r"(?:warning|error): .*\[([A-Za-z0-9.,\-]+)\]\s*$")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Raw compile_commands.json captured by bear")
    ap.add_argument("--output-dir", required=True, help="Where to write the translated compile_commands.json (and findings)")
    ap.add_argument("--clang", default="clang++", help="clang driver name to put in the translated commands")
    ap.add_argument("--dedupe", choices=["kernel-role", "none"], default="kernel-role",
                    help="kernel-role (default): one entry per (kernel, RISC target); none: keep every captured config")
    ap.add_argument("--limit", type=int, default=0, help="Cap the number of entries (0 = no cap)")
    ap.add_argument("--run", action="store_true", help="Also run clang-tidy over the translated entries")
    ap.add_argument("--clang-tidy", default="", help="clang-tidy binary (default: autodetect newest)")
    ap.add_argument("--config-file", default="", help=".clang-tidy config for --run")
    ap.add_argument("--header-filter",
                    default=r".*/(kernels|kernels_ng|kernels_dfb|test_kernels)/.*",
                    help="--header-filter passed to clang-tidy (findings in #included kernel sources)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--fail-on-findings", action="store_true",
                    help="Exit nonzero if any finding is emitted (default: report only)")
    args = ap.parse_args()

    with open(args.input) as f:
        raw = json.load(f)

    seen = set()
    entries = []
    skipped_unknown_cpu = 0
    for e in raw:
        argv = entry_argv(e)
        if not is_sfpi_compile(argv):
            continue
        key = dedupe_key(e, args.dedupe)
        if key in seen:
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
    if skipped_unknown_cpu:
        print(f"[kernel-clang-tidy] skipped {skipped_unknown_cpu} entries with unrecognized -mcpu", file=sys.stderr)
    if not entries:
        print("[kernel-clang-tidy] nothing captured. Did the run compile anything? "
              "(TT_METAL_FORCE_JIT_COMPILE=1 and CCACHE_DISABLE=1 must be set, "
              "and the command must be wrapped in `bear --`)", file=sys.stderr)
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
        futures = [pool.submit(run_tidy_entry, tidy_bin, args.config_file, args.header_filter, e) for e in entries]
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

    print(f"[kernel-clang-tidy] {total_findings} findings across {len(entries)} entries "
          f"({failed_entries} entries had parse/config errors) -> {findings_path}")
    for check, count in sorted(check_counts.items(), key=lambda kv: -kv[1])[:25]:
        print(f"  {count:6d}  {check}")

    if args.fail_on_findings and total_findings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
