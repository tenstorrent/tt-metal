# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
kernel_tier/capture.py — turn a tt-metal JIT build log into a KERNEL fact base.

This is the on-request kernel-tier capture. tt-metal, run with
`TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1`, logs the full `g++ compile cmd: <cmd>`
for every JIT-compiled kernel. We scrape those, translate each RISC-V-GCC command
to a clang invocation `llk_extract` can parse (drop the sfpi-gcc-only flags, add
clang's --target + the sfpi -isystem paths + the SFPU shim, keep the kernel's own
-I/-D), run the extractor per kernel (from the kernel's build dir so its relative
includes + generated kernel_includes.hpp resolve), and merge into ONE fact base
the committed cb-sync / noc-sync / noc-atomic-exit / noc-read-barrier / noc-l1-invalidate / mailbox-sync checkers run over.

Fragility is localized here (per recall-tool-review-lessons): the GCC->clang
translation. Kernels that don't parse are COUNTED and LISTED (a coverage hole is
reported, never silently dropped). Recall is complete only over the kernel
variants the workload actually exercised — stated in the ledger.

Usage:
  python3 capture.py --arch wormhole --log build.log --out OUTDIR \
                     --repo-root /path/to/tt-metal
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXTRACT = os.path.join(HERE, "..", "extractor", "llk_extract")
SHIM = os.path.join(HERE, "..", "out", "sfpi_shim")
EXTRACT_TIMEOUT_SEC = (
    180  # per-kernel extraction cap; expiry -> an EXEC-FAIL ledger entry
)

# The one line tt-metal emits per kernel compile (build.cpp), e.g.
#   ... g++ compile cmd: cd <dir> && <gpp> <flags> -c -o obj src -MF dep <defines>
_CMD_RE = re.compile(r"g\+\+ compile cmd:\s*(.*\S)\s*$")


def _sfpi_gcc_ver(sfpi_root: str) -> str:
    """Discover the sfpi gcc version dir (e.g. 15.1.0) — never hardcoded. Picks the
    highest by NUMERIC version (matching extractor/build.sh's `sort -V`), not lexical
    order (`sorted(['9.5.0','15.1.0'])[-1]` would wrongly give 9.5.0). Fails loudly
    if none is found rather than guessing a path that silently produces wrong facts."""
    base = os.path.join(sfpi_root, "compiler", "lib", "gcc", "riscv-tt-elf")
    vers = [
        os.path.basename(d)
        for d in glob.glob(os.path.join(base, "*"))
        if os.path.isdir(d)
    ]
    if not vers:
        raise RuntimeError(
            f"no sfpi gcc version dir under {base} — is runtime/sfpi present and "
            "matching the pin in tt_metal/sfpi-version?"
        )

    def _key(v: str):
        return tuple(int(p) if p.isdigit() else 0 for p in v.split("."))

    return max(vers, key=_key)


def _clang_base(sfpi_root: str) -> list:
    """The validated GCC->clang translation base (see llk-audit README / memory)."""
    ver = _sfpi_gcc_ver(sfpi_root)
    c = os.path.join(sfpi_root, "compiler")
    return [
        "-x",
        "c++",
        "--target=riscv32-unknown-elf",
        "-D__INT32_TYPE__=long",
        "-std=c++17",
        "-nostdinc++",
        "-nostdinc",
        "-w",
        "-Wno-missing-template-arg-list-after-template-kw",
        "-ferror-limit=0",
        "-isystem",
        os.path.join(c, "lib", "gcc", "riscv-tt-elf", ver, "include"),
        "-isystem",
        os.path.join(c, "riscv-tt-elf", "include"),
        "-isystem",
        os.path.join(c, "riscv-tt-elf", "include", "c++", ver),
        "-isystem",
        os.path.join(c, "riscv-tt-elf", "include", "c++", ver, "riscv-tt-elf"),
        "-isystem",
        os.path.join(sfpi_root, "include"),
        "-I",
        SHIM,
    ]


# The KERNEL surface is not a single substring: JIT/op kernels live under a
# `.../kernels/...` segment, but some model kernel trees use a `<prefix>_kernels/`
# segment (e.g. models/.../unified_kernels/). No single substring matches BOTH
# `/kernels/` and `unified_kernels/` while still EXCLUDING the LLK primitive defs
# under `/ckernels/` (any substring loose enough to catch `unified_kernels/` also
# catches `ckernels/`). The C++ extractor filters by a single `.contains()` substring,
# so we scope it COARSELY (KERNEL_COARSE_SUBSTR, admits all three) and apply the
# PRECISE kernel-surface keep/drop here in Python — no extractor rebuild, and the
# `ckernels/` primitive defs (which would make the checkers flag primitive DEFINITIONS
# as kernel races) never reach the merged fact base. Both sets are overridable.
KERNEL_COARSE_SUBSTR = "kernels/"  # passed to the extractor; a superset pre-scope
KERNEL_SURFACE_KEEP = ("/kernels/", "_kernels/")  # a file is in-surface iff it has one
KERNEL_SURFACE_DROP = ("/ckernels/",)  # ...and none of these (LLK/ckernel defs)


def in_kernel_surface(path: str, keep=KERNEL_SURFACE_KEEP, drop=KERNEL_SURFACE_DROP):
    """True iff `path` is a JIT/op KERNEL source (kept for the merged fact base).

    Keep iff the path contains any KEEP substring AND no DROP substring. `_kernels/`
    matches `unified_kernels/` but NOT `ckernels/` (no underscore) and NOT `/kernels/`
    (no underscore before) — so the two KEEP substrings together cover the kernel
    trees while the DROP guard is belt-and-suspenders against a path that nests both.
    """
    if any(d in path for d in drop):
        return False
    return any(k in path for k in keep)


# HOST implementation / public-API trees. A fact from one of these must NEVER enter
# the analyzed (device-only) base — e.g. the HOST `tt::tt_metal::Semaphore`
# (tt_metal/impl/buffers/semaphore.hpp) shares the bare name of the DEVICE `Semaphore`
# (tt_metal/hw/inc/api/dataflow/noc_semaphore.h), and the checkers match by name only,
# so a host fact would be conflated with a device one. Normally unreachable — the PRIMARY
# guard is cmd_is_device_kernel (jit_build's compile-flag markers) gating TU admission at
# capture, plus a kernel TU can't include host impl headers and the kernels/ +
# in_kernel_surface filters exclude them; is_host_path is a SECONDARY loud tripwire so a
# future scope change can't silently let host code in. Markers are the SPECIFIC host roots (no bare 'impl', so a
# device kernel dir that happens to contain 'impl' isn't misflagged; not
# 'tt_metal/hw/inc/api', which is the DEVICE api) and carry no leading slash so they
# match both relative and absolute fact paths.
HOST_SURFACE_MARKERS = ("tt_metal/impl/", "tt_metal/api/")
# HOST trees whose path CONTAINS a kernels/ segment yet are host code, so they must be
# caught BEFORE the in_kernel_surface yield below. `tt_metal/impl/kernels/` is the host
# `tt::tt_metal::Kernel` / `KernelSource` management classes (kernel.cpp/hpp,
# kernel_source.hpp, kernel_types.cpp) — kernels DIRECTLY under impl/, no subsystem
# segment. DEVICE kernel trees always have an intervening subsystem
# (`impl/dispatch/kernels/`, `impl/buffers/kernels/`, `fabric/impl/kernels/`), so this
# exact substring does not touch them.
HOST_KERNEL_MGMT_MARKERS = ("tt_metal/impl/kernels/",)


def is_host_path(path: str) -> bool:
    """True iff `path` is in a HOST implementation / public-API tree (must not enter
    the device-only base). SECONDARY, path-based cross-check: the PRIMARY device/host
    decision is `cmd_is_device_kernel` (jit_build's own compile-flag markers) applied
    per-TU at capture; this per-fact tripwire is defense-in-depth for a fact whose TU
    slipped through. `in_kernel_surface` WINS in general: a path that is a kernel
    surface (contains `/kernels/`) is DEVICE code even under a host tree — e.g. the JIT
    dispatch/prefetch kernels at `tt_metal/impl/dispatch/kernels/`,
    `tt_metal/impl/buffers/kernels/` — and must NOT be dropped as host. The ONE
    exception is the host Kernel-management tree at `tt_metal/impl/kernels/` (kernels
    directly under impl/), which also contains `/kernels/` but is host — checked first.
    Only a non-kernel host file (e.g. the host `Semaphore` at impl/buffers/semaphore.hpp)
    or that Kernel-management tree is host."""
    if any(h in path for h in HOST_KERNEL_MGMT_MARKERS):
        return True
    if in_kernel_surface(path):
        return False
    return any(h in path for h in HOST_SURFACE_MARKERS)


def tu_ledger_status(all_facts: list, pe: int):
    """The SINGLE per-TU coverage-ledger entry: fold the kernel-surface trim, host-leak
    tripwire, and empty-fact ('nonkernel') conditions into ONE status string + the kept
    facts. Returns (kept, status, nf, host_leak).

    Exactly one row per TU — a host leak must NOT emit a *second* ledger row (that
    double-counts the coverage denominator). A HOST-LEAK is PREPENDED so it LEADS the
    status because (a) bootstrap.sh's HOLE grep is anchored at the '[' (`\\[(...|HOST-LEAK
    |...)`) so the marker must come first to be detected, and (b) a scope tripwire must
    NOT read as a clean 'ok' parse in the `ok/N` headline. The surviving non-host device
    facts are still returned in `kept` so the caller writes them (dropping them would be
    a real recall loss)."""
    kept = [f for f in all_facts if in_kernel_surface(f.get("file", ""))]
    host_leak = [f for f in all_facts if is_host_path(f.get("file", ""))]
    kept = [f for f in kept if not is_host_path(f.get("file", ""))]
    dropped = len(all_facts) - len(kept)
    nf = len(kept)
    status = "ok" if pe == 0 else "ok(parse_errors)"
    if dropped:
        status += f":drop={dropped}"
    if nf == 0:
        status += ":nonkernel"
    if host_leak:
        status = f"HOST-LEAK={len(host_leak)}:{status}"
    return kept, status, nf, host_leak


def _parse_cmd(cmd: str):
    """From a `cd <dir> && <gpp> ...` command, return (cwd, src, [-I..], [-D..])."""
    cwd = None
    m = re.match(r"cd\s+(\S+)\s*&&\s*(.*)", cmd)
    if m:
        cwd, cmd = m.group(1), m.group(2)
    toks = shlex.split(cmd)[1:]  # drop the compiler (argv0)
    incs = [t for t in toks if t.startswith("-I")]
    defs = [t for t in toks if t.startswith("-D")]
    srcs = [t for t in toks if t.endswith((".cc", ".cpp"))]  # the .o is not .cc
    return cwd, (srcs[0] if srcs else None), incs, defs


# The BUILD SYSTEM'S OWN device markers, set env-level by jit_build (build.cpp
# JitBuildEnv::init, so present for EVERY device kernel — Tensix data-movement /
# compute AND ethernet/erisc): the RISC-V toolchain (`riscv-tt-elf-g++`) and
# `-DTENSIX_FIRMWARE`. Host code compiles x86 with NEITHER. Classifying a captured TU
# by these is far more reliable than its file path (the path-based is_host_path
# tripwire is only a fallback): it is jit_build's own record of what it compiled as a
# device kernel, not a name heuristic.
DEVICE_KERNEL_DEFINE = "TENSIX_FIRMWARE"
DEVICE_TOOLCHAIN_MARK = "riscv-tt-elf"


def cmd_is_device_kernel(cmd: str, defs: list) -> bool:
    """True iff this compile command is a DEVICE kernel build. Keyed on EITHER the
    RISC-V toolchain in the command OR `-DTENSIX_FIRMWARE` — both set env-level by
    jit_build for all device kernels. Using EITHER is recall-safe: a real kernel is
    never dropped even if one marker were absent (e.g. a ccache-wrapped command still
    contains the toolchain path); only a command with NEITHER — host code that somehow
    reached the kernel-compile log — is treated as non-kernel."""
    if DEVICE_TOOLCHAIN_MARK in cmd:
        return True
    key = "-D" + DEVICE_KERNEL_DEFINE
    return any(d == key or d.startswith(key + "=") for d in defs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="kernel_tier.capture")
    ap.add_argument(
        "--arch", required=True, choices=("wormhole", "blackhole", "quasar")
    )
    ap.add_argument("--log", required=True, help="tt-metal build log (compile cmds)")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--repo-root", required=True)
    ap.add_argument(
        "--sfpi", default=None, help="sfpi root (default <repo>/runtime/sfpi)"
    )
    ap.add_argument(
        "--path-filter",
        default=KERNEL_COARSE_SUBSTR,
        help="COARSE substring passed to the extractor to pre-scope facts "
        f"(default '{KERNEL_COARSE_SUBSTR}'). The precise kernel-surface keep/drop "
        "(KERNEL_SURFACE_KEEP/DROP) is applied here in Python on top of it — see "
        "in_kernel_surface / kernel_tier/README.md.",
    )
    args = ap.parse_args(argv)

    sfpi = args.sfpi or os.path.join(args.repo_root, "runtime", "sfpi")
    os.makedirs(SHIM, exist_ok=True)
    for stub in ("sfpi.h", "sfpi_classes.h"):
        open(os.path.join(SHIM, stub), "a").close()
    base = _clang_base(sfpi)
    # path-filter scopes the fact base to the KERNEL surface. It must NOT be the repo
    # root: sfpi (STL) lives under runtime/sfpi and the dataflow/LLK primitive
    # DEFINITIONS live under hw/inc/api, hw/ckernels, tt_llk_* — all in-repo — so a
    # repo-root filter floods the base with library internals and the checkers then
    # flag the primitive DEFINITIONS (e.g. Semaphore::inc_multicast's own body) as
    # kernel races. The coarse extractor substring ('kernels/') keeps the JIT kernel
    # dirs + ttnn/models kernel trees (incl. '<prefix>_kernels/') and already drops
    # hw/inc/api + sfpi; it DOES admit hw/ckernels, which in_kernel_surface() trims
    # below (see KERNEL_SURFACE_KEEP/DROP). See kernel_tier/README.md.
    pf = args.path_filter

    cmds = []
    with open(args.log, errors="replace") as fh:
        for ln in fh:
            m = _CMD_RE.search(ln)
            if m:
                cmds.append(m.group(1))
    # dedup identical commands (a kernel re-logged across cores/reconfigure)
    seen, uniq = set(), []
    for c in cmds:
        if c not in seen:
            seen.add(c)
            uniq.append(c)

    os.makedirs(args.out, exist_ok=True)
    facts_path = os.path.join(args.out, f"facts.kernel.{args.arch}.jsonl")
    ledger = []  # (label, status, facts, parse_errors)
    with open(facts_path, "w") as merged:
        for cmd in uniq:
            short = cmd[:60]  # single label width — the skip ledgers can't drift
            # shlex.split (in _parse_cmd) raises ValueError on POSIX-unbalanced
            # quotes — reachable, since build.cpp single-quotes each -D value without
            # escaping. Catch it as a NAMED skip of THIS command, not an unhandled
            # exception that aborts main() and discards EVERY kernel's facts.
            try:
                cwd, src, incs, defs = _parse_cmd(cmd)
            except ValueError:
                ledger.append((short, "SKIP-noparse", 0, 0))
                continue
            if not src or not cwd or not os.path.isdir(cwd):
                ledger.append((short, "SKIP-noparse", 0, 0))
                continue
            # The src may be relative to cwd; if it no longer exists on disk (a
            # stale log vs a cleared cache) the extractor would emit empty output
            # with a non-zero exit — catch it here as a NAMED skip, not a clean ok.
            if not os.path.isfile(os.path.join(cwd, src)):
                ledger.append((short, "SKIP-nosrc", 0, 0))
                continue
            # AUTHORITATIVE device/host classification — gate on jit_build's own device
            # markers (RISC-V toolchain / -DTENSIX_FIRMWARE), NOT the file path. A logged
            # command that is not a device-kernel build (host code that somehow reached
            # the kernel-compile log) is skipped as a NAMED coverage hole, never extracted
            # as a kernel. In practice the log holds only kernel builds, so this is a
            # robust guard rather than a live filter; the path-based is_host_path stays as
            # a SECONDARY per-fact cross-check below. Recall-safe (see cmd_is_device_kernel).
            if not cmd_is_device_kernel(cmd, defs):
                ledger.append((short, "NON-KERNEL-CMD", 0, 0))
                continue
            label = (
                os.path.relpath(cwd, os.path.join(args.repo_root))
                if os.path.isabs(cwd)
                else cwd
            )
            inv = [
                EXTRACT,
                f"--arch={args.arch}",
                f"--path-filter={pf}",
                src,
                "--",
                "clang++",
                *base,
                *incs,
                *defs,
            ]
            try:
                r = subprocess.run(
                    inv,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=EXTRACT_TIMEOUT_SEC,
                )
            except (OSError, subprocess.TimeoutExpired) as e:
                ledger.append((label, f"EXEC-FAIL:{type(e).__name__}", 0, 0))
                continue
            out = r.stdout.strip()
            # The extractor emits its JSON only after the frontend action runs; a
            # non-recoverable failure (arg/driver error, missing source, crash)
            # returns non-zero with empty stdout. Recording that as "ok facts=0"
            # would be a FALSE ALL-CLEAR — indistinguishable from a genuinely clean
            # parse, counted in the "N/N parsed" headline, and (writing a bare
            # newline) defeating bootstrap's empty-fact-base guard. Treat it as a
            # named coverage hole and DO NOT write it to the merged fact base.
            if not out:
                ledger.append((label, f"EMPTY-OUT:rc={r.returncode}", 0, 0))
                continue
            try:
                obj = json.loads(out)
            except json.JSONDecodeError:
                ledger.append((label, "PARSE-FAIL", 0, 0))
                continue
            pe = obj.get("parse_errors", 0)
            # PRECISE kernel-surface trim + device-only host-leak tripwire + nonkernel
            # (empty-fact) handling, folded into EXACTLY ONE ledger row per TU (see
            # tu_ledger_status — a second HOST-LEAK row would double-count the coverage
            # denominator). The coarse extractor filter admits the LLK `ckernels/`
            # primitive defs; the trim keeps only true kernel-source facts so a primitive
            # DEFINITION is never flagged as a kernel race, and its dropped count is
            # reported in the status, never silently swallowed.
            all_facts = obj.get("facts", [])
            kept, status, nf, host_leak = tu_ledger_status(all_facts, pe)
            if host_leak:
                # A HOST-tree fact reaching the extractor output means the scope filter
                # has been loosened (e.g. the host `Semaphore` conflated with the device
                # one). Flag it LOUDLY — the tool is device-only. (Normally impossible
                # under the kernels/ filter; the surviving non-host facts are kept below.)
                print(
                    f"llk-audit capture: WARNING {len(host_leak)} HOST-tree fact(s) "
                    f"leaked into the kernel base (e.g. {host_leak[0].get('file','?')}) "
                    f"— dropped; the tool is DEVICE-ONLY. Check the scope filter.",
                    file=sys.stderr,
                )
            obj["facts"] = kept
            # A TU that contributed NO kernel-surface facts (e.g. a pure-library TU) is
            # still counted as parsed (status leads with "ok" unless host-leaked), but we
            # don't write its empty-fact envelope (it would only dilute the base).
            if nf == 0:
                ledger.append((label, status, 0, pe))
                continue
            merged.write(json.dumps(obj) + "\n")
            ledger.append((label, status, nf, pe))

    # coverage ledger — never a silent cap
    led_path = os.path.join(args.out, f"kernel_coverage.{args.arch}.txt")
    ok = sum(1 for _, s, _, _ in ledger if s.startswith("ok"))
    with open(led_path, "w") as lf:
        lf.write(f"kernel-tier capture ({args.arch}): {ok}/{len(ledger)} TUs parsed\n")
        lf.write("(recall is complete ONLY over these workload-exercised variants)\n\n")
        for label, status, nf, pe in ledger:
            lf.write(f"  [{status:16}] facts={nf:4} parse_errors={pe:3}  {label}\n")
    print(f"kernel-tier: {ok}/{len(ledger)} TUs parsed -> {facts_path}")
    print(f"coverage ledger -> {led_path}")
    print(facts_path)  # last line = the fact-base path (bootstrap consumes it)
    return 0


if __name__ == "__main__":
    sys.exit(main())
