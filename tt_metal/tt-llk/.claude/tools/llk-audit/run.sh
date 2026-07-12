#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Orchestrate the LLK race-audit recall tool for one architecture:
#   1. extract a generic fact base from every LLK header (parse once each),
#   2. run the selected Python checkers over the merged fact base.
#
#   run.sh <wormhole|blackhole|quasar> [--checks a,b] [--changed [BASE]] [--full-jit] [out_dir]
#
# Produces <out_dir>/facts.<arch>.jsonl (raw fact base) and
#          <out_dir>/audit.<arch>.json  (advisory findings). Prints a summary.
set -euo pipefail
cd "$(dirname "$0")"
HERE="$PWD"
EXTRACT="$HERE/extractor/llk_extract"
LLK_ROOT="$(cd ../../.. && pwd)"            # tt_metal/tt-llk
METAL_DIR="$(cd "$LLK_ROOT"/../.. && pwd)"  # repo root (also used for git in --changed)

# --- Kernel-tier (JIT: cb-sync / noc-sync / noc-atomic-exit / noc-read-barrier / mailbox-sync) capability probe ------
# The kernel-tier module (cb-sync + noc-sync + mailbox-sync checkers over the
# captured kernels + capture.py, which scrapes JIT compile commands from a build
# log) reaches the JIT-compiled kernel surface that lives OUTSIDE tt-llk. It is
# committed IN-TREE under kernel_tier/ (a kernel_tier/MANIFEST marks it present),
# but it is OPT-IN: it runs only under --full-jit, never on the default path, and
# even then it needs a build log or a runtime (device/sim) to capture kernels
# (see kernel_tier/README.md). This probe just reports whether the module is
# present. The fragile part (the GCC->clang translation) is isolated in capture.py
# with an honest coverage ledger; run.sh never captures automatically.
KERNEL_TIER_DIR="$HERE/kernel_tier"
kernel_tier_available() { [ -f "$KERNEL_TIER_DIR/MANIFEST" ]; }

if [ "${1:-}" = "--kernel-tier-status" ]; then
  if kernel_tier_available; then
    echo "available"
    echo "kernel-tier module present at $KERNEL_TIER_DIR" >&2
  else
    echo "unavailable"
    echo "kernel-tier module missing (no kernel_tier/MANIFEST) — it is normally" >&2
    echo "committed in-tree; restore kernel_tier/ from the repo. See" >&2
    echo ".claude/tools/llk-audit/kernel_tier/README.md." >&2
  fi
  exit 0
fi

# Auto-build the C++ extractor on first run (or if the source is newer than the
# binary). The binary is a git-ignored build artifact, so a fresh checkout won't
# have it. If no suitable Clang/LLVM (>=18) is available the build fails; we then
# exit non-zero so the calling /*-audit skill falls back to its manual method
# (its "Recall preflight" says: if unbuilt, proceed manually — absence != "no findings").
if [ ! -x "$EXTRACT" ] || [ "$HERE/extractor/llk_extract.cpp" -nt "$EXTRACT" ]; then
  echo "llk-audit: extractor not built (or stale) — building once ..." >&2
  if ! "$HERE/extractor/build.sh" >&2; then
    echo "llk-audit: auto-build failed (need Clang/LLVM >= 18 dev libs). The audit" >&2
    echo "           skill should proceed with its manual method." >&2
    exit 1
  fi
fi

ARCH="${1:?usage: run.sh <wormhole|blackhole|quasar> [--checks a,b] [--changed [BASE]] [--full-jit] [out_dir]}"; shift || true
CHECKS="all"
OUT="$HERE/out"
CHANGED=0
CHANGED_BASE="main"
FULLJIT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --checks)
      # Require a real value (not the next flag, not missing) so `--checks` does
      # not swallow `--changed` or crash on an unbound $2.
      if [ $# -lt 2 ] || [ "${2#-}" != "$2" ]; then
        echo "run.sh: --checks needs a value (comma-separated names or 'all')" >&2; exit 2
      fi
      CHECKS="$2"; shift 2;;
    --changed) CHANGED=1; shift
               # An optional next token is the diff BASE — but ONLY if it is a
               # real git ref. Otherwise it is the out_dir; swallowing it would
               # set a bogus base -> empty diff -> every finding scoped away
               # (a silent false all-clear).
               if [ $# -gt 0 ] && [ "${1#-}" = "$1" ] && \
                  git -C "$METAL_DIR" rev-parse --verify --quiet "$1^{commit}" >/dev/null 2>&1; then
                 CHANGED_BASE="$1"; shift
               fi;;
    --full-jit) FULLJIT=1; shift;;   # also run the opt-in kernel tier (cb/noc/read/atomic/mailbox), if built
    --*) echo "run.sh: unknown option '$1'" >&2; exit 2;;
    *) OUT="$1"; shift;;
  esac
done
mkdir -p "$OUT"

SFPI="$LLK_ROOT/tests/sfpi"

# SFPU vector code (sfpi_classes.h) uses GCC vector extensions clang rejects and
# is not where the modeled hazards live; stub it so headers that merely #include
# it parse. Files that structurally use sfpi:: types fail to parse and are logged
# (an explicit coverage gap — see parse.log / parse_errors in the output).
SHIM="$OUT/sfpi_shim"; mkdir -p "$SHIM"; : > "$SHIM/sfpi.h"; : > "$SHIM/sfpi_classes.h"

TGT=riscv32-unknown-elf  # same clang target triple for every arch
case "$ARCH" in
  wormhole)  DEF=ARCH_WORMHOLE; LLK=tt_llk_wormhole_b0; BOOT="-DLLK_BOOT_MODE_BRISC"
             ARCHINC=("-I$METAL_DIR/tt_metal/hw/inc/internal/tt-1xx/wormhole" "-I$METAL_DIR/tt_metal/hw/inc/internal/tt-1xx/wormhole/wormhole_b0_defines" "-I$METAL_DIR/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api");;
  blackhole) DEF=ARCH_BLACKHOLE; LLK=tt_llk_blackhole; BOOT="-DLLK_BOOT_MODE_BRISC"
             ARCHINC=("-I$METAL_DIR/tt_metal/hw/inc/internal/tt-1xx/blackhole" "-I$METAL_DIR/tt_metal/hw/ckernels/blackhole/metal/llk_api");;
  quasar)    DEF=ARCH_QUASAR; LLK=tt_llk_quasar; BOOT="-DLLK_BOOT_MODE_TRISC -DLLK_TRISC_ISOLATE_SFPU"
             ARCHINC=("-I$METAL_DIR/tt_metal/hw/inc/internal/tt-2xx/quasar" "-I$METAL_DIR/tt_metal/hw/ckernels/quasar/metal/llk_api");;
  *) echo "unknown arch $ARCH" >&2; exit 1;;
esac

# sfpi-gcc version whose system-header dirs we add (single source of truth).
# Discover the sfpi gcc version (highest by version, like build.sh's `sort -V` and
# capture.py's _sfpi_gcc_ver) rather than hardcode it — a moved pin otherwise makes
# clang silently miss the STL/builtin -isystem dirs. Override with SFPI_GCC_VER.
GCC_VER="${SFPI_GCC_VER:-}"
if [ -z "$GCC_VER" ]; then
  GCC_VER="$(find "$SFPI/compiler/lib/gcc/riscv-tt-elf" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort -V | tail -1 || true)"
fi
[ -n "$GCC_VER" ] || { echo "llk-audit: no sfpi gcc version under $SFPI/compiler/lib/gcc/riscv-tt-elf (set SFPI_GCC_VER)" >&2; exit 1; }
# -ferror-limit=0: never stop after N errors (clang default ~20), which would
# truncate a partially-parsing header mid-TU and silently drop every fact after the
# cutoff — a partial false-all-clear. Matches kernel_tier/capture.py.
FLAGS=(-x c++-header --target=$TGT -ferror-limit=0 -D__INT32_TYPE__=long
  -DENV_LLK_INFRA -D$DEF -DTENSIX_FIRMWARE -DCOMPILE_FOR_TRISC -std=c++17 -nostdinc++ -nostdinc
  -DLLK_TRISC_UNPACK -DLLK_TRISC_MATH -DLLK_TRISC_PACK -DENABLE_LLK_ASSERT -DRUNTIME_FORMATS $BOOT
  -isystem "$SFPI/compiler/lib/gcc/riscv-tt-elf/$GCC_VER/include/"
  -isystem "$SFPI/compiler/riscv-tt-elf/include"
  -isystem "$SFPI/compiler/riscv-tt-elf/include/c++/$GCC_VER"
  -isystem "$SFPI/compiler/riscv-tt-elf/include/c++/$GCC_VER/riscv-tt-elf"
  -isystem "$SFPI/include"
  -I"$SHIM"
  -I"$LLK_ROOT/common" -I"$LLK_ROOT/$LLK/common/inc" -I"$LLK_ROOT/$LLK/common/inc/sfpu"
  -I"$LLK_ROOT/$LLK/llk_lib" -I"$LLK_ROOT/tests/helpers/include" -I"$METAL_DIR/tt_metal/hw/inc/"
  "${ARCHINC[@]}")

mapfile -t HEADERS < <(find "$LLK_ROOT/$LLK/llk_lib" "$LLK_ROOT/$LLK/common/inc" -name '*.h' | sort)
echo "Extracting fact base from ${#HEADERS[@]} headers for $ARCH ..." >&2

FACTS="$OUT/facts.$ARCH.jsonl"; : > "$FACTS"
: > "$OUT/parse.log"
FAIL=0
for h in "${HEADERS[@]}"; do
  if ! "$EXTRACT" --arch="$ARCH" "$h" -- clang++ "${FLAGS[@]}" >>"$FACTS" 2>>"$OUT/parse.log"; then
    FAIL=$((FAIL+1))
  fi
done
[ "$FAIL" -eq 0 ] || echo "($FAIL header(s) failed to parse — see $OUT/parse.log; expected for SFPU-heavy files)" >&2

# Empty-fact-base floor: if NOTHING parsed, the run is not a clean audit — it is
# a broken toolchain. Refuse to emit a false all-clear (0 findings, exit 0);
# exit non-zero like the build-failure path so the calling skill falls back to
# its manual method. (Some SFPU-heavy failures are expected; ALL failing is not.)
# Gate on a real EXTRACTED FACT, not byte-size: the extractor writes a
# `{"facts":[],...}` envelope per header even on a failed parse, so an all-fail run
# still yields a non-empty file — `-s` alone would let that emit 0 findings/exit 0
# (a false all-clear). `"family"` appears in every real fact and in no empty envelope.
if [ ! -s "$FACTS" ] || ! grep -q '"family"' "$FACTS"; then
  echo "llk-audit: NO facts extracted ($FAIL/${#HEADERS[@]} headers failed to parse)." >&2
  echo "           This is a broken run, NOT a clean audit — refusing a false" >&2
  echo "           all-clear. Check the toolchain / SFPI include paths ($OUT/parse.log)." >&2
  exit 1
fi

# Diff-scoped mode: collect the changed LLK headers (committed vs BASE + working
# tree) so the CLI can filter findings to those touching a changed file. The
# whole tree is still parsed above (cross-file context); only OUTPUT is scoped.
CHANGED_ARG=()
if [ "$CHANGED" -eq 1 ]; then
  # Validate the diff base resolves — INCLUDING the default "main". On a shallow/
  # detached checkout with no local 'main', `git diff main...HEAD` fails silently
  # (2>/dev/null below) -> empty changed-set -> __none__ -> every finding scoped
  # away -> 0 findings, exit 0: a false all-clear. Refuse instead.
  if ! git -C "$METAL_DIR" rev-parse --verify --quiet "$CHANGED_BASE^{commit}" >/dev/null 2>&1; then
    echo "llk-audit: --changed base '$CHANGED_BASE' does not resolve to a commit in" >&2
    echo "           this checkout (shallow/detached?). Refusing to silently scope to" >&2
    echo "           nothing (a false all-clear). Pass an explicit --changed <ref> that" >&2
    echo "           exists here (e.g. origin/main, or a fetched base)." >&2
    exit 1
  fi
  mapfile -t CHG < <(
    { git -C "$METAL_DIR" diff --name-only "$CHANGED_BASE"...HEAD 2>/dev/null
      git -C "$METAL_DIR" diff --name-only 2>/dev/null
      git -C "$METAL_DIR" diff --name-only --cached 2>/dev/null
    } | grep -E "tt_metal/tt-llk/$LLK/.*\.h$" | sort -u)
  echo "Diff-scoped vs '$CHANGED_BASE': ${#CHG[@]} changed $LLK header(s)." >&2
  if [ "${#CHG[@]}" -gt 0 ]; then
    CHANGED_ARG=(--changed-files "$(IFS=,; echo "${CHG[*]}")")
  else
    CHANGED_ARG=(--changed-files "__none__")  # scope to nothing -> empty findings
  fi
fi

AUDIT="$OUT/audit.$ARCH.json"
PYTHONPATH="$HERE" python3 -m llkaudit.cli --arch "$ARCH" --facts "$FACTS" \
  --checks "$CHECKS" --metal-root "$METAL_DIR" "${CHANGED_ARG[@]}" > "$AUDIT"

# Summary.
PYTHONPATH="$HERE" python3 - "$AUDIT" "$FACTS" <<'PY'
import json,sys,collections
audit,facts=sys.argv[1],sys.argv[2]
d=json.load(open(audit))
print(f'\n=== llk-audit [{d["arch"]}]  (authority: {d["authority"]}, parse_errors: {d["parse_errors"]}) ===')
for name,c in d["checks"].items():
    hb=collections.Counter(f["hint"] for f in c["findings"])
    print(f'\n{name}: {c["count"]} findings  {dict(hb)}')
# Surface DEGRADED prominently: a degraded run (empty fact base, unreadable cfg
# defines, or a --changed file that parsed to 0 facts) produces "0 findings" that
# must NOT read as a clean all-clear. cli.py records these in the JSON envelope;
# echo them here so the human-facing summary can't hide a non-analyzed run.
deg = d.get("degraded") or []
if deg:
    print("\n*** DEGRADED — NOT a clean all-clear (some surface was NOT analyzed): ***")
    for note in deg:
        print(f'  ! {note}')
print(f'\nfact base -> {facts}')
print(f'findings  -> {audit}')
PY

# --- Opt-in kernel tier (cb-sync / noc-sync / noc-atomic-exit / noc-read-barrier / mailbox-sync over JIT kernels) ----
# The in-tree audit above always runs. --full-jit additionally RUNS the committed
# kernel tier (kernel_tier/bootstrap.sh); it never captures automatically — the
# capture needs a build log or a runtime, supplied via bootstrap.sh's env controls.
# If the module is missing (partial checkout) it DEGRADES HONESTLY: it says so and
# names the classes left uncovered — a clean result must never read as
# "cb/noc/read/atomic/mailbox covered". See kernel_tier/README.md + the race-audit-all runbook.
if [ "$FULLJIT" -eq 1 ]; then
  echo "" >&2
  echo "=== kernel tier (JIT: cb-sync / noc-sync / noc-atomic-exit / noc-read-barrier / mailbox-sync) ===" >&2
  if kernel_tier_available; then
    echo "kernel-tier module present — running capture + kernel checks ..." >&2
    if ! "$KERNEL_TIER_DIR/bootstrap.sh" "$ARCH" "$OUT" >&2; then
      echo "kernel-tier run FAILED — cb-sync / noc-sync / noc-atomic-exit / noc-read-barrier / mailbox-sync NOT covered this run." >&2
      # Propagate the failure to the EXIT CODE. bootstrap.sh refuses an empty/failed
      # capture (exit non-zero); if run.sh swallowed that and exited 0, a caller doing
      # `run.sh --full-jit && ...` would read cb/noc/read/atomic/mailbox as covered — a false
      # all-clear at the exit-code level. Match every other failure path (exit 1).
      exit 1
    fi
  else
    echo "kernel-tier module is MISSING (no kernel_tier/), so cb-sync / noc-sync /" >&2
    echo "mailbox-sync were NOT TOOL-RECALLED here (their kernel surface is JIT-" >&2
    echo "compiled kernels outside tt-llk). This is NOT 'not audited': the" >&2
    echo "race-audit-all skill still audits them LLM-driven (each skill's ttnn-" >&2
    echo "widened grep + reasoning + ISA docs) — you only forgo the extra" >&2
    echo "deterministic candidate list. Restore kernel_tier/ (it is committed" >&2
    echo "in-tree) — see .claude/tools/llk-audit/kernel_tier/README.md." >&2
  fi
fi
