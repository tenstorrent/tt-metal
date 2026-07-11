#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Orchestrate the LLK race-audit recall tool for one architecture:
#   1. extract a generic fact base from every LLK header (parse once each),
#   2. run the selected Python checkers over the merged fact base.
#
#   run.sh <wormhole|blackhole|quasar> [--checks a,b] [out_dir]
#
# Produces <out_dir>/facts.<arch>.jsonl (raw fact base) and
#          <out_dir>/audit.<arch>.json  (advisory findings). Prints a summary.
set -euo pipefail
cd "$(dirname "$0")"
HERE="$PWD"
EXTRACT="$HERE/extractor/llk_extract"

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

ARCH="${1:?usage: run.sh <wormhole|blackhole|quasar> [--checks a,b] [--changed [BASE]] [out_dir]}"; shift || true
CHECKS="all"
OUT="$HERE/out"
CHANGED=0
CHANGED_BASE="main"
while [ $# -gt 0 ]; do
  case "$1" in
    --checks) CHECKS="$2"; shift 2;;
    --changed) CHANGED=1; shift
               # optional non-flag next token is the diff base (default: main)
               if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then CHANGED_BASE="$1"; shift; fi;;
    *) OUT="$1"; shift;;
  esac
done
mkdir -p "$OUT"

LLK_ROOT="$(cd ../../.. && pwd)"           # tt_metal/tt-llk
METAL_DIR="$(cd "$LLK_ROOT"/../.. && pwd)"  # repo root
SFPI="$LLK_ROOT/tests/sfpi"

# SFPU vector code (sfpi_classes.h) uses GCC vector extensions clang rejects and
# is not where the modeled hazards live; stub it so headers that merely #include
# it parse. Files that structurally use sfpi:: types fail to parse and are logged
# (an explicit coverage gap — see parse.log / parse_errors in the output).
SHIM="$OUT/sfpi_shim"; mkdir -p "$SHIM"; : > "$SHIM/sfpi.h"; : > "$SHIM/sfpi_classes.h"

case "$ARCH" in
  wormhole)  DEF=ARCH_WORMHOLE; LLK=tt_llk_wormhole_b0; TGT=riscv32-unknown-elf; BOOT="-DLLK_BOOT_MODE_BRISC"
             ARCHINC=("-I$METAL_DIR/tt_metal/hw/inc/internal/tt-1xx/wormhole" "-I$METAL_DIR/tt_metal/hw/inc/internal/tt-1xx/wormhole/wormhole_b0_defines" "-I$METAL_DIR/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api");;
  blackhole) DEF=ARCH_BLACKHOLE; LLK=tt_llk_blackhole; TGT=riscv32-unknown-elf; BOOT="-DLLK_BOOT_MODE_BRISC"
             ARCHINC=("-I$METAL_DIR/tt_metal/hw/inc/internal/tt-1xx/blackhole" "-I$METAL_DIR/tt_metal/hw/ckernels/blackhole/metal/llk_api");;
  quasar)    DEF=ARCH_QUASAR; LLK=tt_llk_quasar; TGT=riscv32-unknown-elf; BOOT="-DLLK_BOOT_MODE_TRISC -DLLK_TRISC_ISOLATE_SFPU"
             ARCHINC=("-I$METAL_DIR/tt_metal/hw/inc/internal/tt-2xx/quasar" "-I$METAL_DIR/tt_metal/hw/ckernels/quasar/metal/llk_api");;
  *) echo "unknown arch $ARCH" >&2; exit 1;;
esac

FLAGS=(-x c++-header --target=$TGT -D__INT32_TYPE__=long
  -DENV_LLK_INFRA -D$DEF -DTENSIX_FIRMWARE -DCOMPILE_FOR_TRISC -std=c++17 -nostdinc++ -nostdinc
  -DLLK_TRISC_UNPACK -DLLK_TRISC_MATH -DLLK_TRISC_PACK -DENABLE_LLK_ASSERT -DRUNTIME_FORMATS $BOOT
  -isystem "$SFPI/compiler/lib/gcc/riscv-tt-elf/15.1.0/include/"
  -isystem "$SFPI/compiler/riscv-tt-elf/include"
  -isystem "$SFPI/compiler/riscv-tt-elf/include/c++/15.1.0"
  -isystem "$SFPI/compiler/riscv-tt-elf/include/c++/15.1.0/riscv-tt-elf"
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

# Diff-scoped mode: collect the changed LLK headers (committed vs BASE + working
# tree) so the CLI can filter findings to those touching a changed file. The
# whole tree is still parsed above (cross-file context); only OUTPUT is scoped.
CHANGED_ARG=()
if [ "$CHANGED" -eq 1 ]; then
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
print(f'\nfact base -> {facts}')
print(f'findings  -> {audit}')
PY
