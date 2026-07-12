#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# kernel_tier/bootstrap.sh <arch> <out_dir>
#
# The on-request kernel tier for cb-sync / noc-sync / noc-atomic-exit / noc-read-barrier / mailbox-sync over the
# JIT-compiled kernel surface (OUTSIDE tt-llk). Invoked by `run.sh --full-jit`
# when this module is present (kernel_tier/MANIFEST). It:
#   1. obtains a tt-metal build log carrying `g++ compile cmd:` lines (either a
#      pre-captured LLK_KT_LOG, or by RUNNING a workload with the capture env var),
#   2. translates + extracts each kernel TU into a KERNEL fact base (capture.py),
#   3. runs cb-sync / noc-sync / noc-atomic-exit / noc-read-barrier / mailbox-sync over it,
#   4. writes out/audit.kernel.<arch>.json + a coverage ledger.
#
# Controls (env):
#   LLK_KT_LOG=<file>      use a pre-captured build log (no run; the recommended
#                          path — capture on hardware once, audit offline).
#   LLK_KT_WORKLOAD=<cmd>  the command to run that JIT-compiles the target kernels
#                          (default: the metal_example_eltwise_binary example).
#   LLK_KT_CLEAR_CACHE=1   clear ~/.cache/tt-metal-cache first so the op kernels
#                          RECOMPILE (else already-cached kernels emit no command).
#                          Off by default (non-destructive); if the ledger shows 0
#                          op kernels, set this and re-run.
#
# NOTE: this is periodic-sweep-grade. Recall is complete ONLY over the kernel
# variants the workload actually exercised (stated in the ledger). Needs a runtime
# (WH/BH device or sim) when it runs a workload.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ARCH="${1:?usage: bootstrap.sh <arch> <out_dir>}"
OUT="${2:?usage: bootstrap.sh <arch> <out_dir>}"
TOOL="$(cd "$HERE/.." && pwd)"                 # .../llk-audit
REPO="$(cd "$HERE/../../../../../.." && pwd)"   # repo root
export TT_METAL_HOME="${TT_METAL_HOME:-$REPO}"
mkdir -p "$OUT"
LOG="$OUT/kernel_build.$ARCH.log"

if [ -n "${LLK_KT_LOG:-}" ]; then
  [ -f "$LLK_KT_LOG" ] || { echo "kernel-tier: LLK_KT_LOG '$LLK_KT_LOG' not found" >&2; exit 1; }
  echo "kernel-tier: using pre-captured log $LLK_KT_LOG" >&2
  LOG="$LLK_KT_LOG"
else
  WORKLOAD="${LLK_KT_WORKLOAD:-$REPO/build/programming_examples/metal_example_eltwise_binary}"
  if [ "${LLK_KT_CLEAR_CACHE:-0}" = "1" ]; then
    echo "kernel-tier: clearing tt-metal kernel cache (force recompile+capture) ..." >&2
    # Guard a set-but-EMPTY $HOME (set -u only catches UNSET): an empty $HOME makes
    # the glob "/.cache/..." resolve at filesystem root. Refuse rather than rm there.
    [ -n "$HOME" ] || { echo "kernel-tier: \$HOME is empty — refusing to clear cache" >&2; exit 1; }
    rm -rf "$HOME"/.cache/tt-metal-cache/* 2>/dev/null || true
  else
    echo "kernel-tier: NOT clearing the kernel cache — already-cached kernels emit" >&2
    echo "             no compile command. If the coverage ledger shows 0 op kernels," >&2
    echo "             re-run with LLK_KT_CLEAR_CACHE=1." >&2
  fi
  echo "kernel-tier: capturing compile commands by running: $WORKLOAD" >&2
  # TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 makes tt-metal log the full g++ cmd
  # per kernel. Do NOT set FORCE_JIT_COMPILE (it force-rebuilds dispatch kernels).
  TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 $WORKLOAD > "$LOG" 2>&1 || {
    echo "kernel-tier: workload exited non-zero (build/runtime issue) — see $LOG." >&2
    echo "             Continuing with whatever compile commands were logged." >&2
  }
fi

# `|| true`: under `set -euo pipefail` a capture.py crash would abort HERE (tail
# still exits 0) before the friendly guard below can report the empty fact base.
FACTS="$(PYTHONPATH="$TOOL" python3 "$HERE/capture.py" --arch "$ARCH" --log "$LOG" \
          --out "$OUT" --repo-root "$REPO" | tail -1)" || true
# Gate on a real EXTRACTED FACT, not byte-size: capture.py writes a non-empty
# envelope even for a TU that parsed-with-errors but produced zero facts, so an
# all-errored kernel set would defeat a `-s` check and give a false all-clear.
if [ ! -s "$FACTS" ] || ! grep -q '"family"' "$FACTS"; then
  echo "kernel-tier: no kernel facts extracted — no kernel TUs yielded facts (see the ledger)." >&2
  echo "             Not emitting a false all-clear for cb/noc/read/atomic/mailbox." >&2
  exit 1
fi

AUDIT="$OUT/audit.kernel.$ARCH.json"
PYTHONPATH="$TOOL" python3 -m llkaudit.cli --arch "$ARCH" --facts "$FACTS" \
  --checks cb-sync,noc-sync,noc-atomic-exit,noc-read-barrier,noc-l1-invalidate,mailbox-sync \
  --metal-root "$REPO" > "$AUDIT"

echo "" >&2
echo "=== kernel-tier findings ($ARCH) ===" >&2
PYTHONPATH="$TOOL" python3 - "$AUDIT" <<'PY' >&2
import json,sys,collections
d=json.load(open(sys.argv[1]))
for n,c in d["checks"].items():
    print(f'{n}: {c["count"]} findings  {dict(collections.Counter(f["hint"] for f in c["findings"]))}')
PY
echo "kernel-tier audit -> $AUDIT" >&2
