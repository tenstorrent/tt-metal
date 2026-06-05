#!/bin/bash
# precompile_verify.sh — Robustness R1: a --precompile run must produce IDENTICAL results to a cold
# run (same passed/failed/skipped/xfailed, test-for-test). The warm cache holds content-hashed
# kernels that are byte-identical to what a cold run compiles, so results MUST match; this proves it.
#
# Usage: scripts/precompile_verify.sh <pytest target/args...>
#   COLD_FIRST=0  run --precompile first (default 1 = cold first)
#
# Prints both result lines + a PASS/FAIL verdict. Logs in /tmp/verify_{cold,pre}.log.
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
TGT=("$@")
COLD_FIRST="${COLD_FIRST:-1}"

_counts() { grep -oE '[0-9]+ (passed|failed|skipped|xfailed|error)' "$1" 2>/dev/null | sort | tr '\n' ' '; }

run_cold() {
    echo "=== COLD (standard inline) ===" >&2
    PYTHONPATH="$REPO_DIR" scripts/run_safe_pytest.sh --run-all "${TGT[@]}" > /tmp/verify_cold.log 2>&1
    echo "cold exit=$?" >&2
}
run_pre() {
    echo "=== PRECOMPILE ===" >&2
    PYTHONPATH="$REPO_DIR" scripts/run_safe_pytest.sh --precompile --run-all "${TGT[@]}" > /tmp/verify_pre.log 2>&1
    echo "precompile exit=$?" >&2
}

if [[ "$COLD_FIRST" == 1 ]]; then run_cold; run_pre; else run_pre; run_cold; fi

COLD="$(_counts /tmp/verify_cold.log)"
PRE="$(_counts /tmp/verify_pre.log)"
echo ""
echo "target:     ${TGT[*]}"
echo "cold:       $COLD"
echo "precompile: $PRE"
grep -E "PRECOMPILE: (✓|✗)" /tmp/verify_pre.log | sed 's/^/  /'
if [[ "$COLD" == "$PRE" && -n "$COLD" ]]; then
    echo "R1 VERDICT: ✓ PASS — precompiled run is identical to cold"
    exit 0
else
    echo "R1 VERDICT: ✗ FAIL — results differ (cold='$COLD' vs precompile='$PRE')"
    exit 1
fi
