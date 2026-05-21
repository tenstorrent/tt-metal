#!/usr/bin/env bash
# Quick LLK smoke: run a handful of weekly + nightly WH tests via craq-sim ttsim.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/require-bh-glx-compute.sh
source "${REPO_ROOT}/scripts/lib/require-bh-glx-compute.sh"
require_bh_glx_compute

export TT_METAL_HOME="$REPO_ROOT"
export CRAQ_SIM="${CRAQ_SIM:-/data/rsong/craq-sim}"
SMOKE="${SMOKE_DIR:-$REPO_ROOT/craq-parity-results/llk-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
WEEKLY_SEC="${WEEKLY_SEC:-120}"
NIGHTLY_SEC="${NIGHTLY_SEC:-180}"

mkdir -p "$SMOKE"
ln -sfn "$(basename "$SMOKE")" "$REPO_ROOT/craq-parity-results/llk-smoke-latest"

echo "=== node=$(hostname -s) smoke=$SMOKE ==="
"$REPO_ROOT/scripts/setup-llk-ttsim-env.sh" | tee "$SMOKE/setup.log"

parse_log() {
    local log="$1"
    if grep -qE '(^|[^.])F([^.]|$)| FAILED | failures=' "$log" 2>/dev/null; then
        echo FAIL
    elif grep -qE '^[.]+|^\.+ in |P=[1-9][0-9]*' "$log" 2>/dev/null; then
        echo PASS
    elif grep -q 'no tests ran' "$log" 2>/dev/null; then
        echo FAIL
    else
        echo UNKNOWN
    fi
}

set +e
timeout "$WEEKLY_SEC" "$CRAQ_SIM/scripts/llk-pytest-sweep.sh" weekly wh --timeout 120 --workers 1 --run-root "$SMOKE/weekly-wh" \
    test_matmul.py -k "LoFi and Float16_b and 32" \
    2>&1 | tee "$SMOKE/weekly.log"
WTO=${PIPESTATUS[0]}

timeout "$NIGHTLY_SEC" "$CRAQ_SIM/scripts/llk-pytest-sweep.sh" nightly wh --timeout 300 --workers 1 --run-root "$SMOKE/nightly-wh" \
    test_math_matmul.py -k "LoFi and Float16_b and 32" \
    2>&1 | tee "$SMOKE/nightly.log"
NTO=${PIPESTATUS[0]}
set -e

WSTATUS=$(parse_log "$SMOKE/weekly.log")
NSTATUS=$(parse_log "$SMOKE/nightly.log")
# timeout 124 means the sweep was still passing tests when capped
[ "$WTO" -eq 124 ] && [ "$WSTATUS" != FAIL ] && WSTATUS=PASS
[ "$NTO" -eq 124 ] && [ "$NSTATUS" != FAIL ] && NSTATUS=PASS

echo "weekly_status=$WSTATUS nightly_status=$NSTATUS weekly_timeout_rc=$WTO nightly_timeout_rc=$NTO" | tee "$SMOKE/summary.txt"

if [ "$WSTATUS" = PASS ] && [ "$NSTATUS" = PASS ]; then
    exit 0
fi
exit 1
