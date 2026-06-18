#!/usr/bin/env bash
# Hardware-determinism sweep driver.
#
# Phase 1 (sweep):    run every applicable test file N times each via the
#                     determinism harness, recording a JSONL report.
# Phase 2 (deepdive): re-run only the flaky node ids at a higher N to confirm.
#
# Execution goes through the sanctioned wrapper (.claude/scripts/run_test.sh) —
# never pytest directly. The harness itself lives in conftest.py and is
# activated purely by the TT_LLK_DETERMINISM_RUNS env var exported here.
#
# Usage:
#   ./determinism_sweep.sh sweep    <arch> [N]            # default N=20
#   ./determinism_sweep.sh deepdive <arch> [N] [list]     # default N=1000, list=flaky_nodeids.txt
#
# Examples:
#   ./determinism_sweep.sh sweep wormhole 20
#   python determinism_triage.py "$REPORT"               # -> flaky_nodeids.txt
#   ./determinism_sweep.sh deepdive wormhole 1000
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="$(cd "$HERE/../.." && pwd)"   # repo root (tt-llk), required by run_test.sh
RUN_TEST="$(cd "$HERE/../../.claude/scripts" && pwd)/run_test.sh"
REPORT="${TT_LLK_DETERMINISM_REPORT:-$HERE/determinism_report.jsonl}"
TIMEOUT="${DET_TIMEOUT:-1200}"

CMD="${1:-}"; ARCH="${2:-}"
if [[ -z "$CMD" || -z "$ARCH" ]]; then
    grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 2
fi
[[ -x "$RUN_TEST" ]] || { echo "error: run_test.sh not found/executable at $RUN_TEST" >&2; exit 2; }

# Files applicable to ARCH: drop the other two arches' suffixed files.
applicable_files() {
    local arch="$1" f base
    for f in "$HERE"/test_*.py; do
        base="$(basename "$f")"
        case "$base" in
            *_quasar.py)    [[ "$arch" == quasar ]]    && echo "$base" ;;
            *_blackhole.py) [[ "$arch" == blackhole ]] && echo "$base" ;;
            *_wormhole.py)  [[ "$arch" == wormhole ]]  && echo "$base" ;;
            *)              echo "$base" ;;   # cross-arch
        esac
    done
}

export TT_LLK_DETERMINISM_REPORT="$REPORT"

case "$CMD" in
  sweep)
    N="${3:-20}"
    export TT_LLK_DETERMINISM_RUNS="$N"
    : > "$REPORT"   # fresh report for the triage pass
    mapfile -t FILES < <(applicable_files "$ARCH")
    echo "=== determinism sweep: arch=$ARCH N=$N files=${#FILES[@]} report=$REPORT ==="
    i=0
    for f in "${FILES[@]}"; do
        i=$((i+1))
        echo ">>> [$i/${#FILES[@]}] $f"
        "$RUN_TEST" run --worktree "$WORKTREE" --arch "$ARCH" --test "$f" --timeout "$TIMEOUT" \
            || echo "    (run_test exit $? for $f — continuing sweep)"
    done
    echo "=== sweep done. Triage with: python determinism_triage.py $REPORT ==="
    ;;

  deepdive)
    N="${3:-1000}"
    LIST="${4:-$HERE/flaky_nodeids.txt}"
    [[ -s "$LIST" ]] || { echo "error: flaky list empty/missing: $LIST" >&2; exit 2; }
    export TT_LLK_DETERMINISM_RUNS="$N"
    mapfile -t IDS < <(grep -v '^[[:space:]]*$' "$LIST")
    echo "=== deepdive: arch=$ARCH N=$N variants=${#IDS[@]} report=$REPORT ==="
    i=0
    for id in "${IDS[@]}"; do
        i=$((i+1))
        file="${id%%::*}"
        echo ">>> [$i/${#IDS[@]}] $id"
        "$RUN_TEST" run --worktree "$WORKTREE" --arch "$ARCH" --test "$file" --test-id "$id" --timeout "$TIMEOUT" \
            || echo "    (run_test exit $? for $id — continuing)"
    done
    echo "=== deepdive done. Re-triage: python determinism_triage.py $REPORT ==="
    ;;

  *)
    echo "error: unknown command '$CMD' (expected: sweep | deepdive)" >&2; exit 2 ;;
esac
